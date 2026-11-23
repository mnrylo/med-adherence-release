"""Staged sweep of E1 and E2 (conf/experiment_plan.json): run, resume, select, index.

    python -m thesis.sweep                 # run / resume
    python -m thesis.sweep --dry-run       # list the work
    python -m thesis.sweep --index-only    # rebuild index/runs.csv

Order: E1-A, E2-A, E1-B, E2-B, E1-C, E2-C, then the finalists of both experiments.

  A  signal: fs × window × overlap × signals × filter with cnn_bilstm, default hyperparameters
  B  model: cnn / cnn_lstm / cnn_bilstm on the 5 best signals of A
  C  hyperparameters: the grid on the 3 best (model, signal) of B
  F  finalists: the 3 best of C (default hyperparameters included as candidates)

Replicates: screening stages (A, C) train 3 folds × 1 seed; B and the finalists add seeds 1 and 2,
so every number that is compared in the thesis tables is a mean over 3 folds × 3 seeds.

Every choice uses the validation set only: mean over the configuration's replicates of
val.timeline.seg_f1_25, ties broken by val.window.macro_f1 (common_protocol.selection_rule).
Training runs are started as child processes; ``--capacity`` sets how many GPU slots are used
at once (configurations at 100 Hz or with >= 250-sample windows take two slots).
Each decision is written to index/selection/<experiment>_<stage>.json. Finished replicates are
skipped, so the sweep resumes after an interruption; index/runs.csv is rebuilt after every
configuration.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from thesis import common as C

CAPACITY = 2
SCREEN_SEEDS = [0]
FULL_SEEDS = [0, 1, 2]
FOLDS = [0, 1, 2]
N_TOP = {"A": 5, "B": 3, "C": 3}
SCORE, TIE = "val.timeline.seg_f1_25", "val.window.macro_f1"
LOG = C.INDEX / "sweep_log.txt"
PROGRESS = C.INDEX / "sweep_progress.json"


def log(msg):
    """Print a time-stamped line and append it to index/sweep_log.txt."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def flatten(m: dict) -> dict:
    """Scalar metrics of a metrics.json as flat 'set.level.metric' keys."""
    from thesis import metrics as M
    return M.flatten(m)


def build_index() -> pd.DataFrame:
    """Rebuild index/runs.csv from every replicate manifest and metrics.json."""
    rows = []
    for man_f in sorted(C.RUNS.glob("*/*/f*_s*/manifest.json")):
        if man_f.parent.name.startswith("fall"):
            continue
        man = json.loads(man_f.read_text())
        row = {"run_id": man["run_id"], "experiment": man["experiment"], "config_key": man["config_key"],
               "replicate": man["replicate"], **man["factors"], "status": man["status"],
               "n_params": man.get("n_params"), "duration_s": man.get("duration_s"),
               "path": str(man_f.parent.relative_to(C.ROOT)), "git_commit": man.get("git_commit"),
               "created": man.get("created")}
        mf = man_f.parent / "metrics.json"
        if man["status"] == "done" and mf.exists():
            row.update(flatten(json.loads(mf.read_text())))
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        C.INDEX.mkdir(exist_ok=True)
        tmp = C.INDEX / "runs.csv.tmp"
        df.to_csv(tmp, index=False)
        tmp.replace(C.INDEX / "runs.csv")
    return df


def config_scores(df: pd.DataFrame, experiment: str, keys: list[str]) -> pd.DataFrame:
    """Mean (std) selection score and tie-break per configuration, best first."""
    d = df[(df.experiment == experiment) & (df.status == "done") & df.config_key.isin(keys)]
    g = d.groupby("config_key").agg(score=(SCORE, "mean"), score_std=(SCORE, "std"), tie=(TIE, "mean"),
                                    n=(SCORE, "size"))
    return g.sort_values(["score", "tie"], ascending=False)


def stage_a() -> list[str]:
    """Stage-A configuration keys (signal factors, CNN-BiLSTM, default hyperparameters)."""
    f = C.EXPERIMENTS["E1"]["factors"]
    return [C.config_key({"model": "cnn_bilstm", "fs_hz": fs, "window_s": w, "overlap": ov, "signals": s,
                          "filter": fl, "hp": "hpdef"})
            for fs, w, ov, s, fl in itertools.product(f["fs_hz"], f["window_s"], f["overlap"], f["signals"],
                                                      f["filter"])]


def with_(key: str, **kw) -> str:
    """Config key with some factors replaced."""
    return C.config_key({**C.parse_key(key), **kw})


def stage_b(top_a: list[str]) -> list[str]:
    """Stage-B keys: every model on the given signals."""
    return [with_(k, model=m) for k in top_a for m in C.EXPERIMENTS["E1"]["factors"]["model"]]


def stage_c(top_b: list[str]) -> list[str]:
    """Stage-C keys: the hyperparameter grid on the given configurations (no LSTM duplicates for the CNN)."""
    out = []
    pts = C.hp_points()
    first_lstm = C.HP_GRID["lstm_units"][0]
    for k in top_b:
        model = C.parse_key(k)["model"]
        for i, p in enumerate(pts):
            if model == "cnn" and p["lstm_units"] != first_lstm:
                continue
            out.append(with_(k, hp=f"hp{i:02d}"))
    return out


def select(df, experiment, stage, candidates, n_top, note="") -> list[str]:
    """Write a selection file and return the n_top best configurations."""
    sc = config_scores(df, experiment, candidates)
    top = list(sc.index[:n_top])
    C.write_json(C.SELECTION / f"{experiment}_{stage}.json", {
        "experiment": experiment, "stage": stage, "created": C.now(),
        "rule": f"mean {SCORE} over replicates (validation), tie-break {TIE}", "note": note,
        "n_candidates": len(candidates), "n_scored": len(sc), "advanced": top,
        "ranking": [{"config_key": k, "score": r.score, "score_std": r.score_std, "tie_break": r.tie, "n_runs": int(r.n)}
                    for k, r in sc.iterrows()]})
    log(f"{experiment}-{stage}: advanced {top}")
    return top


def cost(key: str) -> int:
    """GPU slots of a configuration (2 at 100 Hz or with >= 250-sample windows)."""
    c = C.parse_key(key)
    return 2 if c["fs_hz"] >= 100 or c["window_s"] * c["fs_hz"] >= 250 else 1


def replicate_done(experiment, key, fold, seed) -> bool:
    """True when the replicate's manifest says done."""
    p = C.config_dir(experiment, key) / C.replicate_name(fold, seed) / "manifest.json"
    return p.exists() and json.loads(p.read_text()).get("status") == "done"


def pending(experiment, key, seeds) -> bool:
    """True when a replicate of the configuration is not done."""
    return not all(replicate_done(experiment, key, f, s) for f in FOLDS for s in seeds)


def run_jobs(jobs: list[tuple[str, str, list[int]]], label: str, retries: int = 1):
    """jobs: (experiment, key, seeds). Runs train_config.py children with CAPACITY GPU slots."""
    todo = [j for j in jobs if pending(*j)]
    log(f"{label}: {len(todo)} of {len(jobs)} configurations to train")
    attempts: dict = {}
    running: list[tuple[subprocess.Popen, tuple, int]] = []
    done_n, t0 = len(jobs) - len(todo), time.time()
    while todo or running:
        for p, j, c in list(running):
            if p.poll() is not None:
                running.remove((p, j, c))
                if p.returncode != 0:
                    attempts[j[:2]] = attempts.get(j[:2], 0) + 1
                    log(f"  FAILED ({p.returncode}) {j[0]} {j[1]} attempt {attempts[j[:2]]}")
                    if attempts[j[:2]] <= retries:
                        todo.append(j)
                else:
                    done_n += 1
                build_index()
                _progress(label, done_n, len(jobs), t0, [r[1] for r in running])
        used = sum(c for _, _, c in running)
        started = False
        for j in list(todo):
            c = cost(j[1])
            if used + c <= CAPACITY:
                todo.remove(j)
                cmd = C.module_cmd("thesis.train_config", "--experiment", j[0], "--key", j[1],
                                   "--folds", *FOLDS, "--seeds", *j[2])
                out = C.config_dir(j[0], j[1])
                out.mkdir(parents=True, exist_ok=True)
                fh = open(out / "stdout.txt", "a")
                env = C.child_env(PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                running.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), j, c))
                used += c
                started = True
                log(f"  start {j[0]} {j[1]} seeds {j[2]}")
                _progress(label, done_n, len(jobs), t0, [r[1] for r in running])
        if not started:
            time.sleep(5)
    build_index()


def _progress(label, done, total, t0, running):
    """Write index/sweep_progress.json."""
    el = time.time() - t0
    C.write_json(PROGRESS, {"stage": label, "done": done, "total": total, "elapsed_h": el / 3600,
                            "running": [f"{j[0]} {j[1]}" for j in running], "updated": C.now()})


def main():
    """Run (or resume) the staged sweep of E1 and E2."""
    global CAPACITY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--experiments", nargs="+", default=["E1", "E2"])
    ap.add_argument("--capacity", type=int, default=CAPACITY, help="concurrent GPU slots")
    a = ap.parse_args()
    CAPACITY = a.capacity
    if a.index_only:
        df = build_index()
        print(f"{len(df)} runs indexed")
        return
    exps = a.experiments
    A = stage_a()
    if a.dry_run:
        print(f"stage A: {len(A)} configurations per experiment × {len(FOLDS) * len(SCREEN_SEEDS)} replicates")
        print(f"stage B: {len(stage_b(A[:N_TOP['A']]))} configurations × {len(FOLDS) * len(FULL_SEEDS)} replicates")
        print(f"stage C: up to {len(stage_c(A[:N_TOP['B']]))} configurations × {len(FOLDS) * len(SCREEN_SEEDS)}")
        return
    log(f"sweep start: {exps}")
    for e in exps:
        run_jobs([(e, k, SCREEN_SEEDS) for k in A], f"{e}-A")
    top_a = {e: select(build_index(), e, "A", A, N_TOP["A"]) for e in exps}
    B = {e: stage_b(top_a[e]) for e in exps}
    for e in exps:
        run_jobs([(e, k, FULL_SEEDS) for k in B[e]], f"{e}-B")
    df = build_index()
    top_b = {e: select(df, e, "B", B[e], N_TOP["B"], note="3 folds × 3 seeds") for e in exps}
    Cc = {e: stage_c(top_b[e]) for e in exps}
    for e in exps:
        run_jobs([(e, k, SCREEN_SEEDS) for k in Cc[e]], f"{e}-C")
    df = build_index()
    df0 = df[df.replicate.str.endswith("_s0")]
    final = {e: select(df0, e, "C", Cc[e] + top_b[e], N_TOP["C"],
                       note="candidates: the hyperparameter grid plus the default hyperparameters of each B finalist; "
                            "scored on seed 0") for e in exps}
    for e in exps:
        run_jobs([(e, k, FULL_SEEDS) for k in final[e]], f"{e}-F")
    df = build_index()
    for e in exps:
        select(df, e, "F", final[e], N_TOP["C"], note="finalists with 3 folds × 3 seeds; these go to E3")
    log("sweep finished")


if __name__ == "__main__":
    main()
