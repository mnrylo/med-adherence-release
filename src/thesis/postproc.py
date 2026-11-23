"""E3: temporal post-processing (HMM, HSMM) and the intake decision on the E1/E2 finalists.

    python -m thesis.postproc [--workers 6] [--force]

Reads only saved files (the finalists' window probabilities, artifacts/ and train_info.json):
nothing is retrained. Every choice is made on the validation streams; the blind test sets are
only measured.

Decoding grid. Window probabilities are averaged onto the 10 Hz Gesture Timeline grid
(metrics.to_timeline) and decoded there, so every model is decoded on the same grid and
durations are in seconds. Emissions are scaled likelihoods log(p / prior) per 10 Hz frame,
divided by the emission temperature (plan grid plus 4.0: on the validation streams the optimum
was 1-2.5 for both experiments, so the grid is extended to keep it inside). Normalising by the
number of frames per new window instead (one window's evidence counted once) moved the E1
optimum to 0.1, outside the grid, with no gain.

Decoders:
  * ``none``: the E1/E2 timeline rule (argmax + merge of too-short gestures), as in thesis.metrics;
  * ``hmm``: ergodic HMM + Viterbi with minimum-duration chains (core.hmm). Grid:
    self_transition = probability per second that rest (G5) continues (0.8 -> 5 s expected
    rest, 0.9 -> 10 s, 0.95 -> 20 s), min_duration_fraction (chain length = fraction x 10th
    percentile of the class duration) and emission temperature;
  * ``hsmm``: explicit-duration Viterbi with log-normal durations (core.hsmm), optionally
    followed by the merge/relabel pass with the same minimum durations (md000 = no relabel).
    Grid: min_duration_fraction (0 or the HMM values) and emission temperature.
Each grid point is also decoded from every second window: every second window of a 75 % stream
is exactly the 50 % stream of the same model (and of a 50 % stream, the 0 % stream). The key
holds the overlap actually decoded (``ov75``, ``ov50``, ``ov00``).

Configurations: the stage-F finalists of E1 and E2 (index/selection/E1_F.json, E2_F.json) plus
the short-window references of thesis.extra_windows (index/selection/E3_windows.json): the best
0.75 s and 1 s configuration of each experiment.

Duration models (HMM chains, HSMM log-normals) are fitted on the validation streams' 10 Hz
labels: the training streams are stored only on the window grid (1.25 s for 5 s / 75 %), and the
validation streams are the tuning set of E3 anyway. No test stream is used.

Intake decisions per stream (val, test_sequences, test_wild):
  * ``g3``: a G3 segment in the decoded timeline (no threshold);
  * ``g2_g3``: a G2 segment followed later by a G3 segment;
  * ``llr_fa05`` / ``llr_fa10``: phased HMM LLR (intake vs. no-intake), max over 40 s
    episodes every 10 s plus one aligned to the stream end, on a 2 Hz grid (the timeline
    averaged over 0.5 s; the forward pass at 10 Hz is ~25x slower). Threshold per replicate:
    best balanced accuracy on the validation streams with at most 0.5 / 1.0 false events per
    hour on the external calibration participants. The HMM parameters are those selected for
    the ``hmm`` decoder of the configuration.

Two passes:
  1. grid: every grid point decoded on the validation streams of every replicate;
     the best point per (configuration, decoder) by mean val timeline.seg_f1_25
     (tie-break val timeline.macro_f1) -> index/selection/E3_grid.json;
  2. eval: the selected decoders on every set of every replicate.

Output (README.md, "Contents of a decoded result"):
  postproc/<exp_slug>/<config_key>/<replicate>/grid_val.csv
  postproc/<exp_slug>/<config_key>/<replicate>/<decoder_key>/{metrics.json, sessions.csv, paths.npz}
  index/postproc.csv               one row per (replicate, decoder): flattened metrics
  index/selection/E3_grid.json     selected grid point per configuration and decoder
  index/selection/E3.json          best decoder per configuration and overall, by validation
"""
from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from core import hsmm as P4
from core.hmm import Durations, PostProcessor, calibrate_threshold, count_events, viterbi_batch
from core.sources import LABELS, label_runs_to_sequence
from thesis import common as C
from thesis import metrics as M

E3 = C.EXPERIMENTS["E3"]
GRID = E3["hyperparameter_grid"]
TEMPERATURES = sorted(set(GRID["emission_temperature"]) | {4.0})
POSTPROC = C.POSTPROC
HZ = 10
LLR_HZ = 2
K = len(LABELS)
G2, G3 = LABELS.index("G2"), LABELS.index("G3")
FA_BUDGETS = GRID["fa_budget_per_hour"]
TEST_SETS = ["test_sequences", "test_wild"]
POST_CFG = C.TRAINING["post"]


def _t(x, n=3):
    """Two-decimal parameter as a zero-padded integer string (0.95 -> '095')."""
    return f"{int(round(x * 100)):0{n}d}"


def _fa(fa):
    """False-event budget as a two-digit string (0.5 -> '05')."""
    return f"{int(round(fa * 10)):02d}"


def decoder_points(ov_cfg: int = 75) -> list[dict]:
    """Grid points of a configuration trained with ov_cfg % overlap."""
    pts = [{"decoder": "none", "ov": ov_cfg}]
    for ov in [ov_cfg] + ([2 * ov_cfg - 100] if ov_cfg >= 50 else []):
        for st, md, tp in itertools.product(GRID["self_transition"], GRID["min_duration_fraction"],
                                            TEMPERATURES):
            pts.append({"decoder": "hmm", "ov": ov, "st": st, "md": md, "t": tp})
        for md, tp in itertools.product([0.0] + GRID["min_duration_fraction"], TEMPERATURES):
            pts.append({"decoder": "hsmm", "ov": ov, "md": md, "t": tp})
    return pts


def decoder_key(p: dict) -> str:
    """Readable key of a decoder grid point (README.md, "Decoder key")."""
    if p["decoder"] == "none":
        return "none"
    st = f"_st{_t(p['st'])}" if p["decoder"] == "hmm" else ""
    return f"{p['decoder']}__ov{p['ov']:02d}{st}_md{_t(p['md'])}_t{_t(p['t'])}"


class Replicate:
    """Everything a replicate's decoders need, loaded once."""

    def __init__(self, rep_dir: Path):
        self.dir, self.cfg_dir = rep_dir, rep_dir.parent
        self.man = json.loads((self.cfg_dir / "artifacts" / "manifest.json").read_text())
        self.info = json.loads((rep_dir / "train_info.json").read_text())
        self.frames = np.load(self.cfg_dir / "artifacts" / "frames.npz")
        self.fold = str(self.info["fold"])
        self.prior = np.array(self.info["prior"])
        self.ov = int(round(100 * (1 - self.man["stride"] / self.man["window"])))
        self.meta = {"val": self.man["folds"][self.fold]["val"], "test_sequences": self.man["lab"],
                     "test_wild": self.man["wild"]}
        self.probs = {}
        self.tl_cache = {}
        self.y10_val = [self.frames[f"y10/{s}"].astype(np.int64) for s in self.sids("val")]
        self.dpar = P4.fit_log_normal(self.y10_val)

    def sids(self, part):
        """Stream ids of a set."""
        return list(self.load(part))

    def load(self, part):
        """Window probabilities of a set, {stream id: (n_windows, 5)}."""
        if part not in self.probs:
            with np.load(self.dir / "predictions" / f"{part}.npz") as z:
                self.probs[part] = {k: z[k].astype(np.float32) for k in z.files}
        return self.probs[part]

    def n_frames(self, part, sid):
        """Number of 10 Hz frames of a stream."""
        if f"y10/{sid}" in self.frames.files:
            return len(self.frames[f"y10/{sid}"])
        p = self.load(part)[sid]
        return int(np.ceil(((len(p) - 1) * self.man["stride"] + 1) / self.man["fs"] * HZ))

    def timelines(self, part, ov):
        """10 Hz probabilities of every stream of a set, from all windows (ov = the configuration's
        overlap) or every second window (the overlap that stride doubling gives)."""
        if (part, ov) not in self.tl_cache:
            fs, win, stride = self.man["fs"], self.man["window"], self.man["stride"]
            step = 1 if ov == self.ov else 2
            self.tl_cache[(part, ov)] = [M.to_timeline(p[::step], self.n_frames(part, s), fs, win, stride * step, HZ)
                                         for s, p in self.load(part).items()]
        return self.tl_cache[(part, ov)]

    @staticmethod
    def alpha(p):
        """Emission scale 1 / temperature of a grid point."""
        return 1.0 / p["t"]

    def durations(self, md, hz=HZ):
        """HMM durations fitted on the validation streams at the given frame rate."""
        y = self.y10_val if hz == HZ else [v[::HZ // hz] for v in self.y10_val]
        return Durations.fit(y, 1, {"min_duration_fraction": md}, hz)

    def post_cfg(self, st):
        """HMM settings with the rest self-transition st (per second) as expected rest duration."""
        return {**POST_CFG, "background_expected_s": 1.0 / (1.0 - st)}

    def logE(self, tls, p):
        """Scaled-likelihood emissions of 10 Hz timelines for a grid point."""
        return [(np.log(np.clip(t, POST_CFG["emission_floor"], 1)) - np.log(self.prior)) * self.alpha(p) for t in tls]

    def decode(self, part, p):
        """Decoded 10 Hz paths of every stream of a set for a grid point."""
        tls = self.timelines(part, p["ov"])
        if p["decoder"] == "none":
            mf = np.array([round(s * HZ) for s in self.info["timeline_min_s"]])
            return [M.timeline_rule(t, mf) for t in tls]
        logs = self.logE(tls, p)
        if p["decoder"] == "hmm":
            pp = PostProcessor(self.durations(p["md"]), self.prior, self.post_cfg(p["st"]), HZ)
            return viterbi_batch(pp.seg, logs)
        paths = [P4.hsmm_decode(e, self.dpar, HZ) for e in logs]
        if p["md"] > 0:
            mf = self.durations(p["md"]).min_frames
            paths = [P4.relabel(q, e, mf) for q, e in zip(paths, logs)]
        return paths

    def llr_scores(self, part, p) -> list[np.ndarray]:
        """Phased LLR per 40 s episode (every 10 s, plus one ending at the stream end), 2 Hz grid."""
        r = HZ // LLR_HZ
        tls = [t[:len(t) // r * r].reshape(-1, r, K).mean(1) if len(t) >= r else t.mean(0, keepdims=True)
               for t in self.timelines(part, p["ov"])]
        a = self.alpha(p) * r
        pp = PostProcessor(self.durations(p["md"], LLR_HZ), self.prior, self.post_cfg(p["st"]), LLR_HZ)
        pp.emissions = lambda post: (np.log(np.clip(post, POST_CFG["emission_floor"], 1)) - np.log(self.prior)) * a
        W, step = int(POST_CFG["episode_s"] * LLR_HZ), int(POST_CFG["episode_step_s"] * LLR_HZ)
        eps, owner = [], []
        for i, t in enumerate(tls):
            starts = list(range(0, max(1, len(t) - W + 1), step))
            if len(t) > W and starts[-1] + W < len(t):
                starts.append(len(t) - W)
            eps += [t[s:s + W] for s in starts]
            owner += [i] * len(starts)
        owner = np.array(owner)
        sc = pp.llrs_batch(eps)["llr_phased"]
        return [sc[owner == i] for i in range(len(tls))]

    def external_hours(self, part):
        """Hours of recording of an external set."""
        return sum(self.n_frames(part, s) for s in self.load(part)) / HZ / 3600


def g2_g3(path) -> int:
    """1 when a G2 segment is followed later by a G3 segment."""
    seq = label_runs_to_sequence(path)
    return int("G2" in seq and "G3" in seq[seq.index("G2") + 1:])


def timeline_metrics(R: Replicate, part, paths) -> tuple[dict, list[dict]]:
    """Window, timeline, order and rule-based intake metrics of decoded paths, plus per-stream rows."""
    sids = R.sids(part)
    truths = [R.frames[f"y10/{s}"].astype(np.int64) for s in sids]
    stride_s = R.man["stride_s"]
    Yw, Pw = [], []
    for s, q in zip(sids, paths):
        y = R.frames[f"y/{s}"].astype(np.int64)
        idx = np.clip(np.round(np.arange(len(y)) * stride_s * HZ).astype(int), 0, len(q) - 1)
        Yw.append(y)
        Pw.append(q[idx])
    res = {"window": M.classification(np.concatenate(Yw), np.concatenate(Pw)),
           "timeline": {**M.classification(np.concatenate(truths), np.concatenate(paths)), **M.seg_f1(paths, truths)}}
    rows = []
    for s, q in zip(sids, paths):
        po, to = label_runs_to_sequence(q), R.meta[part][s]["sequence"]
        rows.append({"set": part, "sid": s, "pred_order": "-".join(po), "true_order": "-".join(to),
                     "exact": int(po == to), "edit": P4.levenshtein(po, to) / max(1, len(to)),
                     "intake": int(R.meta[part][s]["intake"]), "pred_g3": int(G3 in q), "pred_g2_g3": g2_g3(q)})
    res["order"] = {"exact_match": float(np.mean([r["exact"] for r in rows])),
                    "edit_distance": float(np.mean([r["edit"] for r in rows]))}
    y = [r["intake"] for r in rows]
    res["intake_g3"] = M.binary(y, [r["pred_g3"] for r in rows])
    res["intake_g2_g3"] = M.binary(y, [r["pred_g2_g3"] for r in rows])
    return res, rows


def flat(res: dict, part: str) -> dict:
    """Flattened metrics of one set."""
    return M.flatten({part: res})


def rep_dirs() -> list[Path]:
    """Replicates decoded in E3: the stage-F finalists of E1 and E2 and the short-window references."""
    configs = []
    for exp in ("E1", "E2"):
        configs += [(exp, k) for k in json.loads((C.SELECTION / f"{exp}_F.json").read_text())["advanced"]]
    win = C.SELECTION / "E3_windows.json"
    if win.exists():
        configs += [(c["experiment"], c["config_key"]) for c in json.loads(win.read_text())["configs"].values()]
    out = []
    for exp, key in configs:
        reps = sorted(p for p in C.config_dir(exp, key).glob("f*_s*") if (p / "train_info.json").exists()
                      and json.loads((p / "manifest.json").read_text()).get("status") == "done")
        out += [r for r in reps if r not in out]
    return out


def role(rep: Path) -> str:
    """'finalist' (stage F) or 'window_ref' (thesis.extra_windows)."""
    exp = rep.parts[-3][:2]
    fin = json.loads((C.SELECTION / f"{exp}_F.json").read_text())["advanced"]
    return "finalist" if rep.parent.name in fin else "window_ref"


def out_dir(rep: Path) -> Path:
    """Post-processing directory of a replicate."""
    return POSTPROC / rep.relative_to(C.RUNS)


def grid_job(args):
    """Every decoder grid point on the validation streams of one replicate -> grid_val.csv."""
    rep, force = args
    f = out_dir(rep) / "grid_val.csv"
    if f.exists() and not force:
        return f
    R = Replicate(rep)
    rows = []
    for p in decoder_points(R.ov):
        res, _ = timeline_metrics(R, "val", R.decode("val", p))
        rows.append({"decoder_key": decoder_key(p), **p, **flat(res, "val")})
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f.with_suffix(".tmp"), index=False)
    f.with_suffix(".tmp").replace(f)
    return f


def select_grid(reps: list[Path]) -> dict:
    """Best grid point per configuration and decoder (mean val seg F1@25 over the replicates) -> E3_grid.json."""
    df = pd.concat([pd.read_csv(out_dir(r) / "grid_val.csv").assign(config=str(r.parent.relative_to(C.RUNS)))
                    for r in reps])
    rule = C.PLAN["common_protocol"]["selection_rule"] if "selection_rule" in C.PLAN["common_protocol"] else None
    g = df.groupby(["config", "decoder", "decoder_key"])[["val.timeline.seg_f1_25", "val.timeline.macro_f1"]] \
        .mean().reset_index()
    out = {"stage": "E3 grid", "rule": "max mean val.timeline.seg_f1_25, tie-break val.timeline.macro_f1",
           "plan_rule": rule, "configs": {}}
    for cfg, d in g.groupby("config"):
        out["configs"][cfg] = {}
        for dec, dd in d.groupby("decoder"):
            dd = dd.sort_values(["val.timeline.seg_f1_25", "val.timeline.macro_f1"], ascending=False)
            best = dd.iloc[0]
            pt = df[(df.config == cfg) & (df.decoder_key == best.decoder_key)].iloc[0]
            params = {k: (int(pt[k]) if k == "ov" else float(pt[k])) for k in ("ov", "st", "md", "t")
                      if k in pt and pd.notna(pt[k])}
            out["configs"][cfg][dec] = {"decoder_key": best.decoder_key, "params": {"decoder": dec, **params},
                                        "val.timeline.seg_f1_25": float(best["val.timeline.seg_f1_25"]),
                                        "val.timeline.macro_f1": float(best["val.timeline.macro_f1"]),
                                        "top5": dd.head(5)[["decoder_key", "val.timeline.seg_f1_25"]]
                                        .to_dict("records")}
    C.write_json(C.SELECTION / "E3_grid.json", out)
    return out


def eval_job(args):
    """The selected decoders and the LLR intake decision on every set of one replicate."""
    rep, chosen, force = args
    R = Replicate(rep)
    od = out_dir(rep)
    hp = chosen["hmm"]["params"]
    sc = {part: R.llr_scores(part, hp) for part in ["val", *TEST_SETS, "external_calibration", "test_external"]}
    smax = {part: np.array([s.max() if len(s) else -np.inf for s in v]) for part, v in sc.items()}
    yv = np.array([R.meta["val"][s]["intake"] for s in R.sids("val")])
    cal_h, test_h = R.external_hours("external_calibration"), R.external_hours("test_external")
    taus = {fa: calibrate_threshold(smax["val"], yv, sc["external_calibration"], cal_h, fa) for fa in FA_BUDGETS}
    rows = []
    for dec, ch in chosen.items():
        d = od / ch["decoder_key"]
        if (d / "metrics.json").exists() and not force:
            row = json.loads((d / "summary_row.json").read_text())
            rows.append(row)
            continue
        p = ch["params"]
        metrics = {"run_id": str(rep.relative_to(C.RUNS)), "decoder_key": ch["decoder_key"], "params": p,
                   "llr_hmm_params": hp, "llr_thresholds": {f"fa{_fa(fa)}": taus[fa] for fa in FA_BUDGETS}}
        sessions, paths_out, fl = [], {}, {}
        for part in ["val", *TEST_SETS]:
            paths = R.decode(part, p)
            res, srows = timeline_metrics(R, part, paths)
            y = np.array([r["intake"] for r in srows])
            for fa in FA_BUDGETS:
                name = f"intake_llr_fa{_fa(fa)}"
                pred = (smax[part] > taus[fa]).astype(int)
                res[name] = M.binary(y, pred)
                for r, s, q in zip(srows, smax[part], pred):
                    r["llr"], r[f"pred_llr_fa{_fa(fa)}"] = float(s), int(q)
            metrics[part] = res
            fl.update(flat(res, part))
            sessions += srows
            paths_out.update({f"{part}/{s}": q.astype(np.int8) for s, q in zip(R.sids(part), paths)})
        ext = R.decode("test_external", p)
        ext_res = {"hours": test_h,
                   "gesture_events_per_hour": sum(M.n_gesture_events(q) for q in ext) / test_h,
                   "g3_events_per_hour": sum(M.n_gesture_events(q, G3) for q in ext) / test_h}
        for fa in FA_BUDGETS:
            ext_res[f"llr_fa{_fa(fa)}_events_per_hour"] = sum(count_events(s, taus[fa]) for s in sc["test_external"]) / test_h
        metrics["test_external"] = {"external": ext_res}
        fl.update(flat(metrics["test_external"], "test_external"))
        d.mkdir(parents=True, exist_ok=True)
        C.write_json(d / "metrics.json", metrics)
        pd.DataFrame(sessions).to_csv(d / "sessions.csv", index=False)
        np.savez_compressed(d / "paths.npz", **paths_out)
        info = R.info
        row = {"run_id": metrics["run_id"], "experiment": rep.parts[-3][:2], "config_key": rep.parent.name,
               "role": role(rep), "window_s": R.man["window_s"], "replicate": rep.name, "fold": info["fold"],
               "seed": info["seed"], "decoder": dec,
               "decoder_key": ch["decoder_key"], **fl, "path": str(d.relative_to(C.ROOT))}
        C.write_json(d / "summary_row.json", row)
        rows.append(row)
    return rows


def final_selection(df: pd.DataFrame) -> dict:
    """Best decoder per configuration, per experiment and overall -> E3.json."""
    keys = ["val.timeline.seg_f1_25", "val.timeline.macro_f1"]
    g = df.groupby(["experiment", "role", "config_key", "decoder", "decoder_key"]).mean(numeric_only=True).reset_index()
    out = {"stage": "E3", "rule": "max mean val.timeline.seg_f1_25, tie-break val.timeline.macro_f1",
           "per_config": {}, "per_experiment": {}, "note": "per_experiment and overall: finalists only"}
    for (exp, cfg), d in g.groupby(["experiment", "config_key"]):
        b = d.sort_values(keys, ascending=False).iloc[0]
        out["per_config"][f"{exp}/{cfg}"] = {"decoder_key": b.decoder_key, **{k: float(b[k]) for k in keys}}
    for exp, d in g[g.role == "finalist"].groupby("experiment"):
        b = d.sort_values(keys, ascending=False).iloc[0]
        out["per_experiment"][exp] = {"config_key": b.config_key, "decoder_key": b.decoder_key,
                                      **{k: float(b[k]) for k in keys}}
    b = g[g.role == "finalist"].sort_values(keys, ascending=False).iloc[0]
    out["overall"] = {"experiment": b.experiment, "config_key": b.config_key, "decoder_key": b.decoder_key,
                      **{k: float(b[k]) for k in keys}}
    C.write_json(C.SELECTION / "E3.json", out)
    return out


def progress(stage, done, total):
    """Write index/postproc_progress.json."""
    C.write_json(C.INDEX / "postproc_progress.json", {"stage": stage, "done": done, "total": total, "updated": C.now()})


def main():
    """E3: grid pass, selection, evaluation pass, index/postproc.csv and E3.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N replicates (testing)")
    a = ap.parse_args()
    reps = rep_dirs()[: a.limit or None]
    print(f"E3: {len(reps)} replicates, {len(decoder_points())} grid points each", flush=True)
    with Pool(a.workers) as pool:
        for i, f in enumerate(pool.imap_unordered(grid_job, [(r, a.force) for r in reps]), 1):
            progress("grid", i, len(reps))
            print(f"[{C.now()}] grid {i}/{len(reps)}: {f.parent.relative_to(POSTPROC)}", flush=True)
        sel = select_grid(reps)
        jobs = [(r, sel["configs"][str(r.parent.relative_to(C.RUNS))], a.force) for r in reps]
        rows = []
        for i, rr in enumerate(pool.imap_unordered(eval_job, jobs), 1):
            rows += rr
            progress("eval", i, len(reps))
            print(f"[{C.now()}] eval {i}/{len(reps)}: {rr[0]['run_id']}", flush=True)
    df = pd.DataFrame(rows).sort_values(["experiment", "config_key", "replicate", "decoder"])
    df.to_csv(C.INDEX / "postproc.csv", index=False)
    fin = final_selection(df)
    print(json.dumps(fin["per_experiment"], indent=1), flush=True)


if __name__ == "__main__":
    main()
