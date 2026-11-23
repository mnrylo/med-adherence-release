"""Stage D2: panel S, every E1/E2 replicate with saved predictions, reduced grid.

    python -m decoders.d2_run [--workers 14]

Decoders: none, mode, hmm, hsmm (the learned decoders are too costly for 3,222 replicates; they
stay on panel F). Reduced grid, decoded at the configuration's own overlap only (the E3
grid also tried every second window):
  * hmm:  self-transition {0.8, 0.95} x min-duration fraction {0.3, 0.7} x temperature {1, 2, 4}
  * hsmm: min-duration fraction {0.3, 0.7} x temperature {1, 2, 4}
  * mode: decoders.MODE_S
The values span the points E3 selected for panel F (st 0.8-0.95, md 0.3-0.7, t 1-4).

Pass 1 (per replicate): every grid point on val -> index/d2/grid/<run>.csv (pooled segment counts
at IoU 0.25, Order Exact Match, frame macro F1).
Selection (per configuration, like E3): max mean val seg F1@25 over the replicates,
tie-break mean val frame macro F1 -> index/d2/selection.json.
Pass 2 (per replicate): the selected point of each decoder on test_wild (dcommon.session_rows),
the HMM LLR score with thresholds at <= 0.5 / 1 false events/h on external_calibration, and the
G3-rule false events/h on test_external -> index/d2/test/<run>.csv.gz and .json.
"""
from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from decoders import dcommon as D
from decoders import decoders as DE
from decoders.dcommon import M, P4, PP

OUT = D.INDEX / "d2"
ST, MD, TT = [0.8, 0.95], [0.3, 0.7], [1.0, 2.0, 4.0]


def all_reps() -> list[Path]:
    """Every E1/E2 replicate with the allowed prediction files."""
    out = []
    for exp in ("E1_gestures_only", "E2_daily_negatives"):
        for r in sorted((D.C.RUNS / exp).glob("*/f*_s*")):
            p = r / "predictions"
            if all((p / f"{s}.npz").exists() for s in D.ALLOWED) and (r / "train_info.json").exists():
                out.append(r)
    return out


def stem(rep: Path) -> str:
    """File stem of a replicate."""
    return str(rep.relative_to(D.C.RUNS)).replace("/", "__")


def points(ov: int) -> list[dict]:
    """Reduced grid points of a configuration."""
    pts = [{"decoder": "none", "ov": ov}]
    pts += [{"decoder": "hmm", "ov": ov, "st": s, "md": m, "t": t} for s, m, t in itertools.product(ST, MD, TT)]
    pts += [{"decoder": "hsmm", "ov": ov, "md": m, "t": t} for m, t in itertools.product(MD, TT)]
    pts += [{"decoder": f"mode_{int(w * 10):02d}", "ov": ov, "w": w} for w in DE.MODE_S]
    return pts


def key(p: dict) -> str:
    """Key of a reduced-grid point."""
    return p["decoder"] if p["decoder"] in ("none",) or p["decoder"].startswith("mode") else PP.decoder_key(p)


def decode(R: D.Rep, part: str, p: dict):
    """Decoded paths of a set for a grid point (mode filter included)."""
    if p["decoder"].startswith("mode"):
        return [DE.mode_filter(t, p["w"]) for t in R.timelines(part, p["ov"])]
    return R.decode(part, p)


def frame_macro_f1(paths, truths) -> float:
    """Frame-level macro F1 of decoded paths."""
    q, t = np.concatenate(paths), np.concatenate(truths)
    f = []
    for k in range(PP.K):
        tp, fp, fn = np.sum((q == k) & (t == k)), np.sum((q == k) & (t != k)), np.sum((q != k) & (t == k))
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def grid_job(rep: Path):
    """Every reduced-grid point on the validation streams of one replicate."""
    f = OUT / "grid" / f"{stem(rep)}.csv"
    if f.exists():
        return
    R = D.Rep(rep)
    ys = R.truths("val")
    rows = []
    for p in points(R.ov):
        q = decode(R, "val", p)
        r = pd.DataFrame(D.session_rows(R, "val", q, key(p)))
        rows.append({"decoder": p["decoder"].split("_")[0], "decoder_key": key(p), **{k: p.get(k) for k in ("st", "md", "t", "w")},
                     "tp25": r.tp25.sum(), "fp25": r.fp25.sum(), "fn25": r.fn25.sum(), "exact": r.exact.mean(),
                     "frame_macro_f1": frame_macro_f1(q, ys)})
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f.with_suffix(".tmp"), index=False)
    f.with_suffix(".tmp").replace(f)


def select(reps: list[Path]) -> dict:
    """Best point per configuration and decoder -> index/d2/selection.json."""
    df = pd.concat(pd.read_csv(OUT / "grid" / f"{stem(r)}.csv").assign(config=str(r.parent.relative_to(D.C.RUNS)))
                   for r in reps)
    df["f1"] = 2 * df.tp25 / (2 * df.tp25 + df.fp25 + df.fn25).clip(lower=1)
    g = df.groupby(["config", "decoder", "decoder_key"])[["f1", "frame_macro_f1"]].mean().reset_index()
    out = {"rule": "max mean val seg F1@25, tie-break mean val frame macro F1", "configs": {}}
    for (cfg, dec), d in g.groupby(["config", "decoder"]):
        b = d.sort_values(["f1", "frame_macro_f1"], ascending=False).iloc[0]
        pt = df[(df.config == cfg) & (df.decoder_key == b.decoder_key)].iloc[0]
        out["configs"].setdefault(cfg, {})[dec] = {
            "decoder_key": b.decoder_key, "val_seg_f1_25": float(b.f1), "val_frame_macro_f1": float(b.frame_macro_f1),
            "params": {k: float(pt[k]) for k in ("st", "md", "t", "w") if pd.notna(pt[k])}}
    D.write_json(OUT / "selection.json", out)
    return out


def test_job(args):
    """Selected decoders, LLR and external rates of one replicate on the test sets."""
    rep, sel = args
    f_csv, f_json = OUT / "test" / f"{stem(rep)}.csv.gz", OUT / "test" / f"{stem(rep)}.json"
    if f_json.exists():
        return
    R = D.Rep(rep)
    rows, ext = [], {}
    pts = {}
    for dec, s in sel.items():
        p = {"decoder": s["decoder_key"] if dec == "mode" else dec, "ov": R.ov, **s["params"]}
        pts[dec] = p
    hp = pts["hmm"]
    sc = {part: R.llr_scores(part, hp) for part in ["val", D.TEST, "external_calibration", "test_external"]}
    smax = {part: np.array([s.max() if len(s) else -np.inf for s in v]) for part, v in sc.items()}
    yv = np.array([R.meta["val"][s]["intake"] for s in R.sids("val")])
    cal_h, test_h = R.external_hours("external_calibration"), R.external_hours("test_external")
    taus = {fa: PP.calibrate_threshold(smax["val"], yv, sc["external_calibration"], cal_h, fa) for fa in PP.FA_BUDGETS}
    for dec, p in pts.items():
        r = D.session_rows(R, D.TEST, decode(R, D.TEST, p), dec)
        if dec == "hmm":
            for fa in PP.FA_BUDGETS:
                for x, s in zip(r, smax[D.TEST]):
                    x["score"], x[f"pred_score_fa{PP._fa(fa)}"] = float(s), int(s > taus[fa])
        rows += r
        e = decode(R, "test_external", p)
        ext[dec] = {"g3_rule_per_h": sum(M.n_gesture_events(q, PP.G3) for q in e) / test_h,
                    "gesture_events_per_h": sum(M.n_gesture_events(q) for q in e) / test_h}
    ext["hmm"].update({f"score_fa{PP._fa(fa)}_per_h": sum(PP.count_events(s, taus[fa]) for s in sc["test_external"]) / test_h
                       for fa in PP.FA_BUDGETS})
    meta = {"run_id": str(rep.relative_to(D.C.RUNS)), "experiment": rep.parts[-3][:2], "config_key": rep.parent.name,
            "replicate": rep.name}
    f_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**meta, **r} for r in rows]).to_csv(f_csv, index=False)
    D.write_json(f_json, {**meta, "points": pts, "thresholds": {f"fa{PP._fa(fa)}": t for fa, t in taus.items()},
                          "test_external": ext})


def main():
    """Run stage D2."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N configurations")
    a = ap.parse_args()
    reps = all_reps()
    if a.limit:
        keep = sorted({r.parent for r in reps})[: a.limit]
        reps = [r for r in reps if r.parent in keep]
    print(f"[{D.now()}] D2: {len(reps)} replicates, {len({r.parent for r in reps})} configurations", flush=True)
    with Pool(a.workers, maxtasksperchild=8) as pool:
        for i, _ in enumerate(pool.imap_unordered(grid_job, reps), 1):
            if i % 50 == 0 or i == len(reps):
                print(f"[{D.now()}] grid {i}/{len(reps)}", flush=True)
        sel = select(reps)["configs"]
        print(f"[{D.now()}] selection done", flush=True)
        jobs = [(r, sel[str(r.parent.relative_to(D.C.RUNS))]) for r in reps]
        for i, _ in enumerate(pool.imap_unordered(test_job, jobs), 1):
            if i % 50 == 0 or i == len(reps):
                print(f"[{D.now()}] test {i}/{len(reps)}", flush=True)
    read = sorted({l.rsplit("/", 1)[-1] for l in D.READ_LOG.read_text().splitlines() if l})
    print(f"sets read: {read}", flush=True)


if __name__ == "__main__":
    main()
