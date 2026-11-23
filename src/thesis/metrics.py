"""Window, Gesture Timeline, Gesture Order and intake metrics from saved window probabilities.

Definitions and metric keys: GLOSSARY.md. Everything here reads files only (no model, no data
loading), so metrics can be recomputed or extended for any finished run:

    python -m thesis.metrics [--force] [replicate_dir ...]

Inputs: <config>/artifacts/{manifest.json, frames.npz} and <config>/<replicate>/predictions/*.npz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from core import hsmm as P4
from core.sources import LABELS, REST, label_runs_to_sequence
from thesis import common as C

K = len(LABELS)
G3 = LABELS.index("G3")
EVAL_SETS = ["val", "test_sequences", "test_wild"]
SEG_K = (0.10, 0.25, 0.50)


def to_timeline(probs: np.ndarray, n_frames: int, fs: float, win: int, stride: int, hz: float) -> np.ndarray:
    """Window probabilities -> (n_frames, K) on the timeline grid: each frame averages the windows
    that cover it (window j is centred on sample j * stride and spans win samples)."""
    t = np.arange(n_frames) / hz
    ds, hw = stride / fs, win / fs / 2
    n = len(probs)
    lo = np.clip(np.ceil((t - hw) / ds - 1e-9).astype(int), 0, n - 1)
    hi = np.clip(np.floor((t + hw) / ds + 1e-9).astype(int), 0, n - 1)
    hi = np.maximum(hi, lo)
    cs = np.vstack([np.zeros((1, probs.shape[1])), np.cumsum(probs.astype(np.float64), 0)])
    return ((cs[hi + 1] - cs[lo]) / (hi - lo + 1)[:, None]).astype(np.float32)


def segments(lab: np.ndarray) -> list[tuple[int, int, int]]:
    """(class, start, end) runs of every label, rest included."""
    b = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)]
    return [(int(lab[b[i]]), int(b[i]), int(b[i + 1])) for i in range(len(b) - 1)]


def timeline_rule(tl: np.ndarray, min_frames: np.ndarray, passes: int = 3) -> np.ndarray:
    """Timeline without post-processing: argmax, then gesture segments shorter than their class
    minimum are given to the neighbour with the higher mean probability over the segment."""
    lab = tl.argmax(1)
    for _ in range(passes):
        segs = segments(lab)
        short = [i for i, (c, a, b) in enumerate(segs) if c != REST and b - a < min_frames[c]]
        if not short or len(segs) == 1:
            break
        new = lab.copy()
        for i in short:
            c, a, b = segs[i]
            cand = [segs[j][0] for j in (i - 1, i + 1) if 0 <= j < len(segs)]
            new[a:b] = max(cand, key=lambda k: tl[a:b, k].mean())
        lab = new
    return lab


def classification(y: np.ndarray, p: np.ndarray) -> dict:
    """Per-class precision, recall and F1, macro F1, accuracy and confusion matrix (MASK labels ignored)."""
    keep = y >= 0
    y, p = y[keep], p[keep]
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(K)), zero_division=0)
    out = {"accuracy": float((y == p).mean()) if len(y) else np.nan,
           "macro_f1": float(f1[sup > 0].mean()) if (sup > 0).any() else np.nan,
           "n": int(len(y))}
    for i, g in enumerate(LABELS):
        out[f"precision.{g}"], out[f"recall.{g}"], out[f"f1.{g}"] = float(pr[i]), float(rc[i]), float(f1[i])
    out["confusion"] = confusion_matrix(y, p, labels=list(range(K))).tolist()
    return out


def seg_f1(pred_paths, true_paths) -> dict:
    """Segmental F1 at every IoU of SEG_K, TP / FP / FN pooled over the streams."""
    tot = {k: np.zeros(3) for k in SEG_K}
    for p, t in zip(pred_paths, true_paths):
        ps, ts = P4.label_segments(p), P4.label_segments(t)
        for k in SEG_K:
            tot[k] += P4.segment_counts(ps, ts, k)
    return {f"seg_f1_{int(k * 100)}": float(P4.seg_f1(*tot[k])) for k in SEG_K}


def binary(y, p) -> dict:
    """Sensitivity, specificity, balanced accuracy and accuracy of binary decisions."""
    y, p = np.asarray(y), np.asarray(p)
    sens = float(p[y == 1].mean()) if (y == 1).any() else np.nan
    spec = float(1 - p[y == 0].mean()) if (y == 0).any() else np.nan
    return {"sensitivity": sens, "specificity": spec, "balanced_accuracy": float(np.nanmean([sens, spec])),
            "accuracy": float((y == p).mean()) if len(y) else np.nan}


def n_gesture_events(path, cls=None) -> int:
    """Number of gesture segments in a path (of one class when cls is given)."""
    return sum(1 for c, _, _ in segments(path) if c != REST and (cls is None or c == cls))


def evaluate(config_dir: Path, rep_dir: Path) -> dict:
    """metrics.json content for one replicate."""
    man = json.loads((config_dir / "artifacts" / "manifest.json").read_text())
    info = json.loads((rep_dir / "train_info.json").read_text())
    frames = np.load(config_dir / "artifacts" / "frames.npz")
    fs, win, stride, hz = man["fs"], man["window"], man["stride"], man["timeline_hz"]
    min_frames = np.array([round(s * hz) for s in info["timeline_min_s"]])
    fold = str(info["fold"])
    meta = {"val": man["folds"].get(fold, {}).get("val", {}), "test_sequences": man["lab"], "test_wild": man["wild"]}
    out = {"run_id": info["run_id"], "timeline_rule": {"min_s": info["timeline_min_s"]}}
    sessions = []
    for part in EVAL_SETS + ["test_external"]:
        f = rep_dir / "predictions" / f"{part}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            probs = {k: z[k].astype(np.float32) for k in z.files}
        sids = list(probs)
        Yw = np.concatenate([frames[f"y/{s}"][:len(probs[s])].astype(np.int64) for s in sids])
        Pw = np.concatenate([probs[s][:len(frames[f"y/{s}"])].argmax(1) for s in sids])
        res = {"window": classification(Yw, Pw)}
        paths, truths = [], []
        for s in sids:
            yt = frames[f"y10/{s}"].astype(np.int64)
            tl = to_timeline(probs[s], len(yt), fs, win, stride, hz)
            paths.append(timeline_rule(tl, min_frames))
            truths.append(yt)
        if part == "test_external":
            hours = sum(len(t) for t in truths) / hz / 3600
            res["external"] = {"hours": hours,
                               "gesture_events_per_hour": sum(n_gesture_events(p) for p in paths) / hours,
                               "g3_events_per_hour": sum(n_gesture_events(p, G3) for p in paths) / hours}
            out[part] = res
            continue
        res["timeline"] = {**classification(np.concatenate(truths), np.concatenate(paths)), **seg_f1(paths, truths)}
        rows = []
        for s, p, t in zip(sids, paths, truths):
            po, to = label_runs_to_sequence(p), meta[part][s]["sequence"]
            rows.append({"set": part, "sid": s, "pred_order": "-".join(po), "true_order": "-".join(to),
                         "exact": int(po == to), "edit": P4.levenshtein(po, to) / max(1, len(to)),
                         "intake": int(meta[part][s]["intake"]), "pred_intake_g3": int(G3 in p)})
        res["order"] = {"exact_match": float(np.mean([r["exact"] for r in rows])),
                        "edit_distance": float(np.mean([r["edit"] for r in rows]))}
        res["intake_g3"] = binary([r["intake"] for r in rows], [r["pred_intake_g3"] for r in rows])
        sessions += rows
        out[part] = res
    return out, sessions


def flatten(metrics: dict) -> dict:
    """Scalar metrics as 'set.level.metric' (confusion matrices left out)."""
    flat = {}
    for part, levels in metrics.items():
        if not isinstance(levels, dict) or part == "timeline_rule":
            continue
        for level, vals in levels.items():
            for k, v in vals.items():
                if k != "confusion" and isinstance(v, (int, float)):
                    flat[f"{part}.{level}.{k}"] = v
    return flat


def write(config_dir: Path, rep_dir: Path):
    """Write metrics.json and sessions.csv of a replicate and return the metrics."""
    import pandas as pd
    m, sessions = evaluate(config_dir, rep_dir)
    C.write_json(rep_dir / "metrics.json", m)
    pd.DataFrame(sessions).to_csv(rep_dir / "sessions.csv", index=False)
    return m


def main():
    """Write metrics.json and sessions.csv of the given (default: every finished) replicate."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="replicate directories (default: every finished run)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    reps = [Path(r).resolve() for r in a.runs] or [p.parent for p in sorted(C.RUNS.glob("*/*/*/train_info.json"))]
    for r in reps:
        if (r / "metrics.json").exists() and not a.force:
            continue
        write(r.parent, r)
        print(f"metrics: {r.relative_to(C.RUNS)}", flush=True)


if __name__ == "__main__":
    main()
