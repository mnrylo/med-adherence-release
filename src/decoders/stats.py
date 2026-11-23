"""Participant-level statistics (fixed before the test sets were read).

Input: session rows (dcommon.session_rows), one per (replicate, decoder, stream). For a group of
replicates (one configuration, or a set of configurations), each metric is computed per
(participant, replicate) from the participant's sessions, averaged over replicates, then averaged
over participants with equal weight. Confidence intervals: cluster bootstrap, participants
resampled with replacement and sessions resampled within them; the same draws are used for every
decoder, so differences are paired. Paired tests: Wilcoxon signed-rank over participants, Holm
correction across the decoders compared with one reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = {
    "seg_f1_25": ("f1", ("tp25", "fp25", "fn25")),
    "seg_f1_10": ("f1", ("tp10", "fp10", "fn10")),
    "seg_f1_50": ("f1", ("tp50", "fp50", "fn50")),
    "order_exact": ("mean", ("exact",)),
    "order_edit": ("mean", ("edit",)),
    "g3_sens": ("rate_pos", ("pred_g3",)),
    "g3_spec": ("rate_neg", ("pred_g3",)),
    "llr10_sens": ("rate_pos", ("pred_llr_fa10",)),
    "llr10_spec": ("rate_neg", ("pred_llr_fa10",)),
    "score10_sens": ("rate_pos", ("pred_score_fa10",)),
    "score10_spec": ("rate_neg", ("pred_score_fa10",)),
}


class Cube:
    """Session × replicate arrays of one decoder, sessions in a fixed order shared by all decoders."""

    def __init__(self, df: pd.DataFrame, sids: list[str], reps: list[str]):
        d = df.set_index(["sid", "run_id"])
        self.cols = {}
        for c in {c for _, cs in METRICS.values() for c in cs} | {"intake"}:
            if c in d:
                self.cols[c] = d[c].unstack("run_id").reindex(index=sids, columns=reps).to_numpy(float)


def _participant_values(cube: Cube, idx_by_subject: list[np.ndarray], metric: str) -> np.ndarray:
    """Metric per participant (pooled over the participant's sessions, averaged over replicates)."""
    kind, cols = METRICS[metric]
    out = []
    for idx in idx_by_subject:
        if kind == "f1":
            tp, fp, fn = (cube.cols[c][idx].sum(0) for c in cols)
            den = 2 * tp + fp + fn
            v = np.where(den > 0, 2 * tp / np.maximum(den, 1), 1.0)
        elif kind == "mean":
            v = cube.cols[cols[0]][idx].mean(0)
        else:
            y = cube.cols["intake"][idx, 0]
            sel = y == (1 if kind == "rate_pos" else 0)
            if not sel.any():
                out.append(np.nan)
                continue
            p = cube.cols[cols[0]][idx][sel]
            v = p.mean(0) if kind == "rate_pos" else 1 - p.mean(0)
        out.append(np.nanmean(v))
    return np.array(out)


def compare(df: pd.DataFrame, decoders: list[str], reference: str, metrics: list[str],
            n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Participant-level mean, 95 % CI, paired difference to `reference` with CI and Holm-adjusted
    Wilcoxon p, for every decoder and metric of one group of replicates."""
    sess = df[df.decoder == reference].drop_duplicates("sid")[["sid", "subject"]]
    sids, subjects = list(sess.sid), sess.subject.to_numpy()
    reps = sorted(df.run_id.unique())
    cubes = {d: Cube(df[df.decoder == d], sids, reps) for d in decoders}
    subj = sorted(set(subjects))
    idx_by_subject = [np.flatnonzero(subjects == s) for s in subj]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(subj), len(subj))
        draws.append([rng.choice(idx_by_subject[j], len(idx_by_subject[j])) for j in pick])
    rows = []
    for m in metrics:
        if METRICS[m][1][0] not in next(iter(cubes.values())).cols:
            continue
        point = {d: _participant_values(cubes[d], idx_by_subject, m) for d in decoders}
        boot = {d: np.array([np.nanmean(_participant_values(cubes[d], dr, m)) for dr in draws]) for d in decoders}
        ps = {}
        for d in decoders:
            if d == reference:
                continue
            diff = point[d] - point[reference]
            ok = ~np.isnan(diff)
            ps[d] = wilcoxon(diff[ok]).pvalue if ok.sum() >= 5 and np.any(diff[ok] != 0) else 1.0
        order = sorted(ps, key=ps.get)
        holm, run = {}, 0.0
        for i, d in enumerate(order):
            run = max(run, min(1.0, ps[d] * (len(order) - i)))
            holm[d] = run
        for d in decoders:
            b, db = boot[d], boot[d] - boot[reference]
            rows.append({"metric": m, "decoder": d, "mean": np.nanmean(point[d]),
                         "ci_lo": np.nanpercentile(b, 2.5), "ci_hi": np.nanpercentile(b, 97.5),
                         "diff": np.nanmean(point[d] - point[reference]),
                         "diff_lo": np.nanpercentile(db, 2.5), "diff_hi": np.nanpercentile(db, 97.5),
                         "p_holm": holm.get(d, np.nan), "participants": int((~np.isnan(point[d])).sum()),
                         "replicates": len(reps)})
    return pd.DataFrame(rows)
