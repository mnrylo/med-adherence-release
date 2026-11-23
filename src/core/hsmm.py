"""Explicit-duration decoding (HSMM), the relabel pass and segment-level metrics.

* ``fit_log_normal``: log-normal duration model per class from labelled frames;
* ``hsmm_decode``: explicit-duration Viterbi over the five classes;
* ``relabel``: merge of too-short segments into their better neighbour, then one class per
  segment by summed log emission;
* ``label_segments``, ``segment_counts``, ``seg_f1``: segmental F1@k (Lea et al. 2017);
* ``levenshtein``: edit distance between gesture sequences.
"""
from __future__ import annotations

import numpy as np

from core.hmm import _ergodic_weights
from core.sources import LABELS, REST

K = len(LABELS)
SEG_IOU = (0.10, 0.25, 0.50)


def fit_log_normal(label_frames: list[np.ndarray]):
    """Per class (mean, sd, 99th percentile) of the log segment length in frames; classes without segments get 3
    frames.
    """
    segs = {k: [] for k in range(K)}
    for y in label_frames:
        y = y[y >= 0]
        if len(y) == 0:
            continue
        b = np.r_[0, np.flatnonzero(np.diff(y)) + 1, len(y)]
        for i in range(len(b) - 1):
            segs[int(y[b[i]])].append(b[i + 1] - b[i])
    out = {}
    for k in range(K):
        d = np.log(np.array(segs[k] or [3], float))
        out[k] = (d.mean(), max(0.25, d.std()), np.exp(np.percentile(d, 99)))
    return out


def hsmm_decode(logE: np.ndarray, dpar: dict, fr: float) -> np.ndarray:
    """Explicit-duration Viterbi over K classes. Same-class segments may follow each other
    (cheaply for rest, rarely for gestures), so long stretches are never forced to change."""
    T = len(logE)
    caps = np.array([max(2, min(T, int(np.ceil(2 * dpar[k][2])))) for k in range(K)])
    caps[REST] = max(2, min(T, int(20 * fr)))
    D = int(caps.max())
    d = np.arange(1, D + 1)
    logdur = np.full((K, D), -np.inf)
    for k in range(K):
        mu, sd, _ = dpar[k]
        lp = -np.log(d[:caps[k]]) - (np.log(d[:caps[k]]) - mu) ** 2 / (2 * sd ** 2)
        logdur[k, :caps[k]] = lp - np.logaddexp.reduce(lp)
    w = _ergodic_weights({})
    A = np.zeros((K, K))
    for i, g in enumerate(LABELS):
        for j, h in enumerate(LABELS):
            A[i, j] = w[g].get(h, 0.0)
        A[i, i] = 1.0 if i == REST else 0.05
    logA = np.log(A / A.sum(1, keepdims=True))
    logstart = np.log(np.full(K, 1.0 / K))
    cs = np.vstack([np.zeros((1, K)), np.cumsum(logE, 0)])
    V = np.full((T + 1, K), -np.inf)
    inc = np.full((T + 1, K), -np.inf)
    arg_in = np.zeros((T + 1, K), np.int64)
    bestd = np.zeros((T + 1, K), np.int64)
    inc[0] = logstart
    for t in range(1, T + 1):
        dd = d[:min(D, t)]
        s0 = t - dd
        score = inc[s0] + (cs[t][None, :] - cs[s0]) + logdur[:, :len(dd)].T
        j = score.argmax(0)
        V[t] = score[j, np.arange(K)]
        bestd[t] = dd[j]
        if t < T:
            cand = V[t][:, None] + logA
            arg_in[t] = cand.argmax(0)
            inc[t] = cand.max(0)
    path = np.empty(T, np.int64)
    t, k = T, int(V[T].argmax())
    while t > 0:
        L = bestd[t, k]
        path[t - L:t] = k
        s = t - L
        k = int(arg_in[s, k]) if s > 0 else k
        t = s
    return path


def relabel(path: np.ndarray, logE: np.ndarray, min_frames: np.ndarray, passes: int = 2) -> np.ndarray:
    """Merge the shortest segment below its class minimum into its better neighbour until none is left, then give every
    segment the class with the largest summed log emission; repeated ``passes`` times.
    """
    p = path.copy()
    for _ in range(passes):
        while True:
            b = np.r_[0, np.flatnonzero(np.diff(p)) + 1, len(p)]
            if len(b) <= 2:
                break
            segs = [(b[i], b[i + 1], int(p[b[i]])) for i in range(len(b) - 1)]
            short = [(e - s, i) for i, (s, e, k) in enumerate(segs) if e - s < min_frames[k]]
            if not short:
                break
            _, i = min(short)
            s, e, _ = segs[i]
            nb = [segs[j][2] for j in (i - 1, i + 1) if 0 <= j < len(segs)]
            p[s:e] = max(nb, key=lambda c: logE[s:e, c].sum())
        b = np.r_[0, np.flatnonzero(np.diff(p)) + 1, len(p)]
        for i in range(len(b) - 1):
            s, e = b[i], b[i + 1]
            p[s:e] = int(logE[s:e].sum(0).argmax())
    return p


def levenshtein(a, b):
    """Edit distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def label_segments(lab):
    """(class, start, end) runs of non-rest gestures; rest and masked frames are not segments."""
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            if lab[s] >= 0 and lab[s] != REST:
                out.append((int(lab[s]), s, i))
            s = i
    return out


def segment_counts(pred, true, k):
    """TP/FP/FN of segmental F1@k (Lea et al. 2017, as in MS-TCN): each predicted segment takes the
    same-class reference segment of highest IoU; a TP if IoU >= k and that reference is still unused."""
    used, tp = set(), 0
    for c, a, b in pred:
        best, j = 0.0, None
        for i, (ct, at, bt) in enumerate(true):
            if ct == c:
                iou = max(0, min(b, bt) - max(a, at)) / (max(b, bt) - min(a, at))
                if iou > best:
                    best, j = iou, i
        if j is not None and best >= k and j not in used:
            tp += 1
            used.add(j)
    return tp, len(pred) - tp, len(true) - tp


def seg_f1(tp, fp, fn):
    """F1 from pooled TP, FP and FN (1.0 when all are zero)."""
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
