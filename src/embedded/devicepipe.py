"""Reference implementation of the on-device pipeline (phase 1), with the exact streaming semantics.

The Kotlin module (android/core) must reproduce these outputs. Everything here is written as whole-
stream array code, but every output depends only on the samples a device has at that moment, except
the explicit end-of-stream flush. The semantics, step by step:

1. **Filter.** The accelerometer goes through the causal 0.3 Hz high-pass (scipy ``sosfilt``, direct
   form II transposed), starting in the steady state of the first sample. The gyroscope is left as is.
   Units are g and deg/s.
2. **Windows.** Window j is centred on sample j * stride and spans [c - win/2, c + win/2).
   Samples before the stream start repeat the first sample (edge padding, as StreamBank). The window
   is classified as soon as its last sample arrives. At the end of a stream the remaining centres
   (j * stride < T) are flushed with the last sample repeated.
3. **Timeline (10 Hz).** Frame u (time u / 10 s) is the mean of the probabilities of windows lo..hi,
   lo = ceil((t - win/2) / stride), hi = floor((t + win/2) / stride), clipped to the windows that
   exist. It is final when window hi exists, or at the flush.
4. **Gesture Timeline.** Fixed-lag Viterbi on the ergodic HMM: frame u is decided when frame u + L
   arrives, by backtracking from the best state at u + L. At the flush the remaining frames take
   the full Viterbi path (final states only).
5. **Intake LLR (2 Hz).** Block k is the mean of 10 Hz frames 5k..5k+4. Every ``step`` blocks an
   episode of the last 80 blocks (40 s) is scored with the forward algorithm:
   log P(episode | intake) - log P(episode | no intake). Emissions are
   (log clip(p, floor, 1) - log prior) * alpha, with alpha = 1 / t at 10 Hz and 5 / t at 2 Hz.
6. **Alarm.** An alarm is raised at the first positive episode of every run of positive episodes
   (rising edge of score > threshold), at the time the episode ends.

Bundle: pipeline.json (see phase1_export.py). HMMs are stored as sparse transitions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt
from scipy.special import logsumexp


@dataclass
class HMM:
    """A decoding graph with dense log transitions, as stored sparsely in pipeline.json."""
    state_class: np.ndarray
    log_start: np.ndarray
    final: np.ndarray
    logA: np.ndarray

    @classmethod
    def from_json(cls, d: dict) -> "HMM":
        """Graph from its sparse JSON form."""
        S = len(d["state_class"])
        A = np.full((S, S), -np.inf)
        for i, j, v in d["transitions"]:
            A[i, j] = v
        ls = np.full(S, -np.inf)
        for i, v in d["start"]:
            ls[i] = v
        return cls(np.array(d["state_class"]), ls, np.array(d["final"], bool), A)

    @staticmethod
    def to_json(hmm) -> dict:
        """core.hmm.ExpandedHMM -> sparse dict."""
        A, ls = hmm.logA, hmm.log_start
        return {"state_class": [int(k) for k in hmm.state_class],
                "start": [[int(i), float(ls[i])] for i in np.flatnonzero(np.isfinite(ls))],
                "final": [bool(f) for f in hmm.final_mask],
                "transitions": [[int(i), int(j), float(A[i, j])] for i, j in zip(*np.nonzero(np.isfinite(A)))]}


@dataclass
class Bundle:
    """Every parameter of the device pipeline (pipeline.json)."""
    fs: float
    win: int
    stride: int
    hz: int
    llr_hz: int
    sos: np.ndarray
    zi: np.ndarray
    prior: np.ndarray
    floor: float
    alpha: float
    llr_alpha: float
    seg: HMM
    intake: HMM
    no_intake: HMM
    episode_blocks: int
    lag_frames: int
    thresholds: dict
    raw: dict

    @classmethod
    def load(cls, path: Path) -> "Bundle":
        """Bundle from pipeline.json."""
        d = json.loads(Path(path).read_text())
        return cls(d["fs"], d["window"], d["stride"], d["timeline_hz"], d["llr"]["hz"], np.array(d["filter"]["sos"]),
                   np.array(d["filter"]["zi"]), np.array(d["prior"]), d["emission_floor"], d["alpha"],
                   d["llr"]["alpha"], HMM.from_json(d["segmentation_hmm"]), HMM.from_json(d["llr"]["intake_hmm"]),
                   HMM.from_json(d["llr"]["no_intake_hmm"]), d["llr"]["episode_blocks"], d["viterbi_lag_frames"],
                   {int(k): v for k, v in d["llr"]["thresholds"].items()}, d)

    @property
    def lookahead_s(self) -> float:
        """Data needed after a frame before every window covering it exists: half a window plus one stride."""
        return (self.win / 2 + self.stride) / self.fs


def filter_stream(b: Bundle, x: np.ndarray) -> np.ndarray:
    """Causal high-pass of the accelerometer, steady-state start; gyroscope unchanged."""
    x = np.asarray(x, np.float64).copy()
    zi = b.zi[:, :, None] * x[0, :3][None, None, :]
    x[:, :3] = sosfilt(b.sos, x[:, :3], axis=0, zi=zi)[0]
    return x


def window_batch(b: Bundle, xf: np.ndarray) -> np.ndarray:
    """(n_windows, 6, win) float32; centres j * stride for every j * stride < T, edge-padded."""
    T, h = len(xf), b.win // 2
    centres = np.arange(0, T, b.stride)
    idx = np.clip(centres[:, None] + np.arange(b.win)[None, :] - h, 0, T - 1)
    return xf[idx].transpose(0, 2, 1).astype(np.float32)


def n_complete_windows(b: Bundle, T: int) -> int:
    """Windows whose last sample exists before the flush."""
    h = b.win // 2
    return max(0, (T - 1 - (b.win - h - 1)) // b.stride + 1) if T >= b.win - h else 0


def timeline(b: Bundle, probs: np.ndarray, n_frames: int) -> np.ndarray:
    """10 Hz timeline: frame u averages the windows that cover it."""
    t = np.arange(n_frames) / b.hz
    ds, hw = b.stride / b.fs, b.win / b.fs / 2
    n = len(probs)
    lo = np.clip(np.ceil((t - hw) / ds - 1e-9).astype(int), 0, n - 1)
    hi = np.clip(np.floor((t + hw) / ds + 1e-9).astype(int), 0, n - 1)
    out = np.empty((n_frames, probs.shape[1]))
    p = probs.astype(np.float64)
    for u in range(n_frames):
        out[u] = p[lo[u]:hi[u] + 1].mean(0)
    return out


def emissions(b: Bundle, post: np.ndarray, alpha: float) -> np.ndarray:
    """Scaled-likelihood emissions with scale alpha."""
    return (np.log(np.clip(post, b.floor, 1)) - np.log(b.prior)[None, :]) * alpha


def fixed_lag(hmm: HMM, logE: np.ndarray, L: int) -> np.ndarray:
    """Class path: frame u from the best state at u + L; the last L frames from the full path."""
    e = logE[:, hmm.state_class]
    T, S = e.shape
    back = np.zeros((T, S), np.int32)
    best = np.zeros(T, np.int64)
    delta = hmm.log_start + e[0]
    best[0] = delta.argmax()
    for t in range(1, T):
        cand = delta[:, None] + hmm.logA
        back[t] = cand.argmax(0)
        delta = cand.max(0) + e[t]
        best[t] = delta.argmax()
    full = np.empty(T, np.int64)
    full[-1] = np.where(hmm.final, delta, -np.inf).argmax()
    for t in range(T - 1, 0, -1):
        full[t - 1] = back[t, full[t]]
    if L >= T:
        return hmm.state_class[full]
    cur = best[L:].copy()
    t_idx = np.arange(L, T)
    for j in range(L):
        cur = back[t_idx - j, cur]
    return np.concatenate([hmm.state_class[cur], hmm.state_class[full[T - L:]]])


def forward(hmm: HMM, logE: np.ndarray) -> float:
    """Log-likelihood of an episode (forward algorithm, final states only)."""
    e = logE[:, hmm.state_class]
    alpha = hmm.log_start + e[0]
    for t in range(1, len(e)):
        alpha = logsumexp(alpha[:, None] + hmm.logA, axis=0) + e[t]
    return float(logsumexp(alpha[hmm.final]))


def blocks(b: Bundle, tl: np.ndarray) -> np.ndarray:
    """2 Hz blocks: means of consecutive 10 Hz frames."""
    r = b.hz // b.llr_hz
    n = len(tl) // r
    return tl[:n * r].reshape(n, r, -1).mean(1) if n else tl.mean(0, keepdims=True)


def episode_scores(b: Bundle, blk: np.ndarray, step_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Scores of the episodes ending every step (in blocks) and their end block indices."""
    s, W = int(round(step_s * b.llr_hz)), b.episode_blocks
    ends = np.arange(s, len(blk) + 1, s)
    if len(ends) == 0:
        ends = np.array([len(blk)])
    E = emissions(b, blk, b.llr_alpha)
    sc = np.array([forward(b.intake, E[max(0, x - W):x]) - forward(b.no_intake, E[max(0, x - W):x]) for x in ends])
    return sc, ends


def alarms(scores: np.ndarray, ends: np.ndarray, tau: float, llr_hz: int) -> np.ndarray:
    """Times (s, stream time of the episode end) of the rising edges of score > tau."""
    pos = scores > tau
    rise = pos & ~np.r_[False, pos[:-1]]
    return ends[rise] / llr_hz


def classify_windows(classify, w: np.ndarray, batch: int = 1) -> np.ndarray:
    """One window at a time by default, as the device does: the INT8 model quantises its activations
    with a scale computed over the whole batch, so batching changes its outputs."""
    return np.concatenate([classify(w[i:i + batch]) for i in range(0, len(w), batch)])


def run_stream(b: Bundle, x: np.ndarray, classify, steps=(2, 5, 10)) -> dict:
    """Every output of the device pipeline for one stream. classify: (B, 6, win) -> (B, 5)."""
    xf = filter_stream(b, x)
    probs = classify_windows(classify, window_batch(b, xf))
    return {"filtered": xf, **run_from_probs(b, probs, len(x), steps)}


def run_from_probs(b: Bundle, probs: np.ndarray, n_samples: int, steps=(2, 5, 10)) -> dict:
    """Steps 3-6 from the window probabilities."""
    n_frames = int(np.ceil(n_samples / (b.fs / b.hz)))
    tl = timeline(b, probs, n_frames)
    path = fixed_lag(b.seg, emissions(b, tl, b.alpha), b.lag_frames)
    blk = blocks(b, tl)
    out = {"probs": probs, "timeline": tl, "path": path, "blocks": blk, "llr": {}}
    for st in steps:
        sc, ends = episode_scores(b, blk, st)
        out["llr"][st] = {"scores": sc, "ends": ends,
                          "alarms": {k: alarms(sc, ends, v, b.llr_hz) for k, v in b.thresholds[st].items()}}
    return out


def onnx_classifier(model_path: Path, threads: int = 4):
    """Window classifier backed by an ONNX Runtime CPU session."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    return lambda w: sess.run(None, {"imu": w})[0]
