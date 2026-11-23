"""Sources, splits and labels shared by every experiment.

* ``load_sources``: the gesture trials, the lab sessions, InGesture and WISDM at the configured
  rate, with the preprocessing filter applied to each source recording (cached per signal
  setting under outputs/cache).
* ``external_split``: InGesture and WISDM split by participant into train / calibration / test.
* ``gesture_folds``: subject-wise folds over the gesture trials.
* ``edge_dist``: per frame, the distance to the nearest gesture change.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time

import numpy as np

from core import paths
from core import sources as S
from core import streams as V

TRAINING = paths.conf("training.json")
BASE_WORLD = TRAINING["world"]
"""Augmentation defaults of every composed world (validation worlds override some keys)."""

SOURCE_KEYS = ["fs", "lowpass_hz", "post_filter", "ingesture_drink_label", "wisdm_eating_as_rest"]


def load_sources(cfg: dict, log=print) -> dict:
    """{"trials", "lab", "ingesture", "wisdm"} at cfg["data"]["fs"], filtered.

    The result is cached under outputs/cache, keyed by the signal settings and the data paths.
    """
    d = cfg["data"]
    key_src = {k: d.get(k) for k in SOURCE_KEYS}
    key_src["paths"] = [str(p) for p in (paths.GESTURES, paths.LAB_SESSIONS, paths.INGESTURE, paths.WISDM)]
    key = hashlib.sha1(json.dumps(key_src, sort_keys=True).encode()).hexdigest()[:12]
    cache = paths.CACHE / f"sources_{key}.pkl"
    if cache.exists():
        log(f"sources from cache {cache.name}")
        return pickle.loads(cache.read_bytes())
    t0 = time.time()
    fs, lp, pf = d["fs"], d["lowpass_hz"], d.get("post_filter")
    filt = (lambda v: S.apply_post_filter(v, pf, fs)) if pf else (lambda v: v)
    out = {
        "trials": V.filter_trials(V.load_gesture_trials(paths.GESTURES, fs, lp), pf, fs),
        "lab": filt(V.load_lab_sessions(paths.LAB_SESSIONS, fs, lp)),
        "ingesture": filt(S.load_ingesture(paths.INGESTURE, fs, lp, d["ingesture_drink_label"])),
        "wisdm": filt(S.load_wisdm(paths.WISDM, fs, lp, d["wisdm_eating_as_rest"])),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(pickle.dumps(out))
    tmp.replace(cache)
    log(f"loaded and cached sources in {time.time() - t0:.0f}s")
    return out


def external_split(src: dict, d: dict) -> dict:
    """{"ingesture"|"wisdm": {"train"|"calibration"|"test": [Stream]}}, split by participant.

    The test participants are drawn first (external_test_fraction), then the calibration
    participants from the rest (external_calibration_fraction).
    """
    out = {}
    for name in ("ingesture", "wisdm"):
        tr, te = S.split_external(src[name], d["external_test_fraction"], d["split_seed"])
        rest = [s for s in src[name] if s.subject in tr]
        tr, cal = S.split_external(rest, d["external_calibration_fraction"], d["split_seed"] + 1)
        out[name] = {part: [s for s in src[name] if s.subject in subj]
                     for part, subj in (("train", tr), ("calibration", cal), ("test", te))}
    return out


def gesture_folds(trials, d: dict) -> list[dict]:
    """d["n_folds"] subject-wise folds: [{"train": [subjects], "val": [subjects]}]."""
    subjects = sorted({t.subject for t in trials})
    rng = np.random.default_rng(d["split_seed"])
    perm = [str(x) for x in rng.permutation(subjects)]
    k = d["n_folds"]
    return [{"val": sorted(perm[i::k]), "train": sorted(s for s in subjects if s not in perm[i::k])} for i in range(k)]


def edge_dist(y: np.ndarray, stride: int, fs: float) -> np.ndarray:
    """Per frame: seconds to the nearest change between two valid labels (inf if none)."""
    v = np.flatnonzero(y >= 0)
    pos = np.arange(0, len(y), stride)
    if len(v) < 2:
        return np.full(len(pos), np.inf, np.float32)
    ch = v[1:][y[v[1:]] != y[v[:-1]]]
    if len(ch) == 0:
        return np.full(len(pos), np.inf, np.float32)
    j = np.clip(np.searchsorted(ch, pos), 1, len(ch)) - 1
    k = np.clip(j + 1, 0, len(ch) - 1)
    return (np.minimum(np.abs(pos - ch[j] + 0.5), np.abs(pos - ch[k] + 0.5)) / fs).astype(np.float32)
