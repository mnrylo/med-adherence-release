"""Shared code of the decoder study: paths, the set allowlist and per-session records.

Every decoder reads the saved window probabilities of the E1/E2 replicates (outputs/thesis/runs)
through ``Rep``, which refuses any set outside ALLOWED: decoders are fitted and tuned on the
validation streams, thresholds use the external calibration participants, and the only test set
is test_wild (the lab sessions embedded in InGesture/WISDM activity of held-out participants).
Every set file opened is appended to index/sets_read.txt, so the allowlist can be audited.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from core import hsmm as P4
from core import paths
from thesis import common as C
from thesis import metrics as M
from thesis import postproc as PP

ROOT = paths.DECODERS

INDEX = ROOT / "index"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
TEST = "test_wild"
ALLOWED = ("val", TEST, "external_calibration", "test_external")
READ_LOG = INDEX / "sets_read.txt"


class Rep(PP.Replicate):
    """An E1/E2 replicate restricted to the allowed sets."""

    def __init__(self, rep_dir: Path):
        super().__init__(rep_dir)
        self.meta = {k: v for k, v in self.meta.items() if k in ALLOWED}

    def load(self, part):
        """Window probabilities of an allowed set (any other set raises PermissionError)."""
        if part not in ALLOWED:
            raise PermissionError(f"set {part!r} is outside the allowlist {ALLOWED}")
        if part not in self.probs:
            with open(READ_LOG, "a") as f:
                f.write(f"{self.dir / 'predictions' / (part + '.npz')}\n")
        return super().load(part)

    def truths(self, part):
        """10 Hz labels of every stream of a set."""
        return [self.frames[f"y10/{s}"].astype(np.int64) for s in self.sids(part)]

    def subject(self, part, sid):
        """Participant of a stream (empty for composed validation streams)."""
        return self.meta[part][sid].get("subject", "")


def session_rows(R: Rep, part: str, paths, decoder: str) -> list[dict]:
    """One row per stream: segment counts at every IoU, order and the G3 rule, so that metrics can be
    pooled over any grouping (replicate, participant) without decoding again."""
    rows = []
    for s, q, t in zip(R.sids(part), paths, R.truths(part)):
        ps, ts = P4.label_segments(q), P4.label_segments(t)
        po, to = PP.label_runs_to_sequence(q), R.meta[part][s]["sequence"]
        r = {"set": part, "sid": s, "subject": R.subject(part, s), "decoder": decoder,
             "intake": int(R.meta[part][s]["intake"]), "pred_g3": int(PP.G3 in q),
             "exact": int(po == to), "edit": P4.levenshtein(po, to) / max(1, len(to)),
             "n_true_seg": len(ts), "n_pred_seg": len(ps), "frames": len(t)}
        for k in M.SEG_K:
            tp, fp, fn = P4.segment_counts(ps, ts, k)
            r.update({f"tp{int(k * 100)}": int(tp), f"fp{int(k * 100)}": int(fp), f"fn{int(k * 100)}": int(fn)})
        rows.append(r)
    return rows


def write_json(path: Path, obj):
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def now() -> str:
    """Local time stamp."""
    return time.strftime("%Y-%m-%d %H:%M:%S")
