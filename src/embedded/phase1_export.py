"""Phase 1: deployment bundle, replay files and golden fixtures for the Android apps.

    python -m embedded.phase1_export

Deployed model: the finalist retrained with the causal filter, deployment replicate (fall_s0),
decoder parameters of the E2 selection, fixed-lag Viterbi L = 5 s, streaming LLR thresholds =
median over the 9 fold replicates of the per-step thresholds calibrated in phase 0.

Output:
  android/bundle/pipeline.json          every parameter of devicepipe.py (filter, HMMs, thresholds)
  android/bundle/model_{fp32,fp16,int8}.onnx
  outputs/embedded/phase1/replay/<set>.f32       raw streams, float32 little-endian (T, 6): acc g, gyro deg/s,
                                        100 Hz, NOT high-pass filtered (the device filters)
  outputs/embedded/phase1/replay/<set>.json      index: sid, sample offset, length, labels needed for latency
  android/core/src/test/resources/golden/*.json.gz  short streams with every reference output
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt_zi

from embedded import devicepipe as D
from core import paths
from embedded import emb as E
from embedded import phase0_eval as P0E
from embedded import phase0_predict as P0
from core import sources as S
from core.hmm import PostProcessor

ANDROID = paths.ANDROID
BUNDLE = ANDROID / "bundle"
REPLAY = E.PHASE1 / "replay"
GOLDEN = ANDROID / "core" / "src" / "test" / "resources" / "golden"


def streaming_variant() -> str:
    """Phase-0 variant used for streaming (outputs/embedded/phase0/summary.json)."""
    return json.loads((E.PHASE0 / "summary.json").read_text())["streaming_variant"]

LAG_S = 5
STEPS_S = P0E.STEPS_S
PRECISIONS = {"fp32": "fp32", "fp16": "fp16", "int8": "int8_dynamic"}


def bundle() -> dict:
    """pipeline.json content: filter, window, HMMs, emission scales, lag and streaming thresholds."""
    p = {k: v for k, v in E.postprocessing()["decoder"].items() if k in ("decoder", "ov", "st", "md", "t")}
    VARIANT = streaming_variant()
    R = P0E.VRep(E.DEPLOY_REP, VARIANT)
    seg = PostProcessor(R.durations(p["md"]), R.prior, R.post_cfg(p["st"]), P0E.HZ).seg
    pp, r = R.llr_pp(p)
    m1, m0 = pp.models["phased"]
    L = pd.read_csv(E.PHASE0 / "tables" / "llr_streaming.csv")
    L = L[(L.variant == VARIANT) & (L.replicate != E.DEPLOY_REP)]
    taus = {str(st): {b: float(g.tau.median()) for b, g in L[L.step_s == st].groupby("budget")} for st in STEPS_S}
    sos = butter(4, 0.3, btype="high", fs=R.man["fs"], output="sos")
    return {
        "created": E.now(),
        "model": {"config_key": E.CAUSAL_KEY, "replicate": E.DEPLOY_REP, "input": f"imu: float32 (batch, 6, {int(R.man['window'])})",
                  "output": "probabilities (batch, 5)", "files": {k: f"model_{k}.onnx" for k in PRECISIONS}},
        "labels": S.LABELS,
        "channels": ["acc_x_g", "acc_y_g", "acc_z_g", "gyr_x_dps", "gyr_y_dps", "gyr_z_dps"],
        "fs": float(R.man["fs"]), "window": int(R.man["window"]), "stride": int(R.man["stride"]),
        "timeline_hz": P0E.HZ,
        "filter": {"type": "butterworth highpass, accelerometer only, causal (sosfilt), steady-state start",
                   "order": 4, "cutoff_hz": 0.3, "sos": sos.tolist(), "zi": sosfilt_zi(sos).tolist()},
        "prior": [float(v) for v in R.prior], "emission_floor": float(P0E.PP.POST_CFG["emission_floor"]),
        "alpha": float(R.alpha(p)), "decoder_params": p,
        "segmentation_hmm": D.HMM.to_json(seg), "viterbi_lag_frames": LAG_S * P0E.HZ,
        "llr": {"hz": P0E.LLR_HZ, "alpha": float(R.alpha(p) * r), "episode_blocks": int(P0E.PP.POST_CFG["episode_s"] * P0E.LLR_HZ),
                "steps_s": STEPS_S, "thresholds": taus, "threshold_rule": "median over the 9 fold replicates of the "
                "phase-0 streaming thresholds of the same step (best validation balanced accuracy under the budget)",
                "intake_hmm": D.HMM.to_json(m1), "no_intake_hmm": D.HMM.to_json(m0)},
        "lookahead_s": R.lookahead_s(),
    }


def replay(cfg) -> dict:
    """Unfiltered test streams (the device filters), composed exactly as in phase 0."""
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["post_filter"] = None
    win, stride = S.window_and_stride(cfg["data"])
    sets = P0.build_streams(cfg)
    P0.check_labels({"val": {}, "test_sequences": [], "test_wild": sets["test_wild"]}, stride,
                    int(round(cfg["data"]["fs"] / cfg["timeline_hz"])))
    REPLAY.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ("test_wild", "test_external"):
        index, pos = [], 0
        with open(REPLAY / f"{name}.f32", "wb") as f:
            for s in sets[name]:
                x = np.ascontiguousarray(s.x, dtype="<f4")
                f.write(x.tobytes())
                row = {"sid": s.sid, "offset": pos, "length": len(x), "intake": int(s.intake)}
                if name == "test_wild":
                    y10 = s.y[::10]
                    off, n = int(s.meta["offset"]), int(s.meta["length"])
                    a, b = off // 10, -(-(off + n) // 10)
                    idx = np.flatnonzero(np.isin(y10[a:b], [S.LABELS.index("G3"), S.LABELS.index("G4")]))
                    i3 = np.flatnonzero(y10[a:b] == S.LABELS.index("G3"))
                    row.update({"session_start_s": off / 100.0, "session_length": n,
                                "intake_end_s": (a + idx[-1] + 1) / 10 if len(idx) and s.intake else None,
                                "g3_end_s": (a + i3[-1] + 1) / 10 if len(i3) and s.intake else None})
                index.append(row)
                pos += len(x)
        E.write_json(REPLAY / f"{name}.json", {"set": name, "fs": 100.0, "channels": 6, "dtype": "float32 LE",
                                               "units": "acc g, gyro deg/s, not high-pass filtered",
                                               "n_samples": pos, "hours": pos / 100 / 3600, "streams": index})
        out[name] = sets[name]
        print(f"[{E.now()}] {name}: {len(index)} streams, {pos / 360000:.2f} h, {pos * 24 / 2**20:.0f} MiB", flush=True)
    return out


def golden(b: D.Bundle, sets: dict, n_each=2):
    """A few short streams with every intermediate output, for the Kotlin tests."""
    clf = D.onnx_classifier(BUNDLE / "model_fp32.onnx", threads=8)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    picks = []
    wild = sorted(sets["test_wild"], key=lambda s: len(s.x))
    picks += [s for s in wild if s.intake][:n_each] + [s for s in wild if not s.intake][:1]
    picks += sorted(sets["test_external"], key=lambda s: len(s.x))[:1]
    for s in picks:
        r = D.run_stream(b, s.x, clf)
        doc = {"sid": s.sid, "model": "model_fp32.onnx", "n_samples": len(s.x),
               "x": s.x.astype(np.float64).tolist(),
               "filtered": r["filtered"].tolist(), "probs": r["probs"].astype(np.float64).tolist(),
               "timeline": r["timeline"].tolist(), "path": r["path"].tolist(), "blocks": r["blocks"].tolist(),
               "llr": {str(st): {"scores": v["scores"].tolist(), "ends": v["ends"].tolist(),
                                 "alarms": {k: a.tolist() for k, a in v["alarms"].items()}}
                       for st, v in r["llr"].items()}}
        with gzip.open(GOLDEN / f"{s.sid}.json.gz", "wt") as f:
            json.dump(doc, f)
        print(f"golden {s.sid}: {len(s.x)} samples, {len(r['probs'])} windows", flush=True)


def aligned_copy(src, dst, max_raw=64):
    """Copy an ONNX model, moving small tensors out of raw_data into the typed fields.

    A small raw_data string sits inside the protobuf string object (short-string optimisation), at
    an odd address. ONNX Runtime reads it as float/int64 in place, which raises SIGBUS (BUS_ADRALN) on
    32-bit ARM (the Galaxy Watch runs armeabi-v7a). The typed fields are separate, aligned arrays.
    """
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(src))

    def fix(t):
        if t.raw_data and len(t.raw_data) <= max_raw:
            arr = numpy_helper.to_array(t)
            t.CopyFrom(helper.make_tensor(t.name, t.data_type, list(t.dims), arr.flatten().tolist()))

    def walk(g):
        for t in g.initializer:
            fix(t)
        for n in g.node:
            for at in n.attribute:
                if at.HasField("t"):
                    fix(at.t)
                for sub in ([at.g] if at.HasField("g") else []) + list(at.graphs):
                    walk(sub)

    walk(m.graph)
    onnx.save(m, str(dst))


def main():
    """Write the bundle, the ONNX models, the replay files and the golden fixtures."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-replay", action="store_true")
    a = ap.parse_args()
    BUNDLE.mkdir(parents=True, exist_ok=True)
    E.write_json(BUNDLE / "pipeline.json", bundle())
    for k, v in PRECISIONS.items():
        aligned_copy(E.PHASE0 / "onnx" / E.DEPLOY_REP / f"{v}.onnx", BUNDLE / f"model_{k}.onnx")
    b = D.Bundle.load(BUNDLE / "pipeline.json")
    print(f"bundle: seg {len(b.seg.state_class)} states, intake {len(b.intake.state_class)}, "
          f"no-intake {len(b.no_intake.state_class)}; thresholds {b.thresholds}", flush=True)
    if not a.skip_replay:
        sets = replay(E.config())
        golden(b, sets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
