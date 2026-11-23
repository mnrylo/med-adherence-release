"""Phase 1: device-equivalent reference on the PC (the target the phones and watches must reproduce).

    python -m embedded.phase1_reference

Runs devicepipe.py on the replay files (unfiltered composed streams, filtered causally as one
continuous signal, as a device sees them) with the deployed ONNX model, and compares it with the
same pipeline fed with the phase-0 predictions (each source recording filtered before composition).
The difference between the two is the effect of filtering across the junctions of the synthetic
streams; it must be small for the replay to stand for the blind evaluation.

Output: outputs/embedded/phase1/reference/<precision>/{test_wild,test_external}.npz  window probabilities
        outputs/embedded/phase1/reference/<precision>/<set>_paths.npz                 Gesture Timeline (10 Hz)
        outputs/embedded/phase1/reference/<precision>/alarms.json                    alarms per stream/step/budget
        outputs/embedded/phase1/reference/summary.csv                                 metrics of both sources
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from embedded import devicepipe as D
from embedded import emb as E
from thesis import metrics as M
from embedded import phase0_eval as P0E
from embedded.phase1_export import BUNDLE, REPLAY, streaming_variant

OUT = E.PHASE1 / "reference"
_B = None


def load_set(name):
    """Replay index and streams of a set."""
    idx = json.loads((REPLAY / f"{name}.json").read_text())
    x = np.fromfile(REPLAY / f"{name}.f32", dtype="<f4").reshape(-1, 6)
    return idx, {r["sid"]: x[r["offset"]:r["offset"] + r["length"]] for r in idx["streams"]}


def _post(args):
    """Device back end on one stream's probabilities (worker)."""
    sid, probs, n = args
    r = D.run_from_probs(_B, probs, n, P0E.STEPS_S)
    return sid, r["path"], {st: (v["scores"], v["ends"], v["alarms"]) for st, v in r["llr"].items()}


def _init():
    """Load the bundle in a worker process."""
    global _B
    _B = D.Bundle.load(BUNDLE / "pipeline.json")


def evaluate(source: str, probs: dict, idx: dict, R, workers: int) -> tuple[list[dict], dict]:
    """Metrics and alarms of the device pipeline on a source of window probabilities."""
    b = D.Bundle.load(BUNDLE / "pipeline.json")
    lens = {s["sid"]: s["length"] for name in idx for s in idx[name]["streams"]}
    with Pool(workers, initializer=_init) as pool:
        res = {name: {sid: (path, llr) for sid, path, llr in
                      pool.map(_post, [(sid, p, lens[sid]) for sid, p in probs[name].items()], chunksize=4)}
               for name in probs}
    rows = []
    wild = {s["sid"]: s for s in idx["test_wild"]["streams"]}
    sids = R.sids("test_wild")
    seg, _ = P0E.PP.timeline_metrics(R, "test_wild", [res["test_wild"][s][0] for s in sids])
    hours = sum(lens[s] for s in res["test_external"]) / b.fs / 3600
    for st in P0E.STEPS_S:
        for bud in b.thresholds[st]:
            det, lat, lat3 = [], [], []
            for s in sids:
                al = res["test_wild"][s][1][st][2][bud]
                al = al[al >= wild[s]["session_start_s"]]
                det.append(len(al) > 0)
                if wild[s]["intake"] and len(al) and wild[s]["intake_end_s"] is not None:
                    lat.append(al[0] + b.lookahead_s - wild[s]["intake_end_s"])
                    if wild[s]["g3_end_s"] is not None:
                        lat3.append(al[0] + b.lookahead_s - wild[s]["g3_end_s"])
            yi = np.array([wild[s]["intake"] for s in sids])
            m = M.binary(yi, np.array(det, int))
            fa = sum(len(res["test_external"][s][1][st][2][bud]) for s in res["test_external"]) / hours
            rows.append({"source": source, "step_s": st, "budget": bud, "seg_f1_25": seg["timeline"]["seg_f1_25"],
                         "order_exact_match": seg["order"]["exact_match"], "sensitivity": m["sensitivity"],
                         "specificity": m["specificity"], "false_alarms_per_h": fa,
                         "latency_median_s": float(np.median(lat)), "latency_p95_s": float(np.percentile(lat, 95)),
                         "latency_g3_p95_s": float(np.percentile(lat3, 95))})
    return rows, res


def main():
    """Run the PC reference and compare the replay with phase 0."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", default="int8", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = OUT / a.precision
    out.mkdir(parents=True, exist_ok=True)
    b = D.Bundle.load(BUNDLE / "pipeline.json")
    clf = D.onnx_classifier(BUNDLE / f"model_{a.precision}.onnx", threads=1)
    idx, probs = {}, {}
    for name in ("test_wild", "test_external"):
        idx[name], xs = load_set(name)
        probs[name] = {}
        for sid, x in xs.items():
            w = D.window_batch(b, D.filter_stream(b, x))
            probs[name][sid] = D.classify_windows(clf, w)
        np.savez_compressed(out / f"{name}.npz", **probs[name])
        print(f"[{E.now()}] {name}: predicted {len(xs)} streams", flush=True)
    VARIANT = streaming_variant()
    R = P0E.VRep(E.DEPLOY_REP, VARIANT)
    rows, res = evaluate("replay (device-equivalent)", probs, idx, R, a.workers)
    ph0 = {name: {k: v.astype(np.float32) for k, v in R.load(name).items()} for name in probs}
    rows0, _ = evaluate("phase 0 (filtered per source)", ph0, idx, R, a.workers)
    agree = float(np.mean(np.concatenate([probs["test_wild"][s].argmax(1) == ph0["test_wild"][s].argmax(1)
                                          for s in probs["test_wild"]])))
    for name in res:
        np.savez_compressed(out / f"{name}_paths.npz", **{sid: path.astype(np.int8) for sid, (path, _) in res[name].items()})
    E.write_json(out / "alarms.json", {name: {sid: {str(st): {bud: al.tolist() for bud, al in v[2].items()}
                                                     for st, v in llr.items()}
                                              for sid, (_, llr) in res[name].items()} for name in res})
    df = pd.DataFrame(rows + rows0)
    df.insert(0, "precision", a.precision)
    f = OUT / "summary.csv"
    if f.exists():
        old = pd.read_csv(f)
        df = pd.concat([old[old.precision != a.precision], df])
    df.to_csv(f, index=False)
    print(f"window argmax agreement replay vs phase 0 (test_wild): {agree:.4f}")
    print(df[df.precision == a.precision].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
