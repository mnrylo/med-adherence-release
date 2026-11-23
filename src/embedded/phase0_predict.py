"""Phase 0: window probabilities of the E2 finalist on streams preprocessed with a filter variant.

    python -m embedded.phase0_predict --variant causal

The streams are rebuilt exactly as thesis.train_config builds them:
  * same sources and splits;
  * the high-pass applied to every source recording before composition;
  * the same composition seeds.
Only the filter changes. The composition draws depend on stream lengths, not on signal values, so
the labels are identical to the saved artifacts; this script checks that.

Variants (emb.VARIANTS):
  * ``zerophase``: as trained. It must reproduce the saved predictions of the E2 runs
    (sanity check of this script).
  * ``causal``: the forward-only filter a device can run.

Every fold replicate predicts its own validation streams, the external calibration participants
and the test sets. fall_s0, the deployment model, predicts the test sets only.

Output: outputs/embedded/phase0/predictions/<variant>/<replicate>/<set>.npz (float16, as in the E2 runs)
        outputs/embedded/phase0/predictions/<variant>/streams.json (wild offsets, label check)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from embedded import emb as E
from core import datasets as T5
from core import streams as V
from core import sources as S
from core.bank import StreamBank
from core.models import KERAS_STYLE, MultiHeadClassifier, build_encoder, main_head_for
from core.train import predict


def log(msg):
    """Time-stamped print."""
    print(f"[{E.now()}] {msg}", flush=True)


def build_streams(cfg: dict) -> dict:
    """Validation streams per fold and the fixed sets, as thesis.train_config composes them."""
    d, fs = cfg["data"], cfg["data"]["fs"]
    src = T5.load_sources(cfg, log)
    ext = T5.external_split(src, d)
    gf = T5.gesture_folds(src["trials"], d)
    wild = V.embed_in_background(src["lab"], ext["ingesture"]["test"] + ext["wisdm"]["test"], fs, d["split_seed"],
                                 tuple(d["wild_context_s"]))
    ext_cal = ext["ingesture"]["calibration"] + ext["wisdm"]["calibration"]
    val = {}
    for f in range(d["n_folds"]):
        va = [t for t in src["trials"] if t.subject in gf[f]["val"]]
        seed_f = d["split_seed"] + 10 * f
        val[f] = [s for i, vw in enumerate(d["val_worlds"])
                  for s in V.compose_world(va, ext_cal, {**T5.BASE_WORLD, **vw, "n_passes": d["val_passes"]}, fs,
                                           seed_f + 1 + i, f"v{f}_{i}")]
    return {"val": val, "test_sequences": src["lab"], "test_wild": wild,
            "test_external": ext["ingesture"]["test"] + ext["wisdm"]["test"], "external_calibration": ext_cal}


def check_labels(sets: dict, stride: int, step10: int) -> dict:
    """The rebuilt streams must carry the saved labels (same composition)."""
    frames = np.load(E.CONFIG_DIR / "artifacts" / "frames.npz")
    streams = [s for f in sets["val"].values() for s in f] + sets["test_sequences"] + sets["test_wild"]
    bad = [s.sid for s in streams
           if not (np.array_equal(frames[f"y/{s.sid}"], s.y[::stride].astype(np.int8))
                   and np.array_equal(frames[f"y10/{s.sid}"], s.y[::step10].astype(np.int8)))]
    if bad:
        raise SystemExit(f"{len(bad)} rebuilt streams differ from the saved labels, e.g. {bad[:3]}")
    return {"checked_streams": len(streams), "label_mismatches": 0}


def load_model(rep: str, device):
    """Trained classifier of a replicate, in the mixed-precision mode it was trained with."""
    ck = torch.load(E.CONFIG_DIR / rep / "model.pt", map_location="cpu", weights_only=False)
    enc = build_encoder(ck["encoder"], ck["window"], **ck["encoder_opts"])
    model = MultiHeadClassifier(enc, ck["n_classes"], ck["aux"], main_head=main_head_for(ck["encoder"]))
    model.load_state_dict(ck["state"])
    model.amp = ck["encoder"] not in KERAS_STYLE
    model.amp_dtype = torch.float16 if ck["encoder"] in KERAS_STYLE else torch.bfloat16
    return model.to(device).eval()


def main():
    """Predict every replicate on the streams of one filter variant."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=sorted(E.VARIANTS))
    ap.add_argument("--reps", nargs="+", default=[*E.FOLD_REPS, E.DEPLOY_REP])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = E.config()
    cfg["data"]["post_filter"] = E.VARIANTS[a.variant]
    d, fs = cfg["data"], cfg["data"]["fs"]
    win, stride = S.window_and_stride(d)
    step10 = int(round(fs / cfg["timeline_hz"]))
    sets = build_streams(cfg)
    check = check_labels(sets, stride, step10)
    log(f"variant {a.variant}: labels identical on {check['checked_streams']} streams")
    E.write_json(E.PHASE0 / "predictions" / a.variant / "streams.json", {
        "variant": a.variant, "post_filter": d["post_filter"], **check,
        "test_wild": {s.sid: {"offset": int(s.meta["offset"]), "length": int(s.meta["length"])}
                      for s in sets["test_wild"]}})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_streams = [s for f in sets["val"].values() for s in f] + [s for k, v in sets.items() if k != "val" for s in v]
    bank = StreamBank(all_streams, win).to(device)
    for rep in a.reps:
        out = E.pred_dir(a.variant, rep)
        names = ["test_sequences", "test_wild", "test_external", *([] if rep == E.DEPLOY_REP else
                                                                   ["val", "external_calibration"])]
        if not a.force and all((out / f"{n}.npz").exists() for n in names):
            log(f"{rep}: done")
            continue
        model = load_model(rep, device)
        out.mkdir(parents=True, exist_ok=True)
        fold = None if rep == E.DEPLOY_REP else int(rep[1])
        for name in names:
            streams = sets["val"][fold] if name == "val" else sets[name]
            probs = {s.sid: predict(model, bank, bank.frame_centres(s.sid, stride), device).astype(np.float16)
                     for s in streams}
            np.savez_compressed(out / f"{name}.npz", **probs)
        log(f"{rep}: {', '.join(names)}")
        del model
    return 0


if __name__ == "__main__":
    sys.exit(main())
