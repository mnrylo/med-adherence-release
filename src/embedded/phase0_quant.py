"""Phase 0: ONNX precision variants of the causal-trained finalist and their predictions.

    python -m embedded.phase0_quant [--reps ...]

For every replicate of the finalist retrained with the causal filter (emb.CAUSAL_DIR):
  * ``fp32``: torch -> ONNX (opset 17), the registry export;
  * ``fp16``: weights and activations in float16, float32 input and output
    (onnxruntime.transformers.float16);
  * ``int8_dynamic``: int8 weights, activations quantised at run time (quantize_dynamic) on
    LSTM, MatMul and Gemm; the convolution stays float because ONNX Runtime has no CPU
    ConvInteger kernel for int8 weights;
  * ``int8_static`` (optional, not in the default list): QDQ int8 with per-channel weights,
    calibrated on 2048 windows of the replicate's validation streams. It leaves the LSTM in
    float, so the file is not smaller. On f0_s0 it agreed with fp32 on only 91 % of the argmax
    (min-max calibration). Percentile and entropy calibration ran out of memory (23 GB).

Every variant predicts the same windows as the torch model, with ONNX Runtime on the CPU. The
predictions are saved where phase0_eval.py reads them, as variant ``causal_trained_<precision>``.

With --batch 1 --skip-build the predictions go to variant causal_trained_<precision>_b1: one window
per call, as a device runs it (INT8 dynamic quantises the activations over the whole batch).

Output: outputs/embedded/phase0/onnx/<replicate>/<precision>.onnx
        outputs/embedded/phase0/predictions/causal_trained_<precision>/<replicate>/<set>.npz
        outputs/embedded/phase0/onnx/info.json (sizes, max |diff| vs fp32, ops left in float)
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.transformers import float16

from embedded import emb as E
from embedded import phase0_predict as P0
from thesis import registry as REG
from core import sources as S
from core.bank import StreamBank

PRECISIONS = ["fp32", "fp16", "int8_dynamic"]
ONNX_DIR = E.PHASE0 / "onnx"


def windows(bank: StreamBank, centres: np.ndarray) -> np.ndarray:
    """(n, 6, win) float32 windows of the bank at the given centres."""
    idx = centres[:, None] + np.arange(bank.win)[None, :] - bank.half
    return bank.x[idx].transpose(0, 2, 1).astype(np.float32)


class Reader(CalibrationDataReader):
    """Calibration data for static quantisation."""
    def __init__(self, x: np.ndarray, batch=64):
        self.it = iter([{"imu": x[i:i + batch]} for i in range(0, len(x), batch)])

    def get_next(self):
        """Next calibration batch, or None."""
        return next(self.it, None)


def build(rep: str, calib: np.ndarray, precisions: list[str]) -> dict:
    """Write the ONNX precision variants of one replicate and their size and agreement."""
    d = ONNX_DIR / rep
    d.mkdir(parents=True, exist_ok=True)
    fp32 = d / "fp32.onnx"
    info = {"fp32": REG.export_onnx(E.CAUSAL_DIR / rep / "model.pt", fp32)}
    m16 = float16.convert_float_to_float16(onnx.load(fp32), keep_io_types=True)
    onnx.save(m16, d / "fp16.onnx")
    pre = d / "fp32_pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    quantize_dynamic(str(pre), str(d / "int8_dynamic.onnx"), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["LSTM", "MatMul", "Gemm"])
    if "int8_static" in precisions:
        quantize_static(str(pre), str(d / "int8_static.onnx"), Reader(calib), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    pre.unlink()
    for prec in ["fp32", *(p for p in precisions if p != "fp32")]:
        f = d / f"{prec}.onnx"
        g = onnx.load(f).graph
        info.setdefault(prec, {}).update({
            "bytes": f.stat().st_size,
            "ops": sorted({n.op_type for n in g.node}),
        })
    return info


def main():
    """Build the ONNX variants and predict with them."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", nargs="+", default=[*E.FOLD_REPS, E.DEPLOY_REP])
    ap.add_argument("--precisions", nargs="+", default=PRECISIONS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024,
                    help="windows per ONNX call; INT8 dynamic depends on it (1 = as on a device)")
    ap.add_argument("--skip-build", action="store_true", help="predict with the existing ONNX files")
    a = ap.parse_args()
    suffix = "" if a.batch == 1024 else f"_b{a.batch}"

    cfg = E.config()
    cfg["data"]["post_filter"] = E.VARIANTS["causal"]
    win, stride = S.window_and_stride(cfg["data"])
    sets = P0.build_streams(cfg)
    streams = [s for f in sets["val"].values() for s in f] + [s for k, v in sets.items() if k != "val" for s in v]
    bank = StreamBank(streams, win)
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    info_f = ONNX_DIR / "info.json"
    all_info = json.loads(info_f.read_text()) if info_f.exists() else {}
    rng = np.random.default_rng(0)
    for rep in a.reps:
        val = [s for f in sets["val"].values() for s in f] if rep == E.DEPLOY_REP else sets["val"][int(rep[1])]
        c = np.concatenate([bank.frame_centres(s.sid, stride) for s in val])
        calib = windows(bank, np.sort(rng.choice(c, 2048, replace=False)))
        if a.skip_build:
            info = all_info[rep]
        else:
            info = build(rep, calib, a.precisions)
        names = ["test_sequences", "test_wild", "test_external", *([] if rep == E.DEPLOY_REP else
                                                                   ["val", "external_calibration"])]
        ref = {}
        order = ["fp32"] + [p for p in a.precisions if p != "fp32"]
        for prec in (a.precisions if suffix else order):
            sess = ort.InferenceSession(str(ONNX_DIR / rep / f"{prec}.onnx"), so, providers=["CPUExecutionProvider"])
            out = E.pred_dir(f"causal_trained_{prec}{suffix}", rep)
            out.mkdir(parents=True, exist_ok=True)
            diffs, agree = [], []
            for name in names:
                probs = {}
                for s in (sets["val"][int(rep[1])] if name == "val" else sets[name]):
                    x = windows(bank, bank.frame_centres(s.sid, stride))
                    probs[s.sid] = np.concatenate([sess.run(None, {"imu": x[i:i + a.batch]})[0]
                                                   for i in range(0, len(x), a.batch)])
                    if prec == "fp32":
                        ref[s.sid] = probs[s.sid]
                    elif s.sid in ref:
                        diffs.append(np.abs(probs[s.sid] - ref[s.sid]).max())
                        agree.append(probs[s.sid].argmax(1) == ref[s.sid].argmax(1))
                np.savez_compressed(out / f"{name}.npz", **{k: v.astype(np.float16) for k, v in probs.items()})
            if prec != "fp32" and not suffix:
                info[prec].update({"max_abs_diff_vs_fp32": float(max(diffs)),
                                   "argmax_agreement_vs_fp32": float(np.mean(np.concatenate(agree)))})
            print(f"[{E.now()}] {rep} {prec}{suffix}: {info[prec]['bytes'] / 1024:.0f} KiB "
                  f"{'' if prec == 'fp32' or suffix else 'argmax agree %.4f' % info[prec]['argmax_agreement_vs_fp32']}",
                  flush=True)
        if not suffix:
            all_info[rep] = info
            E.write_json(info_f, all_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
