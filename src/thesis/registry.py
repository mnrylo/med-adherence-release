"""Model registry: deployable copies of the selected models.

    python -m thesis.registry [--dry-run]

Models: the best E3 configuration of E1 and of E2 (index/selection/E3.json, per_experiment) and
the best configuration of each short-window track (index/selection/E3_windows.json). Each one
gets the decoder that E3 selected for it on validation (index/selection/E3.json, per_config).

For each model:
  1. deployment training: thesis.train_config --folds all --seeds 0 (replicate fall_s0), every
     training subject, final weights after N steps, N = median best step of the 9 fold runs
     (rounded to 500);
  2. models/<name>/v<k>/ with
       model.pt            the fall_s0 weights
       model.onnx          main head softmax, input (batch, 6, window_samples) float32
       preprocessing.json  from the run (units, fs, filter, window, stride, channels)
       postprocessing.json timeline grid, decoder and its parameters, duration models, prior,
                           intake LLR thresholds per false-alarm budget
       metrics.json        fold runs (mean, std over 3 folds x 3 seeds) with the selected decoder,
                           and the deployment model on the blind sets (no decoder)
       model_card.json     description, selection, inputs, outputs, metrics and limitations
  3. models/registry.csv, one row per model version.

A version directory is never overwritten: a new export of the same configuration with different
weights is v<k+1>. An existing version with the same deployment run is left as it is.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core import hsmm as P4
from core.hmm import Durations
from core.models import MultiHeadClassifier, build_encoder, main_head_for
from core.sources import LABELS
from thesis import common as C
from thesis import metrics as M

REGISTRY = C.MODELS / "registry.csv"
SEED = 0
SETS = ["val", "test_sequences", "test_wild", "test_external"]
REPORT_KEYS = ["window.macro_f1", "window.accuracy", "window.recall.G3", "window.precision.G3",
               "timeline.macro_f1", "timeline.seg_f1_25", "order.exact_match", "order.edit_distance",
               "intake_g3.balanced_accuracy", "intake_g2_g3.balanced_accuracy",
               "intake_llr_fa05.sensitivity", "intake_llr_fa05.specificity", "intake_llr_fa05.balanced_accuracy",
               "intake_llr_fa10.sensitivity", "intake_llr_fa10.specificity", "intake_llr_fa10.balanced_accuracy",
               "external.gesture_events_per_hour", "external.g3_events_per_hour",
               "external.llr_fa05_events_per_hour", "external.llr_fa10_events_per_hour"]


def models_to_export() -> list[dict]:
    """The models of the registry: best E3 configuration per experiment and the best of every short-window track."""
    e3 = json.loads((C.SELECTION / "E3.json").read_text())
    out = [{"experiment": exp, "config_key": v["config_key"], "role": "finalist",
            "selection_file": "index/selection/E3.json", "stage": "E3 (best decoder of the stage-F finalists)"}
           for exp, v in e3["per_experiment"].items()]
    win = json.loads((C.SELECTION / "E3_windows.json").read_text())
    for track, v in win["configs"].items():
        out.append({"experiment": v["experiment"], "config_key": v["config_key"], "role": "window_ref",
                    "selection_file": f"index/selection/{track}_F.json",
                    "stage": f"short-window track {track} (A -> B -> C -> F)"})
    for m in out:
        m["decoder_key"] = e3["per_config"][f"{m['experiment']}/{m['config_key']}"]["decoder_key"]
        f = C.parse_key(m["config_key"])
        m["name"] = "-".join([m["experiment"].lower(), f["model"], f"fs{f['fs_hz']}",
                              f"w{int(round(f['window_s'] * 100)):03d}", f"ov{int(round(f['overlap'] * 100)):02d}",
                              C.SIGNAL_TAG[f["signals"]], "hp03" if f["filter"] == "hp0.3" else "raw"])
    return out


def fold_runs(m) -> list:
    """The fold replicates of a model's configuration."""
    d = C.config_dir(m["experiment"], m["config_key"])
    return sorted(p for p in d.glob("f[0-9]_s*") if (p / "train_info.json").exists())


def deployment_steps(m) -> int:
    """Training steps of the deployment model: median best step of the fold runs, rounded to 500."""
    steps = [json.loads((p / "train_info.json").read_text())["best_step"] for p in fold_runs(m)]
    return int(max(500, 500 * round(np.median(steps) / 500)))


def train_deployment(m) -> Path:
    """Train the deployment replicate fall_s0 (if not done) and return its directory."""
    rep = C.config_dir(m["experiment"], m["config_key"]) / f"fall_s{SEED}"
    man = rep / "manifest.json"
    if man.exists() and json.loads(man.read_text()).get("status") == "done":
        return rep
    cmd = C.module_cmd("thesis.train_config", "--experiment", m["experiment"], "--key", m["config_key"],
                       "--folds", "all", "--seeds", SEED, "--steps", deployment_steps(m))
    rep.parent.mkdir(parents=True, exist_ok=True)
    with open(rep.parent / "stdout.txt", "a") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT,
                       env=C.child_env(PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    return rep


class Probabilities(torch.nn.Module):
    """Main-head softmax of a trained classifier (the deployment graph)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """(B, 6, L) window -> (B, 5) main-head softmax."""
        return torch.softmax(self.model(x)[0], 1)


def load_model(path: Path):
    """Deployment graph and checkpoint dict of a model.pt."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc = build_encoder(ck["encoder"], ck["window"], **ck["encoder_opts"])
    model = MultiHeadClassifier(enc, ck["n_classes"], ck["aux"], main_head=main_head_for(ck["encoder"]))
    model.load_state_dict(ck["state"])
    return Probabilities(model).eval(), ck


def export_onnx(model_pt: Path, out: Path) -> dict:
    """Export the main-head softmax to ONNX (opset 17) and check it against torch on random windows."""
    import onnxruntime as ort
    net, ck = load_model(model_pt)
    x = torch.randn(64, 6, ck["window"]) * torch.tensor([0.3] * 3 + [50.0] * 3)[None, :, None]
    torch.onnx.export(net, (x[:1],), str(out), input_names=["imu"], output_names=["probabilities"],
                      dynamic_axes={"imu": {0: "batch"}, "probabilities": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    net.eval()
    with torch.no_grad():
        ref = net(x).numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"imu": x.numpy()})[0]
    return {"opset": 17, "input": f"imu: float32 (batch, 6, {ck['window']})", "output": "probabilities: (batch, 5)",
            "max_abs_diff_vs_torch": float(np.abs(ref - got).max())}


def postprocessing(m, dep: Path) -> dict:
    """Decoder and intake parameters for deployment (E3 choices; duration models from every
    fold's validation streams, as in E3)."""
    cfg_dir = C.config_dir(m["experiment"], m["config_key"])
    grid = json.loads((C.SELECTION / "E3_grid.json").read_text())["configs"][f"{C.SLUGS[m['experiment']]}/{m['config_key']}"]
    dec = next(v for v in grid.values() if v["decoder_key"] == m["decoder_key"])
    p = dec["params"]
    man = json.loads((cfg_dir / "artifacts" / "manifest.json").read_text())
    frames = np.load(cfg_dir / "artifacts" / "frames.npz")
    y10 = [frames[f"y10/{s}"].astype(np.int64) for f in man["folds"].values() for s in f["val"]]
    md = p.get("md", 0.0)
    dur = Durations.fit(y10, 1, {"min_duration_fraction": md}, 10) if md > 0 else None
    dpar = P4.fit_log_normal(y10)
    thr = {}
    for rep in fold_runs(m):
        mj = json.loads((C.POSTPROC / rep.relative_to(C.RUNS) / m["decoder_key"] / "metrics.json").read_text())
        for k, v in mj["llr_thresholds"].items():
            thr.setdefault(k, []).append(v)
        llr_params = mj["llr_hmm_params"]
    info = json.loads((dep / "train_info.json").read_text())
    post = C.TRAINING["post"]
    return {
        "timeline": {"hz": 10, "construction": "each 0.1 s frame averages the softmax of every window covering it "
                                               "(window j centred on sample j x stride)"},
        "decoder": {"key": m["decoder_key"], **p,
                    "emissions": "log(p / prior) per frame / temperature t",
                    "self_transition_meaning": "probability per second that rest (G5) continues (hmm only)",
                    "decode_every_nth_window": 1 if p["decoder"] == "none" or p["ov"] == man_ov(man) else 2},
        "prior": info["prior"],
        "timeline_rule_min_s": info["timeline_min_s"],
        "hmm_durations_10hz": {"min_frames": dur.min_frames.tolist(), "p_stay": dur.p_stay.tolist()} if dur else None,
        "hsmm_log_normal_10hz": {LABELS[k]: {"mu": v[0], "sigma": v[1], "p99_frames": v[2]} for k, v in dpar.items()},
        "intake": {"llr": {"model": "phased HMM intake vs no-intake (core.hmm)", "grid_hz": 2,
                           "episode_s": post["episode_s"], "episode_step_s": post["episode_step_s"],
                           "hmm_params": llr_params,
                           "thresholds": {k: float(np.median(v)) for k, v in thr.items()},
                           "threshold_rule": "median over the 9 fold runs of the threshold with the best validation "
                                             "balanced accuracy under the false-event budget (per hour) on the "
                                             "external calibration participants"},
                   "rules": {"g3": "a G3 segment in the decoded timeline", "g2_g3": "G2 followed later by G3"}},
        "post_config": post,
    }


def man_ov(man) -> int:
    """Overlap in percent of a run manifest."""
    return int(round(100 * (1 - man["stride"] / man["window"])))


def metrics(m, dep: Path) -> dict:
    """Fold-run metrics with the selected decoder (mean, std) and the deployment model's metrics."""
    df = pd.read_csv(C.INDEX / "postproc.csv")
    d = df[(df.experiment == m["experiment"]) & (df.config_key == m["config_key"]) & (df.decoder_key == m["decoder_key"])]
    out = {"source": "fold runs, selected decoder; mean and std over 3 folds x 3 seeds", "n_runs": int(len(d)),
           "decoder_key": m["decoder_key"], "sets": {}}
    for s in SETS:
        out["sets"][s] = {k: {"mean": float(d[f"{s}.{k}"].mean()), "std": float(d[f"{s}.{k}"].std())}
                          for k in REPORT_KEYS if f"{s}.{k}" in d}
    dm = json.loads((dep / "metrics.json").read_text())
    out["deployment_model_no_decoder"] = M.flatten(dm)
    return out


def next_version(name: str, dep_run: str) -> tuple[int, bool]:
    """(version, exists) for a model name: the existing version built from the same deployment run, else the next one.
    """
    base = C.MODELS / name
    vs = sorted(int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit())
    for v in vs:
        card = json.loads((base / f"v{v}" / "model_card.json").read_text())
        if card.get("deployment_run") == dep_run:
            return v, True
    return (vs[-1] + 1 if vs else 1), False


def export(m) -> dict:
    """Write one model version and return its registry row."""
    dep = train_deployment(m)
    dep_run = str(dep.relative_to(C.RUNS))
    v, exists = next_version(m["name"], dep_run)
    d = C.MODELS / m["name"] / f"v{v}"
    if not exists:
        d.mkdir(parents=True)
        shutil.copy2(dep / "model.pt", d / "model.pt")
        shutil.copy2(dep / "preprocessing.json", d / "preprocessing.json")
        onnx = export_onnx(d / "model.pt", d / "model.onnx")
        post = postprocessing(m, dep)
        C.write_json(d / "postprocessing.json", post)
        met = metrics(m, dep)
        C.write_json(d / "metrics.json", met)
        pre = json.loads((dep / "preprocessing.json").read_text())
        tinfo = json.loads((dep / "train_info.json").read_text())
        commit, _ = C.git_commit()
        ts = met["sets"]["test_sequences"]
        card = {
            "model_name": m["name"], "version": v, "role": m["role"],
            "purpose": "Medication-intake gesture recognition from wrist IMU (G1 open container, G2 pick up pill, "
                       "G3 pill to mouth, G4 drink water, G5 rest/other); Gesture Timeline, Gesture Order and "
                       "intake decision after decoding.",
            "config_key": m["config_key"], "experiment": m["experiment"],
            "source_runs": [str(p.relative_to(C.RUNS)) for p in fold_runs(m)],
            "deployment_run": dep_run, "deployment_steps": tinfo["best_step"],
            "selection": {"experiment": m["experiment"], "stage": m["stage"],
                          "rule": "max mean val.timeline.seg_f1_25, tie-break val window/timeline macro F1",
                          "selection_file": m["selection_file"]},
            "input": {"fs_hz": pre["fs_hz"], "window_s": pre["window_s"], "overlap": pre["overlap"],
                      "channels": pre["channels_used"], "units": pre["units"],
                      "filter": json.dumps(pre["filter"]) if pre["filter"] else "none"},
            "output": {"classes": LABELS, "decoder": m["decoder_key"],
                       "intake_threshold": post["intake"]["llr"]["thresholds"].get("fa10")},
            "metrics": met["sets"],
            "headline": {k: ts[k]["mean"] for k in ("window.macro_f1", "timeline.seg_f1_25", "order.exact_match")
                         if k in ts},
            "onnx": onnx,
            "limitations": [
                "Filters (resampling low-pass, 0.3 Hz high-pass) are zero-phase over the whole stream (offline); "
                "a streaming deployment needs a causal equivalent and should be re-validated.",
                "Trained on 15 subjects performing scripted gestures composed into synthetic streams; blind tests "
                "are other subjects' annotated sequences, alone and embedded in daily activity.",
                "Intake thresholds are medians over fold models; recalibrate on the target population if possible.",
                "The deployment model has no held-out validation: reported metrics come from the fold runs.",
            ],
            "files": {"weights": "model.pt", "onnx": "model.onnx", "preprocessing": "preprocessing.json",
                      "postprocessing": "postprocessing.json", "metrics": "metrics.json"},
            "git_commit": commit, "created": C.now(),
        }
        C.write_json(d / "model_card.json", card)
    card = json.loads((d / "model_card.json").read_text())
    ts = card["metrics"]["test_sequences"]
    tw = card["metrics"]["test_wild"]
    ext = card["metrics"]["test_external"]
    g = lambda s, k: s.get(k, {}).get("mean")
    return {"model_name": m["name"], "version": v, "role": m["role"], "experiment": m["experiment"],
            "config_key": m["config_key"], "decoder_key": m["decoder_key"], "deployment_status": "candidate",
            "test_sequences.window.macro_f1": g(ts, "window.macro_f1"),
            "test_sequences.timeline.seg_f1_25": g(ts, "timeline.seg_f1_25"),
            "test_sequences.order.exact_match": g(ts, "order.exact_match"),
            "test_wild.timeline.seg_f1_25": g(tw, "timeline.seg_f1_25"),
            "test_wild.intake_llr_fa10.balanced_accuracy": g(tw, "intake_llr_fa10.balanced_accuracy"),
            "test_external.llr_fa10_events_per_hour": g(ext, "external.llr_fa10_events_per_hour"),
            "path": str(d.relative_to(C.ROOT)), "created": card["created"]}


def main():
    """Export every model and update models/registry.csv."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ms = models_to_export()
    for m in ms:
        print(f"{m['name']}: {m['config_key']} + {m['decoder_key']} ({m['role']}), "
              f"deployment steps {deployment_steps(m)}", flush=True)
    if a.dry_run:
        return
    rows = []
    for m in ms:
        rows.append(export(m))
        print(f"[{C.now()}] exported {rows[-1]['path']}", flush=True)
    old = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    reg = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["model_name", "version"], keep="last")
    C.MODELS.mkdir(exist_ok=True)
    reg.to_csv(REGISTRY, index=False)
    print(reg[["model_name", "version", "decoder_key", "test_sequences.order.exact_match"]].to_string(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
