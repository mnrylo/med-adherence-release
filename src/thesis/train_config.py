"""Train one configuration (every requested fold x seed) and save everything needed to evaluate it again.

    python -m thesis.train_config --experiment E1 --key cnn_bilstm__fs50_w200_ov50_acc_hp03__hpdef \
        --folds 0 1 2 --seeds 0 [--steps N] [--out DIR]

Data (conf/experiment_plan.json):
  * training: the fold's training gesture subjects composed into streams (E1: world W0all, gestures
    and rest only; E2: world W3int, inside daily activity of external *training* participants), plus
    the WISDM / InGesture training participants when the experiment's mix includes them;
  * validation (early stopping and every selection): streams composed from the fold's held-out
    gesture subjects inside external *calibration* participants, worlds W3int + W4long, identical
    for every configuration of every experiment;
  * tests (never used for any choice): the annotated sequences (data/labeled_output), the same
    sessions inside unseen daily activity, and the external test participants.

Layout (README.md): <config>/config.json, <config>/artifacts/ (labels and set membership shared by
all replicates) and one <config>/f<fold>_s<seed>/ directory per trained model. A replicate whose
manifest says "done" is skipped, so an interrupted configuration resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from core import datasets as T5
from core import sources as S
from core import streams as V
from core.bank import StreamBank
from core.hmm import Durations
from core.train import predict, train_supervised
from thesis import common as C
from thesis import metrics as M

SET_FILES = {
    "val": "val", "lab": "test_sequences", "wild": "test_wild", "ext_test": "test_external",
    "ext_cal": "external_calibration",
}


def parse_args():
    """Command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=sorted(C.EXPERIMENT_DATA))
    ap.add_argument("--key", required=True, help="config key (README.md)")
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2"],
                    help="fold indices, or 'all': deployment model on every training subject (replicate "
                         "fall_s<seed>; final weights, no held-out validation, test predictions only)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, help="override training steps (smoke tests)")
    ap.add_argument("--out", help="override the config directory (smoke tests)")
    return ap.parse_args()


def min_segment_s(label_streams, fs, fraction):
    """Per class: fraction × 10th percentile of the segment durations in the training streams
    (seconds; 0 for rest). Used by the no-post-processing timeline rule."""
    out = []
    for k in range(len(S.LABELS)):
        d = []
        for y in label_streams:
            for c, a, b in M.segments(y):
                if c == k:
                    d.append((b - a) / fs)
        out.append(0.0 if k == S.REST or not d else float(fraction * np.percentile(d, 10)))
    return out


def step_reporter(rep: Path, total: int):
    """Progress callback of train_supervised: <replicate>/progress.json, written at most every 5 s."""
    last, state = [0.0], {"total": total}

    def report(step, **m):
        state.update({"step": step, **{k: float(v) for k, v in m.items()}})
        if time.time() - last[0] > 5 or step == total:
            last[0] = time.time()
            C.write_json(rep / "progress.json", state)
    return report


def main():
    """Train the requested replicates of one configuration."""
    a = parse_args()
    factors = C.parse_key(a.key)
    cfg = C.training_config(a.experiment, factors)
    if a.steps:
        cfg["train"]["steps"] = a.steps
        cfg["train"]["eval_every"] = min(cfg["train"]["eval_every"], a.steps)
    out = Path(a.out).resolve() if a.out else C.config_dir(a.experiment, a.key)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    folds = [f if f == "all" else int(f) for f in a.folds]
    reps = [(f, s) for f in folds for s in a.seeds]

    def done(f, s):
        p = out / C.replicate_name(f, s) / "manifest.json"
        return p.exists() and json.loads(p.read_text()).get("status") == "done"

    todo = [(f, s) for f, s in reps if not done(f, s)]
    if not todo:
        log("all requested replicates done")
        return
    C.write_json(out / "config.json", cfg)
    d, fs = cfg["data"], cfg["data"]["fs"]
    win, stride = S.window_and_stride(d)
    hz = cfg["timeline_hz"]
    step10 = int(round(fs / hz))
    log(f"{a.experiment} {a.key}: window {win} samples, stride {stride}, replicates {todo}")

    src = T5.load_sources(cfg, log)
    ext = T5.external_split(src, d)
    gf = T5.gesture_folds(src["trials"], d)
    wild = V.embed_in_background(src["lab"], ext["ingesture"]["test"] + ext["wisdm"]["test"], fs, d["split_seed"],
                                 tuple(d["wild_context_s"]))
    use_ext = [k for k in ("ingesture", "wisdm") if cfg["train"]["mix"].get(k, 0) > 0]
    train_bg = [s for k in use_ext for s in ext[k]["train"]]
    n_folds = d["n_folds"]
    all_subjects = sorted({t.subject for t in src["trials"]})
    gf = {**dict(enumerate(gf)), "all": {"train": all_subjects, "val": []}}
    worlds = {}
    for f in [*range(n_folds), *(["all"] if "all" in folds else [])]:
        tr = [t for t in src["trials"] if t.subject in gf[f]["train"]]
        va = [t for t in src["trials"] if t.subject in (gf[f]["val"] or all_subjects)]
        seed_f = d["split_seed"] + 10 * (f if f != "all" else n_folds)
        worlds[f] = {
            "train": V.compose_world(tr, train_bg, cfg["world"], fs, seed_f, f"w{f}"),
            "val": [s for i, vw in enumerate(d["val_worlds"])
                    for s in V.compose_world(va, ext["ingesture"]["calibration"] + ext["wisdm"]["calibration"],
                                             {**T5.BASE_WORLD, **vw, "n_passes": d["val_passes"]}, fs,
                                             seed_f + 1 + i, f"v{f}_{i}")],
        }
    ext_test = ext["ingesture"]["test"] + ext["wisdm"]["test"]
    ext_cal = ext["ingesture"]["calibration"] + ext["wisdm"]["calibration"]

    art = out / "artifacts"
    if not (art / "manifest.json").exists():
        frames = {}
        labelled = src["lab"] + wild + ext_test + [s for f in worlds for s in worlds[f]["val"]]
        for s in labelled:
            frames[f"y/{s.sid}"] = s.y[::stride].astype(np.int8)
            frames[f"y10/{s.sid}"] = s.y[::step10].astype(np.int8)
            frames[f"edge/{s.sid}"] = T5.edge_dist(s.y, stride, fs)
        for f in [f for f in worlds if f != "all"]:
            for s in worlds[f]["train"]:
                frames[f"y/{s.sid}"] = s.y[::stride].astype(np.int8)
        art.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(art / "frames.npz", **frames)
        hours = lambda v: sum(len(s.x) for s in v) / fs / 3600
        man = {
            "experiment": a.experiment, "config_key": a.key, "fs": fs, "window": win, "stride": stride,
            "frame_rate": fs / stride, "window_s": win / fs, "stride_s": stride / fs, "timeline_hz": hz,
            "post_filter": d.get("post_filter"), "labels": S.LABELS, "world": cfg["world"],
            "lab": {s.sid: {"subject": s.subject, "protocol": s.protocol, "intake": s.intake,
                            "sequence": s.gesture_sequence, "protocol_match": bool(s.meta["protocol_match"])}
                    for s in src["lab"]},
            "wild": {s.sid: {"subject": s.subject, "protocol": s.protocol, "intake": s.intake,
                             "sequence": s.gesture_sequence, "session": s.meta["session"]} for s in wild},
            "folds": {str(f): {"train_subjects": gf[f]["train"], "val_subjects": gf[f]["val"],
                               "train": [s.sid for s in worlds[f]["train"]],
                               "train_hours": hours(worlds[f]["train"]),
                               "val": {s.sid: {"recipes": s.protocol, "intake": s.intake,
                                               "sequence": s.gesture_sequence} for s in worlds[f]["val"]}}
                      for f in worlds if f != "all"},
            "external": {name: {part: {"sids": [s.sid for s in v], "hours": hours(v)} for part, v in parts.items()}
                         for name, parts in ext.items()},
        }
        C.write_json(art / "manifest.json", man, indent=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit, dirty = C.git_commit()
    v = cfg["variant"]
    ext_all = [s for parts in ext.values() for p in parts.values() for s in p]
    fixed_sets = {"lab": src["lab"], "wild": wild, "ext_test": ext_test, "ext_cal": ext_cal}
    for f in sorted({f for f, _ in todo}, key=str):
        deploy = f == "all"
        bank = StreamBank(worlds[f]["train"] + worlds[f]["val"] + ext_all + src["lab"] + wild, win).to(device)
        train_y = [s.y for s in worlds[f]["train"]]
        dur = Durations.fit(train_y, stride, cfg["post"], fs / stride)
        tl_min = min_segment_s(train_y, fs, cfg["post"]["min_duration_fraction"])
        ts = max(1, int(round(d.get("train_stride_s", 0.1) * fs)))
        train_c = bank.centres([s.sid for s in worlds[f]["train"]], ts)
        train_c = train_c[bank.y[train_c] >= 0]
        pools = {"world_gesture": train_c[bank.y[train_c] != S.REST], "world": train_c}
        aux = {}
        for name, codes in (("ingesture", S.INGESTURE_CODES), ("wisdm", S.WISDM_CODES)):
            if name in use_ext:
                c = bank.centres([s.sid for s in ext[name]["train"]], ts)
                pools[name] = c[(bank.y[c] >= 0) | (bank.aux[c] >= 0)]
                aux[name] = len(codes)
        pools = {k: (c, cfg["train"]["mix"][k]) for k, c in pools.items() if cfg["train"]["mix"].get(k, 0) > 0}
        val_c = np.concatenate([bank.frame_centres(s.sid, stride) for s in worlds[f]["val"]])
        val_c = val_c[bank.y[val_c] >= 0]
        for seed in [s for ff, s in todo if ff == f]:
            rep = out / C.replicate_name(f, seed)
            run_id = f"{C.SLUGS[a.experiment]}/{a.key}/{rep.name}"
            rep.mkdir(exist_ok=True)
            manifest = {"run_id": run_id, "experiment": a.experiment, "config_key": a.key, "replicate": rep.name,
                        "factors": factors, "hyperparameters": cfg["hyperparameters"], "status": "running",
                        "git_commit": commit, "git_dirty": dirty, "created": C.now(),
                        "train_subjects": gf[f]["train"], "val_subjects": gf[f]["val"],
                        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
            C.write_json(rep / "manifest.json", manifest)
            t0 = time.time()
            try:
                log(f"== {rep.name}: pools " + ", ".join(f"{k}:{len(c)}" for k, (c, _) in pools.items()))
                tcfg = {**cfg["train"], "augment": cfg["augment"], "fs": fs, "label_mode": v["label_mode"],
                        "encoder_opts": {**v["encoder_opts"], "fs": fs}, "keep_last": deploy}
                model, info = train_supervised(v["encoder"], bank, pools, aux, val_c, len(S.LABELS), tcfg, seed,
                                               device, log, progress=step_reporter(rep, tcfg["steps"]),
                                               ckpt_path=rep / "checkpoint.pt")
                (rep / "predictions").mkdir(exist_ok=True)
                sets = fixed_sets if deploy else {"val": worlds[f]["val"], **fixed_sets}
                for name, streams in sets.items():
                    probs = {s.sid: predict(model, bank, bank.frame_centres(s.sid, stride), device).astype(np.float16)
                             for s in streams}
                    np.savez_compressed(rep / "predictions" / f"{SET_FILES[name]}.npz", **probs)
                durations = {"min_frames": dur.min_frames.tolist(), "p_stay": dur.p_stay.tolist()}
                torch.save({"state": model.state_dict(), "encoder": v["encoder"], "encoder_opts": tcfg["encoder_opts"],
                            "aux": aux, "n_classes": len(S.LABELS), "labels": S.LABELS, "window": win, "fs": fs},
                           rep / "model.pt")
                C.write_json(rep / "preprocessing.json", {
                    "sensor_columns": S.SENSORS, "units": {"acc": "g", "gyr": "deg/s"},
                    "raw_units_in_csv": {"acc": "m/s^2", "gyr": "deg/s"},
                    "fs_hz": fs, "resample_lowpass_hz": d["lowpass_hz"], "filter": d.get("post_filter"),
                    "window_samples": win, "window_s": win / fs, "stride_samples": stride, "overlap": d["overlap"],
                    "channels_used": ["acc_x", "acc_y", "acc_z"] if factors["signals"] == "acc" else S.SENSORS,
                    "input_layout": "(batch, 6, window_samples), all 6 channels; the model selects its channels",
                    "input_scale": "acc x1, gyr x1/100 (inside the model)", "classes": S.LABELS,
                    "window_label": "label of the centre sample"})
                with open(rep / "train_log.csv", "w") as fh:
                    fh.write("step,loss,val_macro_f1\n")
                    for h in info["history"]:
                        fh.write(f"{h['step']},{h['loss']:.6f},{h['val_macro_f1']:.6f}\n")
                C.write_json(rep / "train_info.json", {
                    "run_id": run_id, "fold": f, "seed": seed, "best_step": info["best_step"],
                    "best_val_window_macro_f1": info["best_val_macro_f1"], "seconds": info["seconds"],
                    "n_params": info["n_params"], "prior": info["prior"], "durations": durations,
                    "timeline_min_s": tl_min})
                m = M.write(out, rep)
                manifest.update({"status": "done", "finished": C.now(), "duration_s": time.time() - t0,
                                 "n_params": info["n_params"],
                                 "files": {"config": "../config.json", "model": "model.pt",
                                           "preprocessing": "preprocessing.json", "train_log": "train_log.csv",
                                           "train_info": "train_info.json", "metrics": "metrics.json",
                                           "sessions": "sessions.csv", "predictions": "predictions/"}})
                C.write_json(rep / "manifest.json", manifest)
                (rep / "checkpoint.pt").unlink(missing_ok=True)
                log(f"   {rep.name} done in {time.time() - t0:.0f}s: val window F1 {info['best_val_macro_f1']:.3f}, "
                    f"val seg F1@25 {m['val']['timeline']['seg_f1_25'] if 'val' in m else float('nan'):.3f}, "
                    f"test order exact {m['test_sequences']['order']['exact_match']:.3f}")
                del model
            except Exception as e:
                manifest.update({"status": "failed", "error": "".join(traceback.format_exception(e))[-4000:],
                                 "finished": C.now()})
                C.write_json(rep / "manifest.json", manifest)
                log(f"   {rep.name} FAILED: {e!r}")
                raise
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del bank
    log("finished")


if __name__ == "__main__":
    sys.exit(main())
