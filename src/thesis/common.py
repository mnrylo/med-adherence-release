"""Shared definitions of experiments E1-E3: output paths, configuration keys, training configs.

A configuration is a dict of factors (model, fs_hz, window_s, overlap, signals, filter, hp).
``config_key`` turns it into the readable directory name described in README.md, and
``training_config`` turns it into the full training config consumed by ``thesis.train_config``
(conf/training.json with the factors and the experiment's world and training mix applied).
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from core import paths

ROOT = paths.THESIS
RUNS = ROOT / "runs"
POSTPROC = ROOT / "postproc"
INDEX = ROOT / "index"
SELECTION = INDEX / "selection"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
PY = sys.executable

PLAN = paths.conf("experiment_plan.json")
TRAINING = paths.conf("training.json")
WORLD_CONF = paths.conf("worlds.json")

EXPERIMENTS = {e["id"]: e for e in PLAN["experiments"]}
SLUGS = {e["id"]: e["slug"] for e in PLAN["experiments"]}

DEFAULT_HP = WORLD_CONF["default_hyperparameters"]
HP_GRID = EXPERIMENTS["E1"]["hyperparameter_grid"]
WORLDS = WORLD_CONF["worlds"]
VAL_WORLDS = [WORLDS[w] for w in WORLD_CONF["validation_worlds"]]
EXPERIMENT_DATA = WORLD_CONF["experiments"]

SIGNAL_TAG = WORLD_CONF["signals"]
FILTER_TAG = {k: v["tag"] for k, v in WORLD_CONF["filters"].items()}
FILTER_SPEC = {k: v["spec"] for k, v in WORLD_CONF["filters"].items()}
FACTORS = ["model", "fs_hz", "window_s", "overlap", "signals", "filter", "hp"]


def hp_points() -> list[dict]:
    """The hyperparameter grid in key order; hp<NN> indexes this list."""
    keys = list(HP_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(HP_GRID[k] for k in keys))]


def hp_values(hp: str) -> dict:
    """Hyperparameters of an hp tag: "hpdef" (defaults) or "hp<NN>" (grid point NN)."""
    return dict(DEFAULT_HP) if hp == "hpdef" else hp_points()[int(hp[2:])]


def config_key(c: dict) -> str:
    """Readable directory name of a configuration (README.md, "Config key")."""
    return (f"{c['model']}__fs{int(c['fs_hz'])}_w{int(round(c['window_s'] * 100)):03d}"
            f"_ov{int(round(c['overlap'] * 100)):02d}_{SIGNAL_TAG[c['signals']]}_{FILTER_TAG[c['filter']]}__{c['hp']}")


def parse_key(key: str) -> dict:
    """Inverse of ``config_key``."""
    model, sig, hp = key.split("__")
    fs, w, ov, s, f = sig.split("_")
    return {"model": model, "fs_hz": int(fs[2:]), "window_s": int(w[1:]) / 100, "overlap": int(ov[2:]) / 100,
            "signals": {v: k for k, v in SIGNAL_TAG.items()}[s], "filter": {v: k for k, v in FILTER_TAG.items()}[f],
            "hp": hp}


def replicate_name(fold, seed) -> str:
    """Replicate directory name: f<fold>_s<seed> (fold "all" for the deployment model)."""
    return f"f{fold}_s{seed}"


def config_dir(experiment: str, key: str) -> Path:
    """Run directory of a configuration."""
    return RUNS / SLUGS[experiment] / key


def training_config(experiment: str, c: dict) -> dict:
    """Full training config of one configuration of an experiment."""
    cfg = json.loads(json.dumps(TRAINING))
    hp = hp_values(c["hp"])
    d = cfg["data"]
    d.update({"fs": float(c["fs_hz"]), "lowpass_hz": min(0.4 * c["fs_hz"], 40.0), "window_s": c["window_s"],
              "overlap": c["overlap"], "post_filter": FILTER_SPEC[c["filter"]],
              "val_worlds": VAL_WORLDS})
    ed = EXPERIMENT_DATA[experiment]
    cfg["world"] = {**cfg["world"], **WORLDS[ed["world"]]}
    cfg["world_name"] = ed["world"]
    cfg["train"].update({"mix": ed["mix"], "lr": hp["lr"], "batch_size": hp["batch_size"]})
    cfg["variant"] = {"name": c["model"], "label_mode": "hard", "encoder": c["model"],
                      "encoder_opts": {"filters": hp["conv_filters"], "lstm_units": hp["lstm_units"],
                                       "dropout": hp["dropout"], "kernel_size": hp["kernel_size"],
                                       "channels": c["signals"]}}
    cfg["experiment"] = experiment
    cfg["factors"] = dict(c)
    cfg["hyperparameters"] = hp
    cfg["config_key"] = config_key(c)
    cfg["timeline_hz"] = PLAN["common_protocol"]["fixed_unless_stated"]["timeline_grid_hz"]
    return cfg


def module_cmd(module: str, *args) -> list[str]:
    """Command line that runs a module of src/ with the current interpreter."""
    return [PY, "-m", module, *map(str, args)]


def child_env(**extra) -> dict:
    """Environment of a child process: src/ on PYTHONPATH, plus extra variables."""
    pp = os.pathsep.join(p for p in (str(paths.SRC), os.environ.get("PYTHONPATH", "")) if p)
    return {**os.environ, "PYTHONPATH": pp, **extra}


def git_commit() -> tuple[str, bool]:
    """(commit, dirty) of the repository holding the code, ("", False) outside a git checkout."""
    def run(*a):
        return subprocess.run(["git", "-C", str(paths.ROOT), *a], capture_output=True, text=True).stdout.strip()
    try:
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))
    except OSError:
        return "", False


def now() -> str:
    """Local time stamp, ISO format without time zone."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json(path: Path, obj, indent=1):
    """Atomic JSON write (a temporary file renamed over the target)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, default=float))
    tmp.replace(path)
