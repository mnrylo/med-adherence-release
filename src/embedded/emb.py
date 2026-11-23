"""Shared definitions of the embedded study: paths, the E2 finalist and the filter variants.

The finalist is the best E2 configuration after E3 (outputs/thesis/index/selection/E3.json,
per_experiment), unless conf/embedded.json fixes ``config_key``. Its causal counterpart has the
same factors with the forward-only 0.3 Hz high-pass (filter tag ``hp03c``) and is trained by
scripts/run_embedded.sh into outputs/embedded/runs/E2_causal/. The device apps and the replay
files run at 100 Hz, so the finalist must be a 100 Hz configuration.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from core import paths
from thesis import common as TC

CONF = paths.conf("embedded.json")
ROOT = paths.EMBEDDED
RESULTS = ROOT
PHASE0 = ROOT / "phase0"
PHASE1 = ROOT / "phase1"
PHASE2 = ROOT / "phase2"
EXPERIMENT = CONF["experiment"]


def _finalist_key() -> str:
    """config_key of conf/embedded.json, else the E3 choice for the experiment."""
    if CONF.get("config_key"):
        return CONF["config_key"]
    sel = TC.SELECTION / "E3.json"
    if not sel.exists():
        raise SystemExit(f"{sel} not found: run E3 (python -m thesis.postproc) or set config_key in conf/embedded.json")
    return json.loads(sel.read_text())["per_experiment"][EXPERIMENT]["config_key"]


CONFIG_KEY = _finalist_key()
CONFIG_DIR = TC.config_dir(EXPERIMENT, CONFIG_KEY)
FOLD_REPS = [f"f{f}_s{s}" for f in CONF["folds"] for s in CONF["seeds"]]
DEPLOY_REP = CONF["deploy_replicate"]

VARIANTS = CONF["filter_variants"]
"""Filter variants of the 0.3 Hz accelerometer high-pass (the only non-causal step at 100 Hz)."""

_f = TC.parse_key(CONFIG_KEY)
CAUSAL_KEY = TC.config_key({**_f, "filter": "hp0.3c"})
CAUSAL_DIR = ROOT / "runs" / "E2_causal" / CAUSAL_KEY


def model_dir() -> Path:
    """Latest registry version of the finalist (outputs/thesis/models/<name>/v<k>)."""
    name = "-".join([EXPERIMENT.lower(), _f["model"], f"fs{_f['fs_hz']}", f"w{int(round(_f['window_s'] * 100)):03d}",
                     f"ov{int(round(_f['overlap'] * 100)):02d}", TC.SIGNAL_TAG[_f["signals"]],
                     "hp03" if _f["filter"] == "hp0.3" else "raw"])
    vs = sorted((p for p in (TC.MODELS / name).glob("v*") if p.name[1:].isdigit()), key=lambda p: int(p.name[1:]))
    if not vs:
        raise SystemExit(f"no registry model {name}: run python -m thesis.registry")
    return vs[-1]


def config() -> dict:
    """Resolved training config of the finalist."""
    return json.loads((CONFIG_DIR / "config.json").read_text())


def postprocessing() -> dict:
    """Decoder and intake parameters of the registered finalist."""
    return json.loads((model_dir() / "postprocessing.json").read_text())


def pred_dir(variant: str, rep: str) -> Path:
    """Phase-0 predictions of a variant and replicate."""
    return PHASE0 / "predictions" / variant / rep


def now() -> str:
    """Local time stamp, ISO format without time zone."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


write_json = TC.write_json
