"""Locations of the data, configuration and output directories.

Every path is read from ``conf/paths.json`` and resolved against the repository root, so the
code never depends on the directory it is started from. The environment variable
``MEDADH_OUTPUTS`` overrides the output root (used by the tests to write into a temporary
directory).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf"
SRC = ROOT / "src"

_P = json.loads((CONF / "paths.json").read_text())


def _abs(rel: str) -> Path:
    """Absolute path of a configured path (relative paths are resolved against the repository root)."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


DATA = _abs(_P["data"])
GESTURES = _abs(_P["gestures"])
LAB_SESSIONS = _abs(_P["lab_sessions"])
INGESTURE = _abs(_P["ingesture"])
WISDM = _abs(_P["wisdm"])
OUTPUTS = Path(os.environ["MEDADH_OUTPUTS"]) if os.environ.get("MEDADH_OUTPUTS") else _abs(_P["outputs"])
CACHE = OUTPUTS / "cache"
THESIS = OUTPUTS / "thesis"
DECODERS = OUTPUTS / "decoders"
EMBEDDED = OUTPUTS / "embedded"
ANDROID = ROOT / "android"


def conf(name: str) -> dict:
    """A configuration file of ``conf/`` as a dict."""
    return json.loads((CONF / name).read_text())
