"""Test configuration: src/ on the import path, outputs redirected to a temporary directory.

The embedded study resolves its finalist from the E3 selection, so a minimal selection file is
written before any test module imports it. Synthetic data helpers are shared as fixtures.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(tempfile.mkdtemp(prefix="medadh_tests_"))
os.environ["MEDADH_OUTPUTS"] = str(OUT)
FINALIST = "cnn_bilstm__fs100_w200_ov75_accgyr_hp03__hp05"
sel = OUT / "thesis" / "index" / "selection"
sel.mkdir(parents=True, exist_ok=True)
(sel / "E3.json").write_text(json.dumps({"per_experiment": {"E2": {"config_key": FINALIST}}}))

SENSORS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def imu(n: int, rng, gesture: bool = True) -> np.ndarray:
    """(n, 6) synthetic IMU at 100 Hz: gravity on z in m/s^2, a smooth movement, gyro in deg/s."""
    t = np.arange(n) / 100.0
    amp = 3.0 if gesture else 0.2
    acc = np.stack([amp * np.sin(2 * np.pi * 1.1 * t), amp * np.cos(2 * np.pi * 0.7 * t),
                    9.80665 + 0.5 * amp * np.sin(2 * np.pi * 0.4 * t)], 1)
    gyr = 40 * amp * np.stack([np.sin(2 * np.pi * 0.9 * t), np.cos(2 * np.pi * 1.3 * t), np.sin(2 * np.pi * 0.5 * t)], 1)
    return acc + rng.normal(0, 0.05, acc.shape), gyr + rng.normal(0, 1, gyr.shape)


@pytest.fixture
def gesture_dir(tmp_path):
    """data/labeled_output_gestures layout with 3 subjects x G1..G5 x 3 trials."""
    rng = np.random.default_rng(0)
    root = tmp_path / "labeled_output_gestures"
    for s in range(1, 4):
        for g in ["G1", "G2", "G3", "G4", "G5"]:
            d = root / f"Subject{s:02d}" / g
            d.mkdir(parents=True)
            for k in range(3):
                n = int(rng.integers(150, 300)) if g != "G5" else 600
                acc, gyr = imu(n, rng, g != "G5")
                pd.DataFrame(np.concatenate([acc, gyr], 1), columns=SENSORS).to_csv(
                    d / f"gesture_subject{s:02d}_{g}_2025071{k}-120000.csv", index=False)
    return root


@pytest.fixture
def lab_dir(tmp_path):
    """data/labeled_output layout: two subjects, one S01 session (intake) and one S03 session each."""
    rng = np.random.default_rng(1)
    root = tmp_path / "labeled_output"
    seqs = {"S01": (["G5", "G1", "G5", "G2", "G5", "G3", "G5", "G4", "G5"], 1), "S03": (["G5", "G1", "G5", "G4", "G5"], 0)}
    for s in (1, 2):
        d = root / f"Subject{s:02d}"
        d.mkdir(parents=True)
        for k, (proto, (seq, intake)) in enumerate(seqs.items()):
            labels = np.concatenate([[g] * 150 for g in seq])
            acc, gyr = imu(len(labels), rng)
            df = pd.DataFrame(np.concatenate([acc, gyr], 1), columns=SENSORS)
            df["Gesto"] = labels
            df.to_csv(d / f"labeled_subject{s:02d}_20260121-16000{k}_{intake}_{proto}.csv", index=False)
    return root
