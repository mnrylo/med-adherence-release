"""Integrity of the data in data/ (skipped when the data is not present)."""
import re

import numpy as np
import pandas as pd
import pytest

from core import paths
from core import streams as V
from core.sources import LABELS, SENSORS

gestures = pytest.mark.skipif(not paths.GESTURES.exists(), reason="data/labeled_output_gestures missing")
lab = pytest.mark.skipif(not paths.LAB_SESSIONS.exists(), reason="data/labeled_output missing")
third = pytest.mark.skipif(not (paths.INGESTURE.exists() and paths.WISDM.exists()), reason="run scripts/download_datasets.py")


@gestures
@pytest.mark.data
def test_gesture_trials_layout():
    subjects = sorted(p.name for p in paths.GESTURES.glob("Subject*") if p.is_dir())
    assert len(subjects) == 15
    files = sorted(paths.GESTURES.glob("Subject*/G*/*.csv"))
    assert files and all(f.parent.name in LABELS for f in files)
    head = pd.read_csv(files[0], nrows=5)
    assert set(SENSORS) <= set(head.columns)


@lab
@pytest.mark.data
def test_lab_sessions_layout():
    files = sorted(paths.LAB_SESSIONS.glob("Subject*/labeled_*.csv"))
    assert len({f.parent.name for f in files}) == 10
    assert all(V.LAB_RE.search(f.name) for f in files)
    df = pd.read_csv(files[0], nrows=200)
    assert set(SENSORS + ["Gesto"]) <= set(df.columns)
    assert set(df["Gesto"].dropna()) <= set(LABELS)


@lab
@pytest.mark.data
def test_lab_sessions_load():
    s = V.load_lab_sessions(paths.LAB_SESSIONS, 100.0, 40.0)
    assert len(s) == 347
    assert sum(x.intake for x in s) == 217
    acc = np.concatenate([x.x[:, :3] for x in s[:5]])
    assert 0.5 < np.median(np.linalg.norm(acc, axis=1)) < 1.5


@third
@pytest.mark.third_party
def test_third_party_layout():
    assert (paths.INGESTURE / "ingesture_metadata.xlsx").exists()
    assert len(list(paths.INGESTURE.glob("subject_*/*p10min*.csv"))) == 65
    assert len(list((paths.WISDM / "raw/watch/accel").glob("data_*_accel_watch.txt"))) == 51
    assert re.match(r"data_\d+_gyro_watch\.txt", next((paths.WISDM / "raw/watch/gyro").glob("*.txt")).name)
