"""Annotated data loading and stream composition (core.streams, core.datasets)."""
import numpy as np

from core import datasets as T5
from core import sources as S
from core import streams as V


def test_load_gesture_trials(gesture_dir):
    trials = V.load_gesture_trials(gesture_dir, 50.0, 20.0)
    assert len(trials) == 3 * 5 * 3
    assert {t.subject for t in trials} == {"g_Subject01", "g_Subject02", "g_Subject03"}
    assert all(t.x.shape[1] == 6 and t.x.dtype == np.float32 for t in trials)
    g = next(t for t in trials if t.gesture == "G5")
    assert np.isclose(np.median(g.x[:, 2]), 1.0, atol=0.2)


def test_load_lab_sessions(lab_dir):
    lab = V.load_lab_sessions(lab_dir, 100.0, 40.0)
    assert len(lab) == 4
    s01 = [s for s in lab if s.protocol == "S01"]
    assert all(s.intake == 1 and s.gesture_sequence == ["G1", "G2", "G3", "G4"] for s in s01)
    assert all(s.meta["protocol_match"] for s in lab)
    assert {s.subject for s in lab} == {"lab_Subject01", "lab_Subject02"}


def test_compose_world_deterministic_and_aligned(gesture_dir):
    trials = V.load_gesture_trials(gesture_dir, 50.0, 20.0)
    world = {**T5.BASE_WORLD, "recipes": "all", "repeat_g4": 0.3}
    a = V.compose_world(trials, [], world, 50.0, 7, "w")
    b = V.compose_world(trials, [], world, 50.0, 7, "w")
    assert len(a) > 0 and len(a) == len(b)
    for s, t in zip(a, b):
        assert np.array_equal(s.x, t.x) and np.array_equal(s.y, t.y)
        assert len(s.x) == len(s.y)
        expected = int(any(V.RECIPES[r][1] for r in s.protocol.split("+")))
        assert s.intake == expected
        if s.intake:
            assert "G3" in s.gesture_sequence


def test_embed_in_background(lab_dir, gesture_dir):
    """Session sample j >= crossfade sits at meta["offset"] - crossfade + j in the embedded stream."""
    lab = V.load_lab_sessions(lab_dir, 50.0, 20.0)
    rng = np.random.default_rng(0)
    bg = [S.Stream(f"b{i}", "bg", f"p{i}", rng.normal(size=(5000, 6)).astype(np.float32), np.full(5000, S.REST),
                   np.full(5000, S.MASK)) for i in range(3)]
    wild = V.embed_in_background(lab, bg, 50.0, 42, (10, 20))
    cf = 10
    for w, s in zip(wild, lab):
        off, n = w.meta["offset"], w.meta["length"]
        assert n == len(s.x) and w.intake == s.intake
        assert np.array_equal(w.y[off:off + n - 2 * cf], s.y[cf:n - cf])
        assert w.gesture_sequence == s.gesture_sequence


def test_join_crossfade():
    a = (np.zeros((10, 6), np.float32), np.zeros(10, np.int64))
    b = (np.ones((10, 6), np.float32), np.ones(10, np.int64))
    x, y = V.join([a, b], 4)
    assert len(x) == 16 and len(y) == 16
    assert np.all(np.diff(x[:, 0]) >= 0)


def test_edge_dist():
    y = np.array([4] * 10 + [0] * 10)
    d = T5.edge_dist(y, 1, 10.0)
    assert np.isclose(d[10], 0.05) and np.isclose(d[0], 0.95)
    assert np.isinf(T5.edge_dist(np.full(5, 4), 1, 10.0)).all()


def test_gesture_folds_and_external_split():
    trials = [V.Trial(f"s{i}", "G1", np.zeros((3, 6))) for i in range(15)]
    folds = T5.gesture_folds(trials, {"split_seed": 42, "n_folds": 3})
    vals = [set(f["val"]) for f in folds]
    assert all(len(v) == 5 for v in vals) and set.union(*vals) == {f"s{i}" for i in range(15)}
    assert all(not set(f["val"]) & set(f["train"]) for f in folds)
    def mk(n, p):
        return [S.Stream(f"{n}{i}", n, f"{n}_p{i % p}", np.zeros((3, 6)), np.zeros(3), np.zeros(3)) for i in range(2 * p)]

    ext = T5.external_split({"ingesture": mk("i", 50), "wisdm": mk("w", 51)},
                            {"external_test_fraction": 0.2, "external_calibration_fraction": 0.25, "split_seed": 42})
    for name, parts in ext.items():
        subj = {k: {s.subject for s in v} for k, v in parts.items()}
        assert not subj["train"] & subj["test"] and not subj["train"] & subj["calibration"]
        assert not subj["calibration"] & subj["test"]
