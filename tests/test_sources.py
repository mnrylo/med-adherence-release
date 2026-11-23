"""Signal handling of core.sources."""
import numpy as np
import pytest

from core import sources as S


def stream(x, y=None):
    y = np.full(len(x), S.REST) if y is None else y
    return S.Stream("s", "t", "p", x.astype(np.float32), y, np.full(len(x), S.MASK))


def test_label_runs_to_sequence():
    y = np.array([4, 0, 0, 4, -1, 1, 1, 2, 4, 2, 3])
    assert S.label_runs_to_sequence(y) == ["G1", "G2", "G3", "G4"]
    assert S.label_runs_to_sequence(np.array([0, -1, 0])) == ["G1"]


def test_to_units():
    x = S.to_units(np.full((3, 3), S.G), np.full((3, 3), 10.0))
    assert np.allclose(x[:, :3], 1.0) and np.allclose(x[:, 3:], 10.0)


def test_resample_lengths_and_labels():
    x = np.random.default_rng(0).normal(size=(1000, 6))
    y = np.repeat(np.arange(5), 200)
    xo, (yo,) = S.resample(x, [y], 100.0, 20.0, 8.0)
    assert xo.shape == (200, 6) and xo.dtype == np.float32
    assert np.array_equal(np.bincount(yo), [40] * 5)
    same, _ = S.resample(x, [], 100.0, 100.0, 40.0)
    assert np.allclose(same, x, atol=1e-6)


def test_highpass_acc_removes_gravity_only():
    n = 3000
    x = np.zeros((n, 6))
    x[:, 2] = 1.0
    x[:, 3] = 5.0
    for causal in (False, True):
        out = S.apply_post_filter([stream(x)], {"type": "highpass_acc", "cutoff": 0.3, "causal": causal}, 100.0)[0].x
        assert np.abs(out[:, :3]).max() < 1e-4
        assert np.allclose(out[:, 3:], x[:, 3:])


def test_causal_filter_uses_past_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 6))
    spec = {"type": "highpass_acc", "cutoff": 0.3, "causal": True}
    a = S.apply_post_filter([stream(x)], spec, 100.0)[0].x
    b = S.apply_post_filter([stream(x[:1000])], spec, 100.0)[0].x
    assert np.allclose(a[:1000], b, atol=1e-6)
    z = S.apply_post_filter([stream(x)], {"type": "highpass_acc", "cutoff": 0.3}, 100.0)[0].x
    assert not np.allclose(z[:1000], S.apply_post_filter([stream(x[:1000])], {"type": "highpass_acc", "cutoff": 0.3},
                                                         100.0)[0].x, atol=1e-3)


def test_filter_errors():
    x = np.zeros((100, 6))
    with pytest.raises(ValueError):
        S.apply_post_filter([stream(x)], {"type": "lowpass", "cutoff": 60}, 100.0)
    with pytest.raises(KeyError):
        S.apply_post_filter([stream(x)], {"type": "notch", "cutoff": 1}, 100.0)


def test_window_and_stride():
    assert S.window_and_stride({"window_s": 2.0, "fs": 100.0, "overlap": 0.75}) == (200, 50)
    assert S.window_and_stride({"window_s": 0.5, "fs": 20.0, "overlap": 0.0}) == (10, 10)
    assert S.window_and_stride({"window_s": 1.0, "fs": 50.0, "overlap": 0.5, "frame_stride": 5}) == (50, 5)


def test_split_external_is_deterministic_and_disjoint():
    streams = [S.Stream(f"s{i}", "x", f"p{i % 20}", np.zeros((10, 6)), np.zeros(10), np.zeros(10)) for i in range(60)]
    tr, te = S.split_external(streams, 0.2, 42)
    assert len(te) == 4 and not tr & te and len(tr | te) == 20
    assert S.split_external(streams, 0.2, 42) == (tr, te)


def test_stretch_and_rotation():
    x = np.random.default_rng(0).normal(size=(100, 6))
    assert len(S.stretch(x, 1.25)) == 125
    R = S.rotation(np.random.default_rng(3), 25)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
