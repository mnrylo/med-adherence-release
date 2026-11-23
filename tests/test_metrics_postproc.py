"""Timeline construction, metric blocks and decoder keys (thesis.metrics, thesis.postproc)."""
import numpy as np

from thesis import metrics as M
from thesis import postproc as PP


def test_to_timeline_constant_and_centres():
    p = np.tile([0.1, 0.2, 0.3, 0.2, 0.2], (11, 1))
    tl = M.to_timeline(p, 50, 100.0, 200, 50, 10)
    assert tl.shape == (50, 5) and np.allclose(tl, p[0])
    q = np.eye(5)[[0, 0, 1, 1, 1]]
    tl = M.to_timeline(q, 30, 10.0, 10, 10, 10)
    assert tl[0].argmax() == 0 and tl[25].argmax() == 1


def test_timeline_rule_merges_short_segments():
    tl = np.eye(5)[[4] * 10 + [0] * 2 + [4] * 10 + [2] * 8 + [4] * 5].astype(float)
    tl[10:12, 4] = 0.4
    lab = M.timeline_rule(tl, np.array([5, 5, 5, 5, 0]))
    assert 0 not in lab and (lab == 2).sum() == 8


def test_classification_and_binary():
    y = np.array([0, 1, 2, 3, 4, -1])
    r = M.classification(y, np.array([0, 1, 2, 3, 4, 0]))
    assert r["macro_f1"] == 1.0 and r["n"] == 5
    b = M.binary([1, 1, 0, 0], [1, 0, 0, 1])
    assert b["sensitivity"] == 0.5 and b["specificity"] == 0.5 and b["balanced_accuracy"] == 0.5


def test_flatten():
    f = M.flatten({"val": {"window": {"macro_f1": 0.5, "confusion": [[1]]}}, "timeline_rule": {"min_s": [1]}})
    assert f == {"val.window.macro_f1": 0.5}


def test_decoder_keys_and_grid():
    assert PP.decoder_key({"decoder": "none"}) == "none"
    assert PP.decoder_key({"decoder": "hmm", "ov": 75, "st": 0.95, "md": 0.3, "t": 2.0}) == "hmm__ov75_st095_md030_t200"
    assert PP.decoder_key({"decoder": "hsmm", "ov": 50, "md": 0.0, "t": 4.0}) == "hsmm__ov50_md000_t400"
    n_t = len(PP.TEMPERATURES)
    assert len(PP.decoder_points(75)) == 1 + 2 * (9 * n_t + 4 * n_t)
    assert len(PP.decoder_points(25)) == 1 + (9 * n_t + 4 * n_t)
