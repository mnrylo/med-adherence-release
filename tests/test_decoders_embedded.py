"""Decoder-study statistics and filters, and the device reference pipeline of the embedded study."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter, sosfilt_zi

from core import sources as S
from decoders import dcommon as D
from decoders import decoders as DE
from decoders import stats as ST
from embedded import devicepipe as DP
from embedded import emb as E


def test_allowlist():
    assert D.TEST == "test_wild"
    assert "test_sequences" not in D.ALLOWED and "val" in D.ALLOWED


def test_mode_filter_removes_blips():
    y = np.array([4] * 20 + [0] + [4] * 20 + [1] * 15)
    tl = np.eye(5)[y]
    out = DE.mode_filter(tl, 0.5)
    assert 0 not in out and (out == 1).sum() >= 12


def session_frame(decoders, n_subj=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in decoders:
        for rep in ("r0", "r1"):
            for s in range(n_subj):
                for k in range(4):
                    good = d == "better"
                    rows.append({"decoder": d, "run_id": rep, "sid": f"s{s}_{k}", "subject": f"p{s}", "intake": k % 2,
                                 "tp25": 3 + good, "fp25": 1 - good, "fn25": 1 - good,
                                 "exact": int(good or rng.random() < 0.3), "edit": 0.0 if good else 0.5,
                                 "pred_g3": int(k % 2 == 1 or not good)})
    return pd.DataFrame(rows)


def test_compare_paired():
    df = session_frame(["base", "better"])
    r = ST.compare(df, ["base", "better"], "base", ["seg_f1_25", "order_exact", "g3_spec"], n_boot=200)
    b = r[(r.decoder == "better") & (r.metric == "seg_f1_25")].iloc[0]
    assert b["mean"] == 1.0 and b["diff"] > 0 and b["diff_lo"] > 0
    assert r[(r.decoder == "better") & (r.metric == "g3_spec")].iloc[0]["mean"] == 1.0


def test_emb_finalist_and_causal_key():
    assert E.CONFIG_KEY.endswith("__hp05") and "hp03c" in E.CAUSAL_KEY
    assert set(E.VARIANTS) == {"zerophase", "causal"} and len(E.FOLD_REPS) == 9


def bundle_stub(win=20, stride=5, fs=100.0):
    sos = butter(4, 0.3, btype="high", fs=fs, output="sos")
    return DP.Bundle(fs, win, stride, 10, 2, sos, sosfilt_zi(sos), np.full(5, 0.2), 1e-4, 1.0, 5.0, None, None, None,
                     80, 0, {}, {})


def test_device_filter_matches_training_filter():
    x = np.random.default_rng(0).normal(size=(1500, 6))
    b = bundle_stub()
    dev = DP.filter_stream(b, x)
    st = S.Stream("a", "t", "p", x.astype(np.float32), np.zeros(1500, int), np.zeros(1500, int))
    ref = S.apply_post_filter([st], {"type": "highpass_acc", "cutoff": 0.3, "causal": True}, 100.0)[0].x
    assert np.allclose(dev, ref, atol=1e-5)


def test_device_windows_and_timeline_match_offline():
    from thesis import metrics as M
    b = bundle_stub()
    xf = np.random.default_rng(1).normal(size=(203, 6))
    w = DP.window_batch(b, xf)
    assert w.shape == (41, 6, 20)
    assert np.allclose(w[0, :, 10], xf[0]) and np.allclose(w[0, :, 0], xf[0])
    probs = np.random.default_rng(2).dirichlet(np.ones(5), 41)
    assert np.allclose(DP.timeline(b, probs, 21), M.to_timeline(probs, 21, 100.0, 20, 5, 10), atol=1e-6)


def test_fixed_lag_equals_viterbi_for_long_lag():
    from core.hmm import Durations, PostProcessor
    y = np.array([4] * 40 + [0] * 15 + [4] * 20 + [2] * 15 + [4] * 30)
    post = np.full((len(y), 5), 0.05)
    post[np.arange(len(y)), y] = 0.8
    pp = PostProcessor(Durations.fit([y], 1, {"min_duration_fraction": 0.5}, 10), np.full(5, 0.2),
                       E.TC.TRAINING["post"], 10)
    hmm = DP.HMM.from_json(DP.HMM.to_json(pp.seg))
    e = pp.emissions(post)
    full = DP.fixed_lag(hmm, e, len(y))
    assert (full == y).mean() > 0.97
    assert (DP.fixed_lag(hmm, e, 30) == full).mean() > 0.97


@pytest.mark.parametrize("scores,expected", [([0, 2, 3, 0, 2], [1.0, 2.5]), ([-1, -1], [])])
def test_alarms_rising_edges(scores, expected):
    ends = np.arange(1, len(scores) + 1)
    assert DP.alarms(np.array(scores, float), ends, 1.0, 2).tolist() == expected
