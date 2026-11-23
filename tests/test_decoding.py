"""HMM / HSMM decoding, intake thresholds and segment metrics (core.hmm, core.hsmm)."""
import numpy as np

from core import hsmm as H
from core.hmm import Durations, PostProcessor, calibrate_threshold, count_events, viterbi_batch
from core.paths import conf

POST = conf("training.json")["post"]


def labels():
    return np.array([4] * 60 + [0] * 20 + [4] * 15 + [1] * 15 + [4] * 15 + [2] * 20 + [4] * 15 + [3] * 25 + [4] * 60)


def posteriors(y, p=0.9):
    post = np.full((len(y), 5), (1 - p) / 4)
    post[np.arange(len(y)), y] = p
    return post


def test_durations_fit():
    d = Durations.fit([labels()], 1, {"min_duration_fraction": 0.5}, 10)
    assert d.min_frames[0] == 10 and d.min_frames[2] == 10 and (d.p_stay < 1).all()


def test_hmm_viterbi_recovers_clean_sequence():
    y = labels()
    pp = PostProcessor(Durations.fit([y], 1, {"min_duration_fraction": 0.5}, 10), np.full(5, 0.2), POST, 10)
    path = viterbi_batch(pp.seg, [pp.emissions(posteriors(y))])[0]
    assert (path == y).mean() > 0.97


def test_intake_llr_sign():
    y = labels()
    pp = PostProcessor(Durations.fit([y], 2, {"min_duration_fraction": 0.5}, 5), np.full(5, 0.2), POST, 5)
    rest = np.full(len(y[::2]), 4)
    sc = pp.llrs_batch([posteriors(y[::2]), posteriors(rest)])["llr_phased"]
    assert sc[0] > 0 > sc[1]


def test_count_events_and_threshold():
    assert count_events(np.array([0, 2, 3, 0, 5, 0, 2]), 1.0) == 3
    scores = np.array([3.0, 2.5, 2.0, -1.0, -2.0, 0.5])
    labels_ = np.array([1, 1, 1, 0, 0, 0])
    neg = [np.array([1.0, 1.5, 0.2])]
    tau = calibrate_threshold(scores, labels_, neg, 1.0, 1.0)
    assert sum(count_events(s, tau) for s in neg) <= 1
    assert ((scores > tau) == labels_.astype(bool)).all()
    strict = calibrate_threshold(scores, labels_, neg, 1.0, 0.0)
    assert strict >= 1.5


def test_hsmm_and_relabel():
    y = labels()
    logE = np.log(posteriors(y, 0.8))
    dpar = H.fit_log_normal([y])
    path = H.hsmm_decode(logE, dpar, 10)
    assert (path == y).mean() > 0.97
    noisy = y.copy()
    noisy[100:102] = 0
    fixed = H.relabel(noisy, np.log(posteriors(y, 0.8)), np.full(5, 5))
    assert (fixed == y).mean() > 0.99


def test_segment_metrics():
    t = H.label_segments(np.array([4, 0, 0, 0, 4, 2, 2, 4]))
    assert t == [(0, 1, 4), (2, 5, 7)]
    assert H.segment_counts(t, t, 0.5) == (2, 0, 0)
    p = [(0, 1, 2), (2, 5, 7), (3, 7, 8)]
    tp, fp, fn = H.segment_counts(p, t, 0.5)
    assert (tp, fp, fn) == (1, 2, 1)
    assert np.isclose(H.seg_f1(tp, fp, fn), 0.4)
    assert H.seg_f1(0, 0, 0) == 1.0
    assert H.levenshtein(["G1", "G2", "G3"], ["G1", "G3"]) == 1
