"""Stage D3: online (fixed-lag) decoding on panel F (analysis A5).

    python -m decoders.d3_run [--workers 14]

Every decoder runs with the parameters it had in D1/D1b (the same selection on validation; nothing
is re-tuned for the lag), and the label of frame u is committed L seconds after u:
  * hmm, crf, crfdur, crf_fixs: fixed-lag Viterbi, the state reached by backtracking L frames from
    the best state at u + L (embedded.phase0_eval.fixed_lag_paths);
  * hsmm: the same over segments: from the best segment ending at u + L, follow the segment
    back-pointers until the segment that covers u. The relabel pass of the offline HSMM needs the
    whole path and is not applied online;
  * mode: the majority window keeps its width but its right edge is at most L frames after u
    (a trailing window at L = 0);
The last L frames of a stream take the offline path (flush at the stream end). L in LAGS_S;
"inf" is the offline decoder (for hsmm: without relabel, plus "inf_relabel" = the D1 decoder).
The record latency of a frame is L plus the classifier's look-ahead (half a window plus one
stride), stored per replicate.

Output per replicate: index/d3/<run>.csv.gz (session rows on test_wild per decoder and lag) and
.json (look-ahead, G3-rule false events/h on test_external per decoder and lag).
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from decoders import d1_report as D1R
from decoders import dcommon as D
from decoders import decoders as DE
from decoders.dcommon import M, P4, PP

from core.hmm import PostProcessor, _ergodic_weights

OUT = D.INDEX / "d3"
HZ, K, REST = DE.HZ, DE.K, DE.REST
LAGS_S = [0, 1, 2, 5, 10]


@functools.lru_cache(maxsize=None)
def selection() -> dict:
    """E3 grid selection (outputs/thesis/index/selection/E3_grid.json), read once."""
    return json.loads((D.C.SELECTION / "E3_grid.json").read_text())



def hmm_fixed_lag(hmm, logE: np.ndarray, lags: list[int]) -> dict:
    """Class paths for every lag in frames, plus "inf" (offline Viterbi)."""
    e = logE[:, hmm.state_class]
    T, S = e.shape
    back = np.zeros((T, S), np.int32)
    best = np.zeros(T, np.int64)
    delta = hmm.log_start + e[0]
    best[0] = delta.argmax()
    for t in range(1, T):
        cand = delta[:, None] + hmm.logA
        back[t] = cand.argmax(0)
        delta = cand.max(0) + e[t]
        best[t] = delta.argmax()
    full = np.empty(T, np.int64)
    full[-1] = np.where(hmm.final_mask, delta, -np.inf).argmax()
    for t in range(T - 1, 0, -1):
        full[t - 1] = back[t, full[t]]
    out = {"inf": hmm.state_class[full]}
    for L in lags:
        if L >= T:
            out[L] = out["inf"]
            continue
        cur = best[L:].copy()
        t_idx = np.arange(L, T)
        for j in range(L):
            cur = back[t_idx - j, cur]
        out[L] = np.concatenate([hmm.state_class[cur], out["inf"][T - L:]])
    return out


def hsmm_tables(logE: np.ndarray, dpar: dict):
    """Forward pass of core.hsmm.hsmm_decode, keeping the back-pointers."""
    T = len(logE)
    caps = np.array([max(2, min(T, int(np.ceil(2 * dpar[k][2])))) for k in range(K)])
    caps[REST] = max(2, min(T, int(20 * HZ)))
    Dm = int(caps.max())
    d = np.arange(1, Dm + 1)
    logdur = np.full((K, Dm), -np.inf)
    for k in range(K):
        mu, sd, _ = dpar[k]
        lp = -np.log(d[:caps[k]]) - (np.log(d[:caps[k]]) - mu) ** 2 / (2 * sd ** 2)
        logdur[k, :caps[k]] = lp - np.logaddexp.reduce(lp)
    w = _ergodic_weights({})
    A = np.zeros((K, K))
    for i, g in enumerate(P4.LABELS):
        for j, h in enumerate(P4.LABELS):
            A[i, j] = w[g].get(h, 0.0)
        A[i, i] = 1.0 if i == REST else 0.05
    logA = np.log(A / A.sum(1, keepdims=True))
    cs = np.vstack([np.zeros((1, K)), np.cumsum(logE, 0)])
    V = np.full((T + 1, K), -np.inf)
    inc = np.full((T + 1, K), -np.inf)
    arg_in = np.zeros((T + 1, K), np.int64)
    bestd = np.zeros((T + 1, K), np.int64)
    inc[0] = np.log(np.full(K, 1.0 / K))
    for t in range(1, T + 1):
        dd = d[:min(Dm, t)]
        s0 = t - dd
        score = inc[s0] + (cs[t][None, :] - cs[s0]) + logdur[:, :len(dd)].T
        j = score.argmax(0)
        V[t] = score[j, np.arange(K)]
        bestd[t] = dd[j]
        if t < T:
            cand = V[t][:, None] + logA
            arg_in[t] = cand.argmax(0)
            inc[t] = cand.max(0)
    return V, bestd, arg_in


def hsmm_fixed_lag(logE: np.ndarray, dpar: dict, lags: list[int]) -> dict:
    """Fixed-lag HSMM paths for every lag: from the best segment ending L frames later, follow the back-pointers to the
    segment covering each frame.
    """
    T = len(logE)
    V, bestd, arg_in = hsmm_tables(logE, dpar)

    def trace(ends: np.ndarray, us: np.ndarray) -> np.ndarray:
        t, k = ends.copy(), V[ends].argmax(1)
        lab = np.full(len(us), -1)
        todo = np.arange(len(us))
        while len(todo):
            s = t[todo] - bestd[t[todo], k[todo]]
            hit = s <= us[todo]
            lab[todo[hit]] = k[todo[hit]]
            todo, s = todo[~hit], s[~hit]
            k[todo] = arg_in[s, k[todo]]
            t[todo] = s
        return lab

    us = np.arange(T)
    out = {"inf": trace(np.full(T, T), us)}
    for L in lags:
        out[L] = trace(np.minimum(us + L + 1, T), us)
    return out


def mode_fixed_lag(tl: np.ndarray, win_s: float, L: int | str) -> np.ndarray:
    """Mode filter whose window ends at most L frames after the frame."""
    n = max(1, int(round(win_s * HZ)) | 1)
    h = n // 2
    c = uniform_filter1d(np.eye(K)[tl.argmax(1)], n, axis=0, mode="nearest").argmax(1)
    if L == "inf" or L >= h:
        return c
    idx = np.clip(np.arange(len(c)) + L - h, 0, len(c) - 1)
    return c[idx]


def mode_choice() -> dict:
    """Mode-filter width per configuration (as in D1)."""
    sess = pd.concat(pd.read_csv(f, usecols=["config_key", "decoder", "run_id", "set", "tp25", "fp25", "fn25", "exact"])
                     for f in sorted((D.INDEX / "d1").glob("*.csv.gz")))
    return {c: int(d[5:]) / 10 for c, d in D1R.choose_mode(sess).items()}


def job(args):
    """Every online decoder at every lag on one replicate."""
    rep, mode_w = args
    stem = str(rep.relative_to(D.C.RUNS)).replace("/", "__")
    f_csv, f_json = OUT / f"{stem}.csv.gz", OUT / f"{stem}.json"
    if f_json.exists():
        return str(rep)
    t0 = time.time()
    R = D.Rep(rep)
    cfg = str(rep.parent.relative_to(D.C.RUNS))
    hp, hs = selection()["configs"][cfg]["hmm"]["params"], selection()["configs"][cfg]["hsmm"]["params"]
    ex = json.loads((D.INDEX / "d1" / f"{stem}.json").read_text())["exports"]
    ex.update(json.loads((D.INDEX / "d1b" / f"{stem}.json").read_text())["exports"])
    lags = [int(round(s * HZ)) for s in LAGS_S]
    lagname = {**{l: str(s) for l, s in zip(lags, LAGS_S)}, "inf": "inf"}

    def markov(part, name):
        if name == "hmm":
            pp = PostProcessor(R.durations(hp["md"]), R.prior, R.post_cfg(hp["st"]), HZ)
            logs = R.logE(R.timelines(part, hp["ov"]), hp)
        else:
            cfg_, prior = DE.crf_post(R, hp, ex[name])
            dur = DE.Durations(np.asarray(ex[name]["durations"]["min_frames"]), np.asarray(ex[name]["durations"]["p_stay"]))
            pp = PostProcessor(dur, prior, cfg_, HZ)
            cb = np.asarray(ex[name].get("cbias", np.zeros(K)))
            logs = [(np.log(np.clip(t, cfg_["emission_floor"], 1)) - np.log(prior)) * ex[name]["scale"] + cb
                    for t in R.timelines(part, hp["ov"])]
        return [hmm_fixed_lag(pp.seg, e, lags) for e in logs]

    def hsmm(part):
        logs = R.logE(R.timelines(part, hs["ov"]), hs)
        per = [hsmm_fixed_lag(e, R.dpar, lags) for e in logs]
        if hs["md"] > 0:
            mf = R.durations(hs["md"]).min_frames
            for d, e in zip(per, logs):
                d["inf_relabel"] = P4.relabel(d["inf"], e, mf)
        return per

    def mode(part):
        tls = R.timelines(part, hp["ov"])
        return [{L: mode_fixed_lag(t, mode_w, L) for L in lags + ["inf"]} for t in tls]

    decoders = {"hmm": lambda p: markov(p, "hmm"), "crf": lambda p: markov(p, "crf"),
                "crfdur": lambda p: markov(p, "crfdur"), "crf_fixs": lambda p: markov(p, "crf_fixs"),
                "hsmm": hsmm, "mode": mode}
    rows, ext = [], {}
    test_h = R.external_hours("test_external")
    for dec, fn in decoders.items():
        per = fn(D.TEST)
        for L in per[0]:
            lab = lagname.get(L, L)
            for r in D.session_rows(R, D.TEST, [d[L] for d in per], dec):
                rows.append({**r, "lag_s": lab})
        pe = fn("test_external")
        ext[dec] = {lagname.get(L, L): sum(M.n_gesture_events(d[L], PP.G3) for d in pe) / test_h for L in pe[0]}
    fs, win, stride = R.man["fs"], R.man["window"], R.man["stride"]
    meta = {"run_id": str(rep.relative_to(D.C.RUNS)), "experiment": rep.parts[-3][:2], "config_key": rep.parent.name,
            "role": PP.role(rep), "replicate": rep.name}
    pd.DataFrame([{**meta, **r} for r in rows]).to_csv(f_csv, index=False)
    D.write_json(f_json, {**meta, "lookahead_s": (win / 2 + stride) / fs, "mode_w": mode_w,
                          "g3_rule_per_h": ext, "seconds": time.time() - t0})
    return str(rep)


def main():
    """Run stage D3."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    reps = [r for r in PP.rep_dirs() if not r.name.startswith("fall")][: a.limit or None]
    mw = mode_choice()
    jobs = [(r, mw[r.parent.name]) for r in reps]
    print(f"[{D.now()}] D3: {len(reps)} replicates, lags {LAGS_S} s", flush=True)
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{D.now()}] {i}/{len(reps)} {r}", flush=True)
    read = sorted({l.rsplit("/", 1)[-1] for l in D.READ_LOG.read_text().splitlines() if l})
    print(f"sets read: {read}", flush=True)


if __name__ == "__main__":
    main()
