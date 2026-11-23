"""Stage D1: every frozen-classifier decoder on panel F, with decoder-specific intake decisions.

    python -m decoders.d1_run [--workers 8]

Per replicate (resumable: a replicate with its output file is skipped):
  * decoders: none, hmm, hsmm (E3 parameters), mode_<w> (every MODE_S width; the width is
    chosen later on validation), crf, crfdur, refiner (trained on the replicate's validation streams);
  * session rows on val and test_wild (dcommon.session_rows);
  * intake: the G3 rule for every decoder; a score for hmm (E3 LLR), crf and crfdur (the same
    phased LLR with the learned potentials), thresholded per replicate at <= 0.5 / 1 false events/h
    on external_calibration (best val balanced accuracy, as E3);
  * false events/h on test_external: G3 segments per hour of the decoded path (rule) and score
    events per hour (score).

Output: index/d1/<run_id>.csv.gz (sessions), index/d1/<run_id>.json (training info, exports,
thresholds, external rates).
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from decoders import dcommon as D
from decoders import decoders as DE
from decoders.dcommon import M, PP

OUT = D.INDEX / "d1"


@functools.lru_cache(maxsize=None)
def selection() -> dict:
    """E3 grid selection (outputs/thesis/index/selection/E3_grid.json), read once."""
    return json.loads((D.C.SELECTION / "E3_grid.json").read_text())



def out_paths(rep):
    """Output files of a replicate."""
    stem = str(rep.relative_to(D.C.RUNS)).replace("/", "__")
    return OUT / f"{stem}.csv.gz", OUT / f"{stem}.json"


def job(rep):
    """Every D1 decoder and intake decision on one replicate."""
    f_csv, f_json = out_paths(rep)
    if f_json.exists():
        return str(rep)
    t0 = time.time()
    R = D.Rep(rep)
    chosen = selection()["configs"][str(rep.parent.relative_to(D.C.RUNS))]
    hp, hs = chosen["hmm"]["params"], chosen["hsmm"]["params"]
    seed = 1000 * int(R.info["fold"]) + int(R.info["seed"])
    info = {"run_id": str(rep.relative_to(D.C.RUNS)), "hmm_params": hp, "hsmm_params": hs}

    crf, crfdur, ref = DE.train_crf(R, hp, False, seed), DE.train_crf(R, hp, True, seed), DE.train_refiner(R, hp["ov"], seed)
    ex = {"crf": DE.crf_export(crf[0]), "crfdur": DE.crf_export(crfdur[0])}
    info["training"] = {"crf": crf[1], "crfdur": crfdur[1], "refiner": ref[1]}
    info["exports"] = ex
    decode = {
        "none": lambda part: R.decode(part, {"decoder": "none", "ov": R.ov}),
        "hmm": lambda part: R.decode(part, hp),
        "hsmm": lambda part: R.decode(part, hs),
        "crf": lambda part: DE.crf_decode(R, part, hp, ex["crf"]),
        "crfdur": lambda part: DE.crf_decode(R, part, hp, ex["crfdur"]),
        "refiner": lambda part: DE.refiner_decode(ref[0], R, part, hp["ov"]),
    }
    for w in DE.MODE_S:
        decode[f"mode_{int(w * 10):02d}"] = lambda part, w=w: [DE.mode_filter(t, w) for t in R.timelines(part, hp["ov"])]

    scorers = {"hmm": lambda part: R.llr_scores(part, hp),
               "crf": lambda part: DE.crf_llr_scores(R, part, hp, ex["crf"]),
               "crfdur": lambda part: DE.crf_llr_scores(R, part, hp, ex["crfdur"])}
    yv = np.array([R.meta["val"][s]["intake"] for s in R.sids("val")])
    cal_h, test_h = R.external_hours("external_calibration"), R.external_hours("test_external")
    smax, ext_events, taus = {}, {}, {}
    for name, fn in scorers.items():
        sc = {part: fn(part) for part in ["val", D.TEST, "external_calibration", "test_external"]}
        smax[name] = {part: np.array([s.max() if len(s) else -np.inf for s in v]) for part, v in sc.items()}
        taus[name] = {fa: PP.calibrate_threshold(smax[name]["val"], yv, sc["external_calibration"], cal_h, fa)
                      for fa in PP.FA_BUDGETS}
        ext_events[name] = {f"score_fa{PP._fa(fa)}_per_h": sum(PP.count_events(s, taus[name][fa]) for s in sc["test_external"]) / test_h
                            for fa in PP.FA_BUDGETS}
    info["thresholds"] = {n: {f"fa{PP._fa(fa)}": t for fa, t in v.items()} for n, v in taus.items()}

    rows, ext = [], {}
    for dec, fn in decode.items():
        for part in ("val", D.TEST):
            r = D.session_rows(R, part, fn(part), dec)
            if dec in scorers:
                for fa in PP.FA_BUDGETS:
                    pred = (smax[dec][part] > taus[dec][fa]).astype(int)
                    for x, s, q in zip(r, smax[dec][part], pred):
                        x["score"], x[f"pred_score_fa{PP._fa(fa)}"] = float(s), int(q)
            rows += r
        e = fn("test_external")
        ext[dec] = {"hours": test_h, "g3_rule_per_h": sum(M.n_gesture_events(q, PP.G3) for q in e) / test_h,
                    "gesture_events_per_h": sum(M.n_gesture_events(q) for q in e) / test_h, **ext_events.get(dec, {})}
    info["test_external"] = ext
    info["seconds"] = time.time() - t0
    meta = {"run_id": info["run_id"], "experiment": rep.parts[-3][:2], "config_key": rep.parent.name,
            "role": PP.role(rep), "replicate": rep.name}
    pd.DataFrame([{**meta, **r} for r in rows]).to_csv(f_csv, index=False)
    D.write_json(f_json, info)
    return str(rep)


def main():
    """Run stage D1."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    reps = [r for r in PP.rep_dirs() if not r.name.startswith("fall")][: a.limit or None]
    print(f"[{D.now()}] D1: {len(reps)} replicates", flush=True)
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(job, reps), 1):
            print(f"[{D.now()}] {i}/{len(reps)} {r}", flush=True)
    read = sorted({l.rsplit("/", 1)[-1] for l in D.READ_LOG.read_text().splitlines() if l})
    print(f"sets read: {read}", flush=True)


if __name__ == "__main__":
    main()
