"""Stage D1b (post-hoc, declared after D1): two CRF variants that test the D1 collapse on E1.

    python -m decoders.d1b_run [--workers 8]

In D1 the learned emission scale of crf / crfdur fell to 0.04-0.13 on E1 (the decoded path stayed
in rest). The class prior is learnable but sits inside the scale, so it shrinks with it. Variants
(no duration chains, same training protocol as crf):
  * crf_bias: a per-class additive emission bias outside the scale (L2 to 0);
  * crf_fixs: the scale frozen at the selected HMM's 1/temperature.
Outputs index/d1b/<run_id>.csv.gz and .json, in the D1 format (decoders crf_bias, crf_fixs).
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from decoders import d1_run as D1
from decoders import dcommon as D
from decoders import decoders as DE
from decoders.dcommon import PP

OUT = D.INDEX / "d1b"
VARIANTS = {"crf_bias": "bias", "crf_fixs": "fixs"}


def job(rep):
    """Train and decode the crf_bias and crf_fixs variants of one replicate."""
    stem = str(rep.relative_to(D.C.RUNS)).replace("/", "__")
    f_csv, f_json = OUT / f"{stem}.csv.gz", OUT / f"{stem}.json"
    if f_json.exists():
        return str(rep)
    t0 = time.time()
    R = D.Rep(rep)
    hp = D1.selection()["configs"][str(rep.parent.relative_to(D.C.RUNS))]["hmm"]["params"]
    seed = 1000 * int(R.info["fold"]) + int(R.info["seed"])
    info = {"run_id": str(rep.relative_to(D.C.RUNS)), "hmm_params": hp, "training": {}, "exports": {},
            "thresholds": {}, "test_external": {}}
    yv = np.array([R.meta["val"][s]["intake"] for s in R.sids("val")])
    cal_h, test_h = R.external_hours("external_calibration"), R.external_hours("test_external")
    rows = []
    for name, var in VARIANTS.items():
        m, tr = DE.train_crf(R, hp, False, seed, var)
        ex = DE.crf_export(m)
        info["training"][name], info["exports"][name] = tr, ex
        sc = {p: DE.crf_llr_scores(R, p, hp, ex) for p in ["val", D.TEST, "external_calibration", "test_external"]}
        smax = {p: np.array([s.max() if len(s) else -np.inf for s in v]) for p, v in sc.items()}
        taus = {fa: PP.calibrate_threshold(smax["val"], yv, sc["external_calibration"], cal_h, fa) for fa in PP.FA_BUDGETS}
        info["thresholds"][name] = {f"fa{PP._fa(fa)}": t for fa, t in taus.items()}
        for part in ("val", D.TEST):
            r = D.session_rows(R, part, DE.crf_decode(R, part, hp, ex), name)
            for fa in PP.FA_BUDGETS:
                for x, s in zip(r, smax[part]):
                    x["score"], x[f"pred_score_fa{PP._fa(fa)}"] = float(s), int(s > taus[fa])
            rows += r
        e = DE.crf_decode(R, "test_external", hp, ex)
        info["test_external"][name] = {
            "hours": test_h, "g3_rule_per_h": sum(D.M.n_gesture_events(q, PP.G3) for q in e) / test_h,
            "gesture_events_per_h": sum(D.M.n_gesture_events(q) for q in e) / test_h,
            **{f"score_fa{PP._fa(fa)}_per_h": sum(PP.count_events(s, taus[fa]) for s in sc["test_external"]) / test_h
               for fa in PP.FA_BUDGETS}}
    info["seconds"] = time.time() - t0
    meta = {"run_id": info["run_id"], "experiment": rep.parts[-3][:2], "config_key": rep.parent.name,
            "role": PP.role(rep), "replicate": rep.name}
    pd.DataFrame([{**meta, **r} for r in rows]).to_csv(f_csv, index=False)
    D.write_json(f_json, info)
    return str(rep)


def main():
    """Run stage D1b."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    reps = [r for r in PP.rep_dirs() if not r.name.startswith("fall")]
    if a.limit:
        reps = [next(r for r in reps if r.parts[-3].startswith(e)) for e in ("E1", "E2")][: a.limit]
    print(f"[{D.now()}] D1b: {len(reps)} replicates", flush=True)
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(job, reps), 1):
            print(f"[{D.now()}] {i}/{len(reps)} {r}", flush=True)
    read = sorted({l.rsplit("/", 1)[-1] for l in D.READ_LOG.read_text().splitlines() if l})
    print(f"sets read: {read}", flush=True)


if __name__ == "__main__":
    main()
