"""Stage D0: reproduce the E3 decoders on test_wild through the allowlisted loader.

    python -m decoders.d0_reproduce [--workers 6]

For every replicate of panel F (the E3 replicates: E1/E2 finalists and short-window refs), the
decoders none / hmm / hsmm are run with the parameters E3 selected on validation
(outputs/thesis/index/selection/E3_grid.json), together with the HMM LLR intake decision and its
thresholds (best val balanced accuracy at <= 0.5 / 1 false events/h on external_calibration).

Check V2: the test_wild metrics must equal the E3 index/postproc.csv to 1e-9.
Output:
  index/d0_sessions.csv      one row per (replicate, decoder, test_wild stream), with segment counts
  index/d0_check.csv         reproduced vs E3 value per (replicate, decoder, metric)
  index/d0_summary.json      the V2 verdict
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from decoders import dcommon as D
from decoders.dcommon import M, PP

CHECKS = ["timeline.seg_f1_10", "timeline.seg_f1_25", "timeline.seg_f1_50", "timeline.macro_f1",
          "order.exact_match", "order.edit_distance", "intake_g3.balanced_accuracy",
          "intake_llr_fa10.sensitivity", "intake_llr_fa10.specificity", "intake_llr_fa05.sensitivity"]


def job(rep):
    """E3 decoders and the LLR decision on test_wild of one replicate: session rows and reproduced metrics."""
    R = D.Rep(rep)
    chosen = SEL["configs"][str(rep.parent.relative_to(D.C.RUNS))]
    hp = chosen["hmm"]["params"]
    sc = {part: R.llr_scores(part, hp) for part in ["val", D.TEST, "external_calibration"]}
    smax = {part: np.array([s.max() if len(s) else -np.inf for s in v]) for part, v in sc.items()}
    yv = np.array([R.meta["val"][s]["intake"] for s in R.sids("val")])
    cal_h = R.external_hours("external_calibration")
    taus = {fa: PP.calibrate_threshold(smax["val"], yv, sc["external_calibration"], cal_h, fa) for fa in PP.FA_BUDGETS}
    sessions, checks = [], []
    for dec, ch in chosen.items():
        paths = R.decode(D.TEST, ch["params"])
        res, srows = PP.timeline_metrics(R, D.TEST, paths)
        y = np.array([r["intake"] for r in srows])
        rows = D.session_rows(R, D.TEST, paths, dec)
        for fa in PP.FA_BUDGETS:
            pred = (smax[D.TEST] > taus[fa]).astype(int)
            res[f"intake_llr_fa{PP._fa(fa)}"] = M.binary(y, pred)
            for r, s, q in zip(rows, smax[D.TEST], pred):
                r["llr"], r[f"pred_llr_fa{PP._fa(fa)}"] = float(s), int(q)
        fl = M.flatten({D.TEST: res})
        meta = {"run_id": str(rep.relative_to(D.C.RUNS)), "experiment": rep.parts[-3][:2],
                "config_key": rep.parent.name, "role": PP.role(rep), "replicate": rep.name, "decoder": dec,
                "decoder_key": ch["decoder_key"]}
        sessions += [{**meta, **r} for r in rows]
        checks += [{**meta, "metric": k, "reproduced": fl[f"{D.TEST}.{k}"]} for k in CHECKS]
    return sessions, checks


def main():
    """Run stage D0 and write the reproduction check."""
    global SEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    D.INDEX.mkdir(parents=True, exist_ok=True)
    D.READ_LOG.write_text("")
    SEL = json.loads((D.C.SELECTION / "E3_grid.json").read_text())
    reps = [r for r in PP.rep_dirs() if not r.name.startswith("fall")][: a.limit or None]
    print(f"[{D.now()}] D0: {len(reps)} replicates", flush=True)
    sessions, checks = [], []
    with Pool(a.workers) as pool:
        for i, (s, c) in enumerate(pool.imap_unordered(job, reps), 1):
            sessions += s
            checks += c
            if i % 10 == 0 or i == len(reps):
                print(f"[{D.now()}] {i}/{len(reps)}", flush=True)
    pd.DataFrame(sessions).to_csv(D.INDEX / "d0_sessions.csv", index=False)
    e3 = pd.read_csv(D.C.INDEX / "postproc.csv")
    ck = pd.DataFrame(checks)
    ref = e3.melt(id_vars=["run_id", "decoder"], value_vars=[f"{D.TEST}.{k}" for k in CHECKS],
                      var_name="metric", value_name="e3")
    ref["metric"] = ref.metric.str[len(D.TEST) + 1:]
    ck = ck.merge(ref, on=["run_id", "decoder", "metric"], how="left")
    ck["abs_diff"] = (ck.reproduced - ck.e3).abs()
    ck.to_csv(D.INDEX / "d0_check.csv", index=False)
    read = [l for l in D.READ_LOG.read_text().splitlines() if l]
    bad_sets = sorted({l.rsplit("/", 1)[-1] for l in read} - {f"{p}.npz" for p in D.ALLOWED})
    out = {"replicates": len(reps), "rows_checked": len(ck), "missing_reference": int(ck.e3.isna().sum()),
           "max_abs_diff": float(ck.abs_diff.max()), "V2_pass": bool(ck.abs_diff.max() <= 1e-9 and ck.e3.notna().all()),
           "sets_read": sorted({l.rsplit("/", 1)[-1] for l in read}), "V3_pass": not bad_sets,
           "sessions": len(sessions), "finished": D.now()}
    D.write_json(D.INDEX / "d0_summary.json", out)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
