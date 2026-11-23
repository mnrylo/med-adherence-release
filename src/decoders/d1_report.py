"""Stage D1 report: every frozen-classifier decoder, participant-level, on test_wild.

    python -m decoders.d1_report

The mode-filter width is chosen per configuration on validation (max mean val seg F1@25 over the
replicates, tie-break Order Exact Match), like the E3 grids; test values never select.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from decoders import dcommon as D
from decoders import stats as S

warnings.filterwarnings("ignore", category=RuntimeWarning)
GROUPS = [("E2", "finalist", "E2 finalists (2 s CNN-BiLSTM, acc+gyr)"),
          ("E2", "window_ref", "E2 short-window refs (0.75 / 1 s)"),
          ("E1", "finalist", "E1 finalists (5 s, gestures only)"),
          ("E1", "window_ref", "E1 short-window refs")]
DECODERS = ["none", "mode", "hmm", "hsmm", "crf", "crfdur", "refiner", "crf_bias", "crf_fixs"]
METRICS = ["seg_f1_25", "seg_f1_10", "seg_f1_50", "order_exact", "g3_sens", "g3_spec", "score10_sens", "score10_spec"]
NAMES = {"seg_f1_25": "Seg F1@25", "seg_f1_10": "Seg F1@10", "seg_f1_50": "Seg F1@50", "order_exact": "Order Exact Match",
         "g3_sens": "G3-rule sens", "g3_spec": "G3-rule spec", "score10_sens": "Score sens (≤1 FA/h)",
         "score10_spec": "Score spec (≤1 FA/h)"}


def load():
    """Session rows and training info of D1 and D1b."""
    sess = pd.concat(pd.read_csv(f) for d in ("d1", "d1b") for f in sorted((D.INDEX / d).glob("*.csv.gz")))
    info = [json.loads(f.read_text()) for f in sorted((D.INDEX / "d1").glob("*.json"))]
    for f in sorted((D.INDEX / "d1b").glob("*.json")):
        b = json.loads(f.read_text())
        for i in info:
            if i["run_id"] == b["run_id"]:
                for k in ("training", "exports", "test_external"):
                    i[k].update(b[k])
    return sess, info


def choose_mode(sess: pd.DataFrame) -> dict:
    """Mode-filter width per configuration, chosen on validation."""
    v = sess[(sess.set == "val") & sess.decoder.str.startswith("mode_")]
    g = v.groupby(["config_key", "decoder", "run_id"]).agg(tp=("tp25", "sum"), fp=("fp25", "sum"), fn=("fn25", "sum"),
                                                            ex=("exact", "mean")).reset_index()
    g["f1"] = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
    m = g.groupby(["config_key", "decoder"])[["f1", "ex"]].mean().reset_index()
    return {c: d.sort_values(["f1", "ex"], ascending=False).iloc[0].decoder for c, d in m.groupby("config_key")}


def fmt(t, d, kind):
    """Table cell of a decoder."""
    if d not in t.index or np.isnan(t.loc[d, "mean"]):
        return "—"
    if kind == "abs":
        return f"{100 * t.loc[d, 'mean']:.1f} [{100 * t.loc[d, 'ci_lo']:.1f}, {100 * t.loc[d, 'ci_hi']:.1f}]"
    if d == "hmm":
        return "ref."
    return f"{100 * t.loc[d, 'diff']:+.1f} [{100 * t.loc[d, 'diff_lo']:+.1f}, {100 * t.loc[d, 'diff_hi']:+.1f}] p={t.loc[d, 'p_holm']:.3f}"


def main():
    """Write reports/D1/REPORT.md and index/d1_participant_stats.csv."""
    sess, info = load()
    mode = choose_mode(sess)
    test = sess[sess.set == D.TEST].copy()
    keep = test.decoder.isin([d for d in DECODERS if d != "mode"]) | (test.decoder == test.config_key.map(mode))
    test = test[keep]
    test.loc[test.decoder.str.startswith("mode_"), "decoder"] = "mode"
    ext = pd.DataFrame([{"run_id": i["run_id"], "decoder": d, **v} for i in info for d, v in i["test_external"].items()])
    md = ["# Stage D1: decoders on a frozen classifier", "",
          f"Generated {D.now()}. {len(info)} replicates. Test set: `{D.TEST}` only.", "",
          "Decoders: none (E1/E2 argmax + merge rule), mode (majority filter, width chosen on validation), "
          "hmm and hsmm (E3 parameters), crf (the HMM graph without duration chains, trained with the CRF "
          "segmentation loss on validation), crfdur (the same with the duration chains of the selected HMM), "
          "refiner (dilated temporal CNN on the probabilities, trained on validation). "
          "Post hoc (D1b, declared after D1 showed the crf emission scale collapsing on E1): crf_bias adds a "
          "per-class additive emission bias outside the scale; crf_fixs freezes the scale at the HMM's 1/temperature.", "",
          "Intake: G3 rule for every decoder; score (phased LLR with each model's own potentials, ≤ 1 false "
          "event/h on external calibration) for hmm, crf, crfdur, crf_bias and crf_fixs.", "",
          "Mode widths chosen: " + ", ".join(f"`{c}` {d[5:]} ×0.1 s" for c, d in sorted(mode.items())), ""]
    allres = []
    for i, (exp, role, title) in enumerate(GROUPS, 1):
        g = test[(test.experiment == exp) & (test.role == role)]
        if g.empty:
            continue
        r = S.compare(g, DECODERS, "hmm", METRICS)
        r.insert(0, "group", f"{exp}/{role}")
        allres.append(r)
        md += [f"## {i}. {title}", "", f"{g.config_key.nunique()} configurations, {g.run_id.nunique()} replicates, "
               f"{g.subject.nunique()} participants. Mean [95 % CI]; Δ vs HMM [95 % CI], Holm-adjusted Wilcoxon p.", "",
               "| Metric | " + " | ".join(DECODERS) + " |", "|---|" + "---|" * len(DECODERS)]
        for m in METRICS:
            t = r[r.metric == m].set_index("decoder")
            md.append(f"| {NAMES[m]} | " + " | ".join(fmt(t, d, "abs") for d in DECODERS) + " |")
        md += ["", "| Δ vs HMM | " + " | ".join(DECODERS) + " |", "|---|" + "---|" * len(DECODERS)]
        for m in METRICS:
            t = r[r.metric == m].set_index("decoder")
            md.append(f"| {NAMES[m]} | " + " | ".join(fmt(t, d, "diff") for d in DECODERS) + " |")
        e = ext[ext.run_id.isin(g.run_id.unique())].copy()
        e.loc[e.decoder.str.startswith("mode_"), "decoder"] = e.decoder
        e = e[e.decoder.isin(DECODERS) | e.decoder.isin(set(mode.values()))]
        e.loc[e.decoder.str.startswith("mode_"), "decoder"] = "mode"
        ee = e.groupby("decoder")[["g3_rule_per_h", "score_fa10_per_h"]].mean().reindex(DECODERS)
        md += ["", "False events per hour on `test_external` (mean over replicates):", "",
               "| | " + " | ".join(DECODERS) + " |", "|---|" + "---|" * len(DECODERS),
               "| G3 rule | " + " | ".join("—" if np.isnan(v) else f"{v:.2f}" for v in ee.g3_rule_per_h) + " |",
               "| Score (≤ 1 FA/h) | " + " | ".join("—" if np.isnan(v) else f"{v:.2f}" for v in ee.score_fa10_per_h) + " |", ""]
    pd.concat(allres).to_csv(D.INDEX / "d1_participant_stats.csv", index=False)
    tr = pd.DataFrame([{"run_id": i["run_id"], **{f"{k}_steps": v["steps"] for k, v in i["training"].items()},
                        "crf_scale": i["exports"]["crf"]["scale"], "crfdur_scale": i["exports"]["crfdur"]["scale"],
                        "crf_bias_scale": i["exports"].get("crf_bias", {}).get("scale", np.nan),
                        "seconds": i["seconds"]} for i in info])
    md += ["## Training of the learned decoders", "",
           f"Steps chosen on the held-out half of validation (median [min, max]): "
           + "; ".join(f"{k} {tr[k + '_steps'].median():.0f} [{tr[k + '_steps'].min()}, {tr[k + '_steps'].max()}]"
                       for k in ("crf", "crfdur", "refiner", "crf_bias", "crf_fixs") if k + "_steps" in tr)
           + f". Learned emission scale: crf {tr.crf_scale.median():.2f}, crfdur {tr.crfdur_scale.median():.2f}, crf_bias {tr.crf_bias_scale.median():.2f} "
           f"(E3 temperature grid 0.5–4 → scale 0.25–2). Median time per replicate {tr.seconds.median() / 60:.1f} min.", "",
           "## Notes", "",
           "- HSMM has no score: its LLR needs a semi-Markov forward pass, not implemented.",
           "- The crf / crfdur score keeps the E3 2 Hz duration chains; the learned 10 Hz self-loops do not transfer.",
           "- Specificity uses the participants with sessions without intake (7 of 10).", ""]
    out = D.REPORTS / "D1" / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
