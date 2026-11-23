"""Stage D0 report: reproduction checks and the participant-level baseline of none / HMM / HSMM.

    python -m decoders.d0_report
"""
from __future__ import annotations

import json

import pandas as pd

from decoders import dcommon as D
from decoders import stats as S

GROUPS = [("E2", "finalist", "E2 finalists (2 s CNN-BiLSTM, acc+gyr)"),
          ("E2", "window_ref", "E2 short-window refs (0.75 / 1 s)"),
          ("E1", "finalist", "E1 finalists (5 s, gestures only)"),
          ("E1", "window_ref", "E1 short-window refs")]
METRICS = ["seg_f1_25", "seg_f1_10", "seg_f1_50", "order_exact", "g3_sens", "g3_spec", "llr10_sens", "llr10_spec"]
NAMES = {"seg_f1_25": "Seg F1@25", "seg_f1_10": "Seg F1@10", "seg_f1_50": "Seg F1@50", "order_exact": "Order Exact Match",
         "g3_sens": "G3-rule sens", "g3_spec": "G3-rule spec", "llr10_sens": "LLR sens (≤1 FA/h)",
         "llr10_spec": "LLR spec (≤1 FA/h)"}


def pct(x):
    """Percentage text."""
    return f"{100 * x:.1f}"


def pooled(df):
    """Session-pooled means over replicates, as E3 reports them."""
    out = {}
    for dec, d in df.groupby("decoder"):
        r = d.groupby("run_id").agg(tp=("tp25", "sum"), fp=("fp25", "sum"), fn=("fn25", "sum"), ex=("exact", "mean"))
        out[dec] = {"seg_f1_25": float((2 * r.tp / (2 * r.tp + r.fp + r.fn)).mean()), "order_exact": float(r.ex.mean())}
    return out


def main():
    """Write reports/D0/REPORT.md and index/d0_participant_stats.csv."""
    summ = json.loads((D.INDEX / "d0_summary.json").read_text())
    df = pd.read_csv(D.INDEX / "d0_sessions.csv")
    n_subj = df.groupby("subject").sid.nunique()
    md = ["# Stage D0: reproduction and participant-level baseline", "",
          f"Generated {D.now()}. Test set: `{D.TEST}` only (lab sessions interleaved with InGesture/WISDM).", "",
          "## 1. Verification", "",
          "| Check | Result | Note |", "|---|---|---|",
          f"| V2 reproduction | {'PASS' if summ['V2_pass'] else 'FAIL'} | {summ['rows_checked']} values ({summ['replicates']} replicates × 3 decoders × 10 metrics) against the E3 postproc.csv; max abs diff {summ['max_abs_diff']:.1e} |",
          f"| V3 data isolation | {'PASS' if summ['V3_pass'] else 'FAIL'} | sets opened: {', '.join(summ['sets_read'])} |", "",
          "## 2. Test participants", "",
          "Sessions of `test_wild` per participant. Every participant weighs the same in the participant-level "
          "means below, so Subject15 (1 session) weighs as much as Subject01 (50); the session-pooled E3 numbers "
          "are shown next to them for comparison.", "",
          "| " + " | ".join(n_subj.index.str.replace("lab_", "")) + " |", "|" + "---|" * len(n_subj),
          "| " + " | ".join(str(v) for v in n_subj.values) + " |", ""]
    allres = []
    for i, (exp, role, title) in enumerate(GROUPS, 3):
        g = df[(df.experiment == exp) & (df.role == role)]
        if g.empty:
            continue
        r = S.compare(g, ["none", "hmm", "hsmm"], "hmm", METRICS)
        r.insert(0, "group", f"{exp}/{role}")
        allres.append(r)
        po = pooled(g)
        md += [f"## {i}. {title}", "",
               f"{g.config_key.nunique()} configurations, {g.run_id.nunique()} replicates. Participant-level mean "
               "[95 % cluster-bootstrap CI]; Δ = decoder − HMM [95 % CI], Holm-adjusted Wilcoxon p over participants.", "",
               "| Metric | none | HMM | HSMM | Δ none | Δ HSMM |", "|---|---|---|---|---|---|"]
        for m in METRICS:
            t = r[r.metric == m].set_index("decoder")
            if t.empty:
                continue
            cell = lambda d: f"{pct(t.loc[d, 'mean'])} [{pct(t.loc[d, 'ci_lo'])}, {pct(t.loc[d, 'ci_hi'])}]"
            delta = lambda d: (f"{100 * t.loc[d, 'diff']:+.1f} [{100 * t.loc[d, 'diff_lo']:+.1f}, "
                               f"{100 * t.loc[d, 'diff_hi']:+.1f}], p = {t.loc[d, 'p_holm']:.3f}")
            md.append(f"| {NAMES[m]} | {cell('none')} | {cell('hmm')} | {cell('hsmm')} | {delta('none')} | {delta('hsmm')} |")
        md += ["", "Session-pooled (as in E3): " + "; ".join(
            f"{d}: seg F1@25 {pct(v['seg_f1_25'])}, Order EM {pct(v['order_exact'])}" for d, v in po.items()), ""]
    res = pd.concat(allres)
    res.to_csv(D.INDEX / "d0_participant_stats.csv", index=False)
    md += ["## Notes", "",
           "- Specificity uses only the participants with sessions without intake (7 of 10).",
           "- The LLR decision is the E3 one: computed with the configuration's HMM for every decoder, so its "
           "columns do not depend on the decoder. Decoder-specific intake decisions come in stage D1.",
           "- With 10 participants the smallest attainable two-sided Wilcoxon p is 0.002, before Holm correction.", ""]
    out = D.REPORTS / "D0" / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
