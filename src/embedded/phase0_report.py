"""Phase 0 report: figures and a compact results table from phase0_eval.py's tables.

    python -m embedded.phase0_report

Output: outputs/embedded/phase0/figures/{variants,fixed_lag,latency_cdf}.{png,pdf}
        outputs/embedded/phase0/RESULTS_DATA.md (numbers only, mean ± sd over the 9 fold replicates)
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from embedded import emb as E

T = E.PHASE0 / "tables"
FIG = E.PHASE0 / "figures"
LABEL = {"zerophase": "zero-phase (as trained)", "causal": "causal, not retrained", "causal_trained": "causal, retrained",
         "causal_trained_fp32": "retrained, ONNX FP32", "causal_trained_fp16": "retrained, ONNX FP16",
         "causal_trained_int8_dynamic": "retrained, ONNX INT8 dyn.",
         "causal_trained_int8_dynamic_b1": "retrained, ONNX INT8 dyn., batch 1"}
MAIN = [("test_sequences.window.macro_f1", "Window macro F1\n(blind sequences)"),
        ("test_sequences.hmm.seg_f1_25", "Seg F1@25\n(blind sequences)"),
        ("test_sequences.hmm.order_exact_match", "Order EM\n(blind sequences)"),
        ("test_wild.hmm.order_exact_match", "Order EM\n(daily activity)"),
        ("test_wild.llr_fa10.sensitivity", "LLR sensitivity ≤1/h\n(daily activity)")]


def save(fig, name):
    """Save a figure as PNG and PDF."""
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fold_rows(A):
    """Fold replicates with recalibrated thresholds."""
    return A[(A.replicate != E.DEPLOY_REP) & (A.thresholds == "recalibrated")]


def fig_variants(A):
    """Filter-variant figure."""
    d = fold_rows(A)
    vs = [v for v in LABEL if v in set(d.variant)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    w = 0.8 / len(vs)
    for i, v in enumerate(vs):
        g = d[d.variant == v]
        ax.bar(np.arange(len(MAIN)) + i * w, [g[m].mean() for m, _ in MAIN], w, yerr=[g[m].std() for m, _ in MAIN],
               label=LABEL[v], capsize=2)
    ax.set_xticks(np.arange(len(MAIN)) + w * (len(vs) - 1) / 2, [t for _, t in MAIN], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)
    save(fig, "variants")


def fig_fixed_lag(B, v):
    """Fixed-lag Viterbi figure."""
    d = B[(B.variant == v) & (B.set != "val")]
    lags = [x for x in d.lag_s.unique() if x != "inf"]
    x = [float(s) for s in lags]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for ax, m, t in ((axs[0], "seg_f1_25", "Seg F1@25"), (axs[1], "order_exact_match", "Order Exact Match")):
        for st, name in (("test_sequences", "blind sequences"), ("test_wild", "daily activity")):
            g = d[d.set == st].groupby("lag_s")[m]
            ax.errorbar(x, [g.mean()[s] for s in lags], [g.std()[s] for s in lags], marker="o", ms=3, capsize=2,
                        label=name)
            ax.axhline(g.mean()["inf"], ls="--", lw=0.8, color=ax.lines[-1].get_color())
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Viterbi lag (s)")
        ax.set_title(t, fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    save(fig, "fixed_lag")


def fig_latency(L, v):
    """Alarm-latency distribution figure."""
    d = L[(L.variant == v) & (L.detected == 1) & (L.budget == "fa10")]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for (step, rule), g in d.groupby(["step_s", "rule"]):
        x = np.sort(g.latency_s.to_numpy())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), ls="-" if rule == "first" else ":",
                label=f"step {step} s, {rule.replace('_', ' ')}")
    ax.axvline(60, color="k", lw=0.8, ls="--")
    ax.set_xlabel("alarm latency after intake end (s)")
    ax.set_ylabel("cumulative fraction of detected intakes")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    save(fig, "latency_cdf")


def ms(g, m):
    """Mean ± sd text."""
    return f"{g[m].mean():.3f} ± {g[m].std():.3f}"


def main():
    """Write the phase-0 figures and RESULTS_DATA.md."""
    A = pd.read_csv(T / "filter.csv")
    B = pd.read_csv(T / "fixed_lag.csv", dtype={"lag_s": str})
    C = pd.read_csv(T / "llr_streaming.csv")
    L = pd.read_csv(T / "llr_latency_sessions.csv")
    summ = json.loads((E.PHASE0 / "summary.json").read_text())
    v = summ["streaming_variant"]
    fig_variants(A)
    fig_fixed_lag(B, v)
    fig_latency(L, v)

    cols = MAIN + [("test_wild.llr_fa10.specificity", "LLR specificity (daily)"),
                   ("test_external.llr_fa10.events_per_hour", "FA/h (test_external)")]
    lines = ["# Phase 0 results (numbers)", "", f"Created {E.now()} from outputs/embedded/phase0/tables. Mean ± sd over the "
             "9 fold replicates; LLR at the ≤ 1 FA/h budget (fa10).", "",
             "## Variants (offline decoding)", "",
             "| Variant | Thresholds | " + " | ".join(t.replace("\n", " ") for _, t in cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    d = A[A.replicate != E.DEPLOY_REP]
    for (var, thr), g in d.groupby(["variant", "thresholds"], sort=False):
        lines.append(f"| {LABEL.get(var, var)} | {thr} | " + " | ".join(ms(g, m) for m, _ in cols) + " |")
    lines += ["", f"## Fixed-lag Viterbi ({LABEL.get(v, v)})", "", "| Lag (s) | Record latency (s) | "
              "Seg F1@25 seq | Order EM seq | Seg F1@25 daily | Order EM daily |", "|---|---|---|---|---|---|"]
    for lag in B.lag_s.unique():
        r = B[(B.variant == v) & (B.lag_s == lag)]
        s, w = r[r.set == "test_sequences"], r[r.set == "test_wild"]
        lines.append(f"| {lag} | {s.record_latency_s.iloc[0]:.1f} | {ms(s, 'seg_f1_25')} | {ms(s, 'order_exact_match')} | "
                     f"{ms(w, 'seg_f1_25')} | {ms(w, 'order_exact_match')} |")
    lines += ["", f"## Streaming LLR ({LABEL.get(v, v)}, test_wild, pooled latencies of detected intakes)", "",
              "| Step (s) | Budget | Sensitivity | Specificity | FA/h | Latency first: median / p95 (s) | "
              "Latency run end: median / p95 (s) | From G3 end, first: median / p95 (s) |",
              "|---|---|---|---|---|---|---|---|"]
    det = L[(L.variant == v) & (L.detected == 1)]
    for (step, bud), g in C[C.variant == v].groupby(["step_s", "budget"]):
        q = det[(det.step_s == step) & (det.budget == bud)]

        def mp(rule, key="latency_s"):
            x = q[q.rule == rule][key].dropna()
            return f"{x.median():.1f} / {np.percentile(x, 95):.1f}"
        lines.append(f"| {step} | {bud} | {ms(g, 'test_wild.sensitivity')} | {ms(g, 'test_wild.specificity')} | "
                     f"{ms(g, 'test_external.events_per_hour')} | {mp('first')} | {mp('run_end')} | "
                     f"{mp('first', 'latency_g3_s')} |")
    info_f = E.PHASE0 / "onnx" / "info.json"
    if info_f.exists():
        info = json.loads(info_f.read_text())
        lines += ["", "## ONNX precisions (per replicate: size, agreement with FP32)", "",
                  "| Precision | Size (KiB) | argmax agreement vs FP32 (min over replicates) | max abs diff |",
                  "|---|---|---|---|"]
        for prec in ("fp32", "fp16", "int8_dynamic", "int8_static"):
            r = [x[prec] for x in info.values() if prec in x and "bytes" in x[prec]]
            if not r:
                continue
            agr = [x.get("argmax_agreement_vs_fp32") for x in r if x.get("argmax_agreement_vs_fp32") is not None]
            dif = [x.get("max_abs_diff_vs_fp32") for x in r if x.get("max_abs_diff_vs_fp32") is not None]
            lines.append(f"| {prec} | {np.mean([x['bytes'] for x in r]) / 1024:.0f} | "
                         f"{min(agr):.4f} | {max(dif):.3f} |" if agr else
                         f"| {prec} | {np.mean([x['bytes'] for x in r]) / 1024:.0f} | reference | – |")
    (E.PHASE0 / "RESULTS_DATA.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
