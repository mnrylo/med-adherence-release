"""Stage D3 report (analysis A5): accuracy versus latency of the online decoders on panel F.

    python -m decoders.d3_report

Participant-level means and 95 % cluster-bootstrap CIs (stats.compare, 2,000 draws) per group of
panel F, decoder and lag; the loss of each decoder against its own offline path; the record
latency (lag + classifier look-ahead); G3-rule false events per hour on test_external.
Output: reports/D3/REPORT.md, reports/D3/lag_curves.png, index/d3_participant_stats.csv.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from decoders import d1_report as D1R
from decoders import dcommon as D
from decoders import stats as S

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = D.REPORTS / "D3"
DEC = ["mode", "hmm", "hsmm", "crf", "crfdur", "crf_fixs"]
NAME = {"mode": "Mode", "hmm": "HMM", "hsmm": "HSMM", "crf": "CRF", "crfdur": "CRF-dur", "crf_fixs": "CRF-fixed-scale*"}
LAGS = ["0", "1", "2", "5", "10", "inf"]
COLOR = {"mode": "#1baf7a", "hmm": "#2a78d6", "hsmm": "#eb6834", "crf": "#8e5bd0", "crfdur": "#c2408f", "crf_fixs": "#d4a017"}
METRICS = ["seg_f1_25", "order_exact", "g3_sens", "g3_spec"]


def offline_label(dec):
    """Lag label of a decoder's offline reference."""
    return "inf_relabel" if dec == "hsmm" else "inf"


def main():
    """Write the D3 tables and report."""
    OUT.mkdir(parents=True, exist_ok=True)
    sess = pd.concat(pd.read_csv(f) for f in sorted((D.INDEX / "d3").glob("*.csv.gz")))
    info = [json.loads(f.read_text()) for f in sorted((D.INDEX / "d3").glob("*.json"))]
    sess["lag_s"] = sess.lag_s.astype(str)
    sess = sess[sess.lag_s.isin(LAGS + ["inf_relabel"])]
    sess["decoder"] = sess.decoder + "@" + sess.lag_s
    look = pd.DataFrame([{"config_key": i["config_key"], "lookahead_s": i["lookahead_s"]} for i in info]).drop_duplicates()
    ext = pd.DataFrame([{"run_id": i["run_id"], "decoder": f"{d}@{l}", "per_h": v}
                        for i in info for d, lv in i["g3_rule_per_h"].items() for l, v in lv.items()])
    allstats, L = [], [f"# Stage D3: online (fixed-lag) decoding on panel F", "",
                       f"Generated {D.now()}. Test set: `{D.TEST}`. {sess.run_id.nunique()} replicates. "
                       "Decoder parameters as selected in D1 on validation (not re-tuned per lag). Lag = seconds between a "
                       "frame and the moment its label is committed; record latency = lag + the classifier look-ahead "
                       "(half a window + one stride). `inf` = offline; for HSMM the offline D1 decoder also has the "
                       "relabel pass (`inf_relabel`), which needs the whole path and is not applied online. "
                       "* CRF-fixed-scale is post hoc (D1b).", "", "![lag curves](lag_curves.png)", ""]
    labels = [f"{d}@{l}" for d in DEC for l in LAGS] + ["hsmm@inf_relabel"]
    for exp, role, title in D1R.GROUPS:
        g = sess[(sess.experiment == exp) & (sess.role == role)]
        if g.empty:
            continue
        st = S.compare(g, labels, "hmm@inf", METRICS, n_boot=2000)
        st["group"] = title
        allstats.append(st)
        la = look[look.config_key.isin(g.config_key.unique())].lookahead_s
        L += [f"## {title}", "", f"{g.config_key.nunique()} configurations, {g.run_id.nunique()} replicates; "
              f"classifier look-ahead {la.min():.2f}–{la.max():.2f} s.", ""]
        for m in METRICS:
            t = st[st.metric == m].set_index("decoder")
            L += [f"### {D1R.NAMES[m]}", "", "| Decoder | " + " | ".join(f"lag {l} s" if l != "inf" else "offline" for l in LAGS)
                  + " | offline (D1) |", "|---" * (len(LAGS) + 2) + "|"]
            for d in DEC:
                cells = [f"{100 * t.at[f'{d}@{l}', 'mean']:.1f} [{100 * t.at[f'{d}@{l}', 'ci_lo']:.1f}, {100 * t.at[f'{d}@{l}', 'ci_hi']:.1f}]"
                         for l in LAGS]
                off = f"{d}@{offline_label(d)}"
                L.append(f"| {NAME[d]} | " + " | ".join(cells) + f" | {100 * t.at[off, 'mean']:.1f} |")
            L.append("")
        t = st[st.metric == "seg_f1_25"].set_index("decoder")
        L += ["Seg F1@25 lost against the decoder's own offline result (points): " + "; ".join(
            f"{NAME[d]} " + ", ".join(f"{l} s {100 * (t.at[f'{d}@{l}', 'mean'] - t.at[f'{d}@{offline_label(d)}', 'mean']):+.1f}"
                                     for l in LAGS[:-1]) for d in DEC), ""]
        e = ext[ext.run_id.isin(g.run_id.unique())].groupby("decoder").per_h.mean()
        L += ["G3-rule false events per hour on test_external (mean over replicates): " + "; ".join(
            f"{NAME[d]} " + ", ".join(f"{l}: {e.get(f'{d}@{l}', np.nan):.2f}" for l in LAGS) for d in DEC), ""]
    st = pd.concat(allstats)
    st.to_csv(D.INDEX / "d3_participant_stats.csv", index=False)
    figure(st)
    read = sorted({l.rsplit("/", 1)[-1] for l in D.READ_LOG.read_text().splitlines() if l})
    L += ["## Data check", "", f"Sets read by any stage so far: {', '.join(read)} (allowlist: {', '.join(D.ALLOWED)})."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def figure(st):
    """Accuracy vs. lag figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = st.group.unique()
    fig, axes = plt.subplots(2, len(groups), figsize=(3.6 * len(groups), 6.5), sharex=True)
    x = {l: (float(l) if l != "inf" else 15.0) for l in LAGS}
    for j, gname in enumerate(groups):
        for i, m in enumerate(["seg_f1_25", "order_exact"]):
            ax = axes[i, j]
            t = st[(st.group == gname) & (st.metric == m)].set_index("decoder")
            for d in DEC:
                xs = [x[l] for l in LAGS]
                ys = [100 * t.at[f"{d}@{l}", "mean"] for l in LAGS]
                lo = [100 * t.at[f"{d}@{l}", "ci_lo"] for l in LAGS]
                hi = [100 * t.at[f"{d}@{l}", "ci_hi"] for l in LAGS]
                ax.plot(xs, ys, marker="o", ms=3, lw=1.8, color=COLOR[d], label=NAME[d], ls="--" if d == "crf_fixs" else "-")
                ax.fill_between(xs, lo, hi, color=COLOR[d], alpha=0.08, lw=0)
            ax.set_xticks([x[l] for l in LAGS], [l if l != "inf" else "off" for l in LAGS])
            ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(gname, fontsize=9)
            if i == 1:
                ax.set_xlabel("Lag (s)")
            if j == 0:
                ax.set_ylabel(D1R.NAMES[m])
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "lag_curves.png", dpi=150)


if __name__ == "__main__":
    main()
