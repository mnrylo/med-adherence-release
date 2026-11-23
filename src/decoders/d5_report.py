"""Stage D5: analyses A1-A6 in one report .

    python -m decoders.d5_report

Reads only the outputs of the earlier stages:
  A1 main comparison and A6 intake  <- index/d1_participant_stats.csv (D1 + D1b)
  A2 gain vs classifier quality     <- index/d2_bins.csv (D2)
  A3 error decomposition            <- index/d1, index/d1b session rows (test_wild), computed here
  A4 tuning-data fraction           <- index/a4/test (a4_tuning_data.py), computed here
  A5 online decoding                <- index/d3_participant_stats.csv (D3)
A3 and A4 are participant-level (metrics per participant and replicate, averaged over replicates,
then over the 10 participants); their 95 % intervals resample participants only (10,000 draws).
Output: reports/D5/REPORT.md, reports/D5/errors.png, reports/D5/tuning_data.png, index/d5_*.csv.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from decoders import d1_report as D1R
from decoders import dcommon as D

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = D.REPORTS / "D5"
NAME = {"none": "None", "mode": "Mode", "hmm": "HMM", "hsmm": "HSMM", "crf": "CRF", "crfdur": "CRF-dur",
        "refiner": "Refiner", "crf_bias": "CRF-bias*", "crf_fixs": "CRF-fixed-scale*"}
COLOR = {"none": "#6b7280", "mode": "#1baf7a", "hmm": "#2a78d6", "hsmm": "#eb6834", "crf": "#8e5bd0",
         "crfdur": "#c2408f", "refiner": "#0f8b8d", "crf_bias": "#a3a3a3", "crf_fixs": "#d4a017"}
GROUP_KEYS = [f"{e}/{r}" for e, r, _ in D1R.GROUPS]
GROUP_TITLE = {f"{e}/{r}": t for e, r, t in D1R.GROUPS}
rng = np.random.default_rng(0)


def ci(x, n=10000):
    """95 % bootstrap interval of a mean over participants."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


ERR = {"spurious": "Spurious segments (no true segment at IoU 0.10)", "missed": "Missed segments (IoU < 0.10)",
       "boundary": "Found but poorly bounded (IoU 0.10-0.50)", "over_seg": "Predicted / true segments",
       "order_err": "Order errors (1 - Order Exact Match)", "edit": "Normalized order edit distance"}


def errors(sess: pd.DataFrame) -> pd.DataFrame:
    """Error decomposition (spurious, missed, boundary, over-segmentation) per participant and replicate."""
    rows = []
    for (grp, dec, rid, subj), g in sess.groupby(["group", "decoder", "run_id", "subject"]):
        nt = max(1, g.n_true_seg.sum())
        rows.append({"group": grp, "decoder": dec, "run_id": rid, "subject": subj,
                     "spurious": g.fp10.sum() / nt, "missed": g.fn10.sum() / nt,
                     "boundary": (g.tp10.sum() - g.tp50.sum()) / nt, "over_seg": g.n_pred_seg.sum() / nt,
                     "order_err": 1 - g.exact.mean(), "edit": g.edit.mean()})
    per = pd.DataFrame(rows).groupby(["group", "decoder", "subject"]).mean(numeric_only=True).reset_index()
    out = []
    for (grp, dec), g in per.groupby(["group", "decoder"]):
        r = {"group": grp, "decoder": dec}
        for m in ERR:
            r[m] = g[m].mean()
            r[f"{m}_lo"], r[f"{m}_hi"] = ci(g[m])
        out.append(r)
    return pd.DataFrame(out)


def tuning() -> pd.DataFrame:
    """Metrics by validation fraction (analysis A4)."""
    files = sorted((D.INDEX / "a4" / "test").glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    s = pd.concat(pd.read_csv(f) for f in files)
    s["group"] = s.experiment + "/" + s.role
    rows = []
    for (grp, frac, dec, rid, subj), g in s.groupby(["group", "frac", "decoder", "run_id", "subject"]):
        tp, fp, fn = g.tp25.sum(), g.fp25.sum(), g.fn25.sum()
        rows.append({"group": grp, "frac": frac, "decoder": dec, "run_id": rid, "subject": subj,
                     "seg_f1_25": 2 * tp / max(1, 2 * tp + fp + fn), "order_exact": g.exact.mean(), "n_val": g.n_val.iat[0]})
    per = pd.DataFrame(rows).groupby(["group", "frac", "decoder", "subject"]).mean(numeric_only=True).reset_index()
    out = []
    for (grp, frac, dec), g in per.groupby(["group", "frac", "decoder"]):
        r = {"group": grp, "frac": frac, "decoder": dec, "n_val": g.n_val.mean()}
        for m in ("seg_f1_25", "order_exact"):
            r[m] = g[m].mean()
            r[f"{m}_lo"], r[f"{m}_hi"] = ci(g[m])
        out.append(r)
    return pd.DataFrame(out)


def fig_errors(e: pd.DataFrame):
    """Error decomposition figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    decs = [d for d in NAME if d in set(e.decoder)]
    fig, axes = plt.subplots(1, len(GROUP_KEYS), figsize=(4.2 * len(GROUP_KEYS), 3.8), sharey=False)
    parts = [("spurious", "#d1495b"), ("missed", "#2e86ab"), ("boundary", "#edae49")]
    for ax, grp in zip(axes, GROUP_KEYS):
        t = e[e.group == grp].set_index("decoder").reindex(decs)
        bottom = np.zeros(len(decs))
        for m, c in parts:
            ax.bar(range(len(decs)), t[m], bottom=bottom, color=c, edgecolor="white", linewidth=1, label=ERR[m])
            bottom += t[m].to_numpy()
        ax.set_xticks(range(len(decs)), [NAME[d] for d in decs], rotation=60, ha="right", fontsize=7)
        ax.set_title(GROUP_TITLE[grp], fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Errors per true segment")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "errors.png", dpi=150)


def fig_tuning(t: pd.DataFrame):
    """Validation-fraction figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(GROUP_KEYS), figsize=(4.2 * len(GROUP_KEYS), 3.6))
    for ax, grp in zip(axes, GROUP_KEYS):
        g = t[t.group == grp]
        for dec in [d for d in NAME if d in set(g.decoder)]:
            h = g[g.decoder == dec].sort_values("frac")
            ax.plot(100 * h.frac, 100 * h.seg_f1_25, marker="o", ms=3, lw=1.8, color=COLOR[dec], label=NAME[dec],
                    ls="--" if dec == "crf_fixs" else "-")
            ax.fill_between(100 * h.frac, 100 * h.seg_f1_25_lo, 100 * h.seg_f1_25_hi, color=COLOR[dec], alpha=0.08, lw=0)
        ax.set_xticks([25, 50, 100])
        ax.set_xlabel("Validation streams used (%)")
        ax.set_title(GROUP_TITLE[grp], fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Seg F1@25")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "tuning_data.png", dpi=150)


def f(x):
    """Percentage text."""
    return "—" if pd.isna(x) else f"{100 * x:.1f}"


def main():
    """Write reports/D5/REPORT.md."""
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# Stage D5: analyses A1-A6", "", f"Generated {D.now()}. Test set: `{D.TEST}` only. "
         "`*` = post-hoc variant (D1b). Participant-level throughout (10 test participants).", ""]

    st = pd.read_csv(D.INDEX / "d1_participant_stats.csv")
    L += ["## A1. Main comparison (panel F, offline)", "", "Mean [95 % cluster-bootstrap CI]; full tables with differences "
          "and Holm-adjusted p values in reports/D1/REPORT.md.", ""]
    for m in ["seg_f1_25", "order_exact"]:
        L += [f"### {D1R.NAMES[m]}", "", "| Decoder | " + " | ".join(GROUP_TITLE[g] for g in GROUP_KEYS) + " |",
              "|---" * (len(GROUP_KEYS) + 1) + "|"]
        for dec in NAME:
            cells = []
            for g in GROUP_KEYS:
                r = st[(st.group == g) & (st.metric == m) & (st.decoder == dec)]
                cells.append(f"{f(r['mean'].iat[0])} [{f(r.ci_lo.iat[0])}, {f(r.ci_hi.iat[0])}]" if len(r) else "—")
            L.append(f"| {NAME[dec]} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## A6. Intake decisions (panel F)", "", "| Decoder | Decision | " + " | ".join(GROUP_TITLE[g] for g in GROUP_KEYS) + " |",
          "|---|---" + "|---" * len(GROUP_KEYS) + "|"]
    for dec in NAME:
        for kind, sens, spec in (("G3 rule", "g3_sens", "g3_spec"), ("Score ≤1 FA/h", "score10_sens", "score10_spec")):
            cells = []
            for g in GROUP_KEYS:
                a = st[(st.group == g) & (st.decoder == dec) & (st.metric == sens)]
                b = st[(st.group == g) & (st.decoder == dec) & (st.metric == spec)]
                cells.append(f"{f(a['mean'].iat[0])} / {f(b['mean'].iat[0])}" if len(a) and not np.isnan(a["mean"].iat[0]) else "—")
            if any(c != "—" for c in cells):
                L.append(f"| {NAME[dec]} | {kind} | " + " | ".join(cells) + " |")
    L += ["", "Cells: sensitivity / specificity (%).", ""]

    b = pd.read_csv(D.INDEX / "d2_bins.csv")
    L += ["## A2. Gain over no decoding versus classifier quality (panel S)", "", "Seg F1@25 gain (points) per bin of validation "
          "window macro F1; details and figure in reports/D2/REPORT.md.", "",
          "| Regime | Val window macro F1 | reps | none | Mode | HMM | HSMM |", "|---|---|---|---|---|---|---|"]
    for (exp, bn), g in b.groupby(["exp", "bin"], sort=False):
        g = g.set_index("decoder")
        L.append(f"| {exp} | {bn} | {g.at['none', 'replicates']} | {f(g.at['none', 'seg_f1_25'])} | "
                 + " | ".join(f"{100 * g.at[d, 'gain_seg_f1_25']:+.1f}" for d in ("mode", "hmm", "hsmm")) + " |")
    L.append("")

    sess = pd.concat(pd.read_csv(x) for d in ("d1", "d1b") for x in sorted((D.INDEX / d).glob("*.csv.gz")))
    sess = sess[sess.set == D.TEST]
    sess = sess[sess.decoder.isin(NAME) | sess.decoder.str.startswith("mode_")]
    mode = D1R.choose_mode(pd.concat(pd.read_csv(x) for x in sorted((D.INDEX / "d1").glob("*.csv.gz"))))
    keep = sess.decoder.isin(NAME) | (sess.decoder == sess.config_key.map(mode))
    sess = sess[keep].copy()
    sess.loc[sess.decoder.str.startswith("mode_"), "decoder"] = "mode"
    sess["group"] = sess.experiment + "/" + sess.role
    e = errors(sess)
    e.to_csv(D.INDEX / "d5_errors.csv", index=False)
    fig_errors(e)
    L += ["## A3. Which errors does each decoder fix? (panel F)", "", "Per true segment, pooled over a participant's sessions. "
          "Spurious = predicted segments that overlap no true segment of their class at IoU ≥ 0.10; missed = true segments "
          "with no such prediction; poorly bounded = found at IoU 0.10 but not at 0.50.", "", "![errors](errors.png)", ""]
    for grp in GROUP_KEYS:
        t = e[e.group == grp].set_index("decoder")
        L += [f"### {GROUP_TITLE[grp]}", "", "| Decoder | " + " | ".join(ERR.values()) + " |", "|---" * (len(ERR) + 1) + "|"]
        for dec in [d for d in NAME if d in t.index]:
            L.append(f"| {NAME[dec]} | " + " | ".join(
                (f"{t.at[dec, m]:.2f}" if m == "over_seg" else f(t.at[dec, m])) for m in ERR) + " |")
        L.append("")
    L += ["Spurious, missed and poorly bounded are in % of true segments; Predicted / true segments is a ratio.", ""]

    t = tuning()
    if len(t):
        t.to_csv(D.INDEX / "d5_tuning.csv", index=False)
        fig_tuning(t)
        L += ["## A4. How much validation data does each decoder need? (panel F)", "",
              "Each replicate keeps 25, 50 or 100 % of its validation streams (nested subsets); every parameter, including "
              "the duration models, is learned or tuned on the subset (a4_tuning_data.py). HMM/HSMM/mode use the D2 reduced "
              "grid. Seg F1@25 [95 % CI over participants].", "", "![tuning data](tuning_data.png)", ""]
        for grp in GROUP_KEYS:
            g = t[t.group == grp]
            if g.empty:
                continue
            fr = sorted(g.frac.unique())
            nval = ", ".join(f"{100 * x:.0f} % = {g[g.frac == x].n_val.mean():.0f}" for x in fr)
            L += [f"### {GROUP_TITLE[grp]} (validation streams: {nval})", "",
                  "| Decoder | " + " | ".join(f"{100 * x:.0f} %" for x in fr) + " | 25 % − 100 % |", "|---" * (len(fr) + 2) + "|"]
            for dec in [d for d in NAME if d in set(g.decoder)]:
                h = g[g.decoder == dec].set_index("frac")
                L.append(f"| {NAME[dec]} | " + " | ".join(f"{f(h.at[x, 'seg_f1_25'])} [{f(h.at[x, 'seg_f1_25_lo'])}, {f(h.at[x, 'seg_f1_25_hi'])}]"
                                                         for x in fr)
                         + f" | {100 * (h.at[fr[0], 'seg_f1_25'] - h.at[fr[-1], 'seg_f1_25']):+.1f} |")
            L.append("")
    else:
        L += ["## A4. Tuning-data fraction", "", "Not available yet (a4_tuning_data.py has not finished).", ""]

    s3 = pd.read_csv(D.INDEX / "d3_participant_stats.csv")
    s3 = s3[s3.metric == "seg_f1_25"]
    L += ["## A5. Online decoding (panel F)", "", "Seg F1@25 by lag; full tables in reports/D3/REPORT.md. Smallest lag within "
          "1 point of the decoder's own offline result, and the best decoder at each lag.", "",
          "| Group | " + " | ".join(NAME[d] for d in ["mode", "hmm", "hsmm", "crf", "crf_fixs"]) + " | Best at 0 / 1 / 2 / 5 s |",
          "|---" * 7 + "|"]
    for grp in s3.group.unique():
        g = s3[s3.group == grp].set_index("decoder")
        cells = []
        for d in ["mode", "hmm", "hsmm", "crf", "crf_fixs"]:
            off = g.at[f"{d}@{'inf_relabel' if d == 'hsmm' else 'inf'}", "mean"]
            ok = [l for l in ["0", "1", "2", "5", "10"] if g.at[f"{d}@{l}", "mean"] >= off - 0.01]
            cells.append(f"{ok[0]} s" if ok else "> 10 s")
        best = []
        for l in ["0", "1", "2", "5"]:
            c = {d: g.at[f"{d}@{l}", "mean"] for d in ["mode", "hmm", "hsmm", "crf", "crfdur", "crf_fixs"]}
            k = max(c, key=c.get)
            best.append(f"{NAME[k]} {100 * c[k]:.1f}")
        L.append(f"| {grp} | " + " | ".join(cells) + " | " + " / ".join(best) + " |")
    L.append("")
    read = sorted({l.rsplit("/", 1)[-1] for l in D.READ_LOG.read_text().splitlines() if l})
    L += ["## Data check", "", f"Sets read by any stage: {', '.join(read)} (allowlist: {', '.join(D.ALLOWED)})."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
