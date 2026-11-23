"""Stage D2 report (analysis A2): decoder gain versus classifier quality on panel S.

    python -m decoders.d2_report

Inputs: index/d2/test/*.csv.gz and *.json (d2_run.py), index/d2/selection.json, the classifier's
validation window macro F1 from outputs/thesis/index/runs.csv (validation only, so the x axis
never uses a test set), and index/d1 for the panel-F check of the reduced grid.

Per replicate, every metric is computed per test participant from that participant's sessions
and averaged over the 10 participants with equal weight (the D1 convention). Gains are decoder
minus none, per replicate.

Outputs: index/d2_replicates.csv (one row per replicate x decoder), index/d2_bins.csv,
reports/D2/REPORT.md and reports/D2/gain_vs_quality.png.

Uncertainty per bin of classifier quality: 95 % intervals by a bootstrap over participants
(per-participant gains are averaged over the replicates of the bin first; sessions are not
resampled within participants, unlike D1), and the spread over configurations.
"""
from __future__ import annotations

import json
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from decoders import dcommon as D

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = D.REPORTS / "D2"
DEC = ["none", "mode", "hmm", "hsmm"]
NAME = {"none": "None", "mode": "Mode filter", "hmm": "HMM", "hsmm": "HSMM"}
COLOR = {"none": "#6b7280", "mode": "#1baf7a", "hmm": "#2a78d6", "hsmm": "#eb6834"}
BINS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
METRICS = ["seg_f1_25", "order_exact", "g3_sens", "g3_spec", "over_seg"]
MNAME = {"seg_f1_25": "Seg F1@25", "order_exact": "Order Exact Match", "g3_sens": "G3-rule sens",
         "g3_spec": "G3-rule spec", "over_seg": "Segments per true segment"}


def per_participant(f) -> pd.DataFrame:
    """Per-participant metrics of one replicate's test sessions."""
    s = pd.read_csv(f)
    rows = []
    for (dec, subj), g in s.groupby(["decoder", "subject"]):
        tp, fp, fn = g.tp25.sum(), g.fp25.sum(), g.fn25.sum()
        pos, neg = g[g.intake == 1], g[g.intake == 0]
        r = {"run_id": g.run_id.iat[0], "decoder": dec, "subject": subj,
             "seg_f1_25": 2 * tp / max(1, 2 * tp + fp + fn), "order_exact": g.exact.mean(),
             "g3_sens": pos.pred_g3.mean() if len(pos) else np.nan,
             "g3_spec": 1 - neg.pred_g3.mean() if len(neg) else np.nan,
             "over_seg": g.n_pred_seg.sum() / max(1, g.n_true_seg.sum())}
        if dec == "hmm":
            r["score_sens"] = pos.pred_score_fa10.mean() if len(pos) else np.nan
            r["score_spec"] = 1 - neg.pred_score_fa10.mean() if len(neg) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def load():
    """Per-participant metrics and external rates of every replicate."""
    files = sorted((D.INDEX / "d2" / "test").glob("*.csv.gz"))
    with Pool(12) as p:
        pp = pd.concat(p.map(per_participant, files, chunksize=20))
    ext = []
    for f in sorted((D.INDEX / "d2" / "test").glob("*.json")):
        j = json.loads(f.read_text())
        for dec in DEC:
            ext.append({"run_id": j["run_id"], "decoder": dec, "g3_rule_per_h": j["test_external"][dec]["g3_rule_per_h"],
                        **({"score_fa10_per_h": j["test_external"]["hmm"]["score_fa10_per_h"]} if dec == "hmm" else {}),
                        **({f"sel_{k}": v for k, v in j["points"][dec].items() if k in ("st", "md", "t", "w")})})
    return pp, pd.DataFrame(ext)


def boot_ci(x: np.ndarray, rng, n=10000):
    """95 % bootstrap interval of a mean."""
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    """Write the D2 tables, figure and report."""
    OUT.mkdir(parents=True, exist_ok=True)
    pp, ext = load()
    runs = pd.read_csv(D.C.INDEX / "runs.csv", usecols=["run_id", "experiment", "config_key", "model", "window_s",
                                                                 "val.window.macro_f1"])
    runs["exp"] = runs.experiment.str[:2]
    rep = pp.groupby(["run_id", "decoder"]).mean(numeric_only=True).reset_index()
    rep = rep.merge(ext, on=["run_id", "decoder"], how="left").merge(runs, on="run_id", how="left")
    none = rep[rep.decoder == "none"].set_index("run_id")
    for m in METRICS + ["g3_rule_per_h"]:
        rep[f"gain_{m}"] = rep[m] - rep.run_id.map(none[m])
    rep["q"] = rep["val.window.macro_f1"]
    rep["bin"] = pd.cut(rep.q, BINS, include_lowest=True)
    rep.to_csv(D.INDEX / "d2_replicates.csv", index=False)
    missing = rep.q.isna().sum()

    pp = pp.merge(runs[["run_id", "exp", "val.window.macro_f1", "config_key"]], on="run_id")
    pp["bin"] = pd.cut(pp["val.window.macro_f1"], BINS, include_lowest=True)
    base = pp[pp.decoder == "none"].set_index(["run_id", "subject"])
    for m in METRICS:
        pp[f"gain_{m}"] = pp[m].to_numpy() - base[m].reindex(pd.MultiIndex.from_frame(pp[["run_id", "subject"]])).to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for (exp, b, dec), g in pp.groupby(["exp", "bin", "decoder"], observed=True):
        r = {"exp": exp, "bin": str(b), "decoder": dec, "configs": g.run_id.str.rsplit("/", n=1).str[0].nunique(),
             "replicates": g.run_id.nunique(), "q_mean": g.drop_duplicates("run_id")["val.window.macro_f1"].mean()}
        for m in METRICS:
            subj = g.groupby("subject")[m].mean().to_numpy()
            gs = g.groupby("subject")[f"gain_{m}"].mean().to_numpy()
            r[m] = np.nanmean(subj)
            r[f"gain_{m}"] = np.nanmean(gs)
            r[f"gain_{m}_lo"], r[f"gain_{m}_hi"] = boot_ci(gs, rng)
        rr = rep[(rep.exp == exp) & (rep.bin == b) & (rep.decoder == dec)]
        r["hurts_seg_frac"] = float((rr.gain_seg_f1_25 < 0).mean())
        r["g3_rule_per_h_median"] = float(rr.g3_rule_per_h.median())
        rows.append(r)
    bins = pd.DataFrame(rows)
    bins.to_csv(D.INDEX / "d2_bins.csv", index=False)

    d1 = pd.concat(pd.read_csv(f) for f in sorted((D.INDEX / "d1").glob("*.csv.gz")))
    d1 = d1[(d1.set == D.TEST) & (d1.decoder.isin(["hmm", "hsmm", "none"]))]
    chk = []
    for (rid, dec), g in d1.groupby(["run_id", "decoder"]):
        f1 = np.mean([2 * x.tp25.sum() / max(1, 2 * x.tp25.sum() + x.fp25.sum() + x.fn25.sum()) for _, x in g.groupby("subject")])
        r2 = rep[(rep.run_id == rid) & (rep.decoder == dec)]
        if len(r2):
            chk.append({"run_id": rid, "decoder": dec, "d1": f1, "d2": float(r2.seg_f1_25.iat[0])})
    chk = pd.DataFrame(chk)

    figure(rep, bins)
    write_report(rep, bins, chk, missing)


def figure(rep, bins):
    """Decoder gain vs. classifier quality figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, exp in enumerate(["E1", "E2"]):
        for row, m in enumerate(["seg_f1_25", "order_exact"]):
            ax = axes[row, col]
            for dec in ["mode", "hmm", "hsmm"]:
                r = rep[(rep.exp == exp) & (rep.decoder == dec)]
                ax.scatter(r.q, 100 * r[f"gain_{m}"], s=3, alpha=0.12, color=COLOR[dec], linewidths=0)
                b = bins[(bins.exp == exp) & (bins.decoder == dec)].copy()
                b = b[b.replicates >= 20]
                ax.plot(b.q_mean, 100 * b[f"gain_{m}"], color=COLOR[dec], lw=2, marker="o", ms=4, label=NAME[dec])
                ax.fill_between(b.q_mean, 100 * b[f"gain_{m}_lo"], 100 * b[f"gain_{m}_hi"], color=COLOR[dec], alpha=0.15, lw=0)
            ax.axhline(0, color="#999", lw=1)
            ax.set_title(f"{exp}: gain in {MNAME[m]} over no decoding", fontsize=10)
            ax.grid(alpha=0.2)
            if row == 1:
                ax.set_xlabel("Classifier window macro F1 (validation)")
            if col == 0:
                ax.set_ylabel("Gain (points)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "gain_vs_quality.png", dpi=150)


def pct(x):
    """Signed percentage text."""
    return "—" if pd.isna(x) else f"{100 * x:+.1f}"


def write_report(rep, bins, chk, missing):
    """Write reports/D2/REPORT.md."""
    L = [f"# Stage D2: decoder gain versus classifier quality (panel S)", "",
         f"Generated {D.now()}. Test set: `{D.TEST}` only. {rep.run_id.nunique()} replicates, "
         f"{rep.run_id.str.rsplit('/', n=1).str[0].nunique()} configurations ({(rep[rep.decoder == 'none'].exp == 'E1').sum()} E1, "
         f"{(rep[rep.decoder == 'none'].exp == 'E2').sum()} E2). Replicates without a validation window macro F1: {missing // 4}.", "",
         "Decoders: none, mode, HMM, HSMM on a reduced grid (d2_run.py), selected per configuration on validation. "
         "x axis: the classifier's window macro F1 on the validation streams (`outputs/thesis/index/runs.csv`). "
         "Per replicate, metrics are participant means (10 test participants, equal weight). "
         "Bin intervals: 95 % bootstrap over participants of the per-participant gain averaged over the bin's replicates "
         "(sessions are not resampled within participants).", "",
         "![gain vs quality](gain_vs_quality.png)", ""]
    for exp in ["E1", "E2"]:
        L += [f"## {exp}: gain over no decoding per bin of classifier quality", ""]
        for m in ["seg_f1_25", "order_exact", "g3_spec"]:
            L += [f"### {MNAME[m]}", "", "| Val window macro F1 | configs | reps | none (level) | mode | HMM | HSMM | HMM hurts (% reps) |",
                  "|---|---|---|---|---|---|---|---|"]
            for b in [str(x) for x in pd.cut(pd.Series([0.0]), BINS, include_lowest=True).cat.categories]:
                g = bins[(bins.exp == exp) & (bins.bin == b)].set_index("decoder")
                if g.empty:
                    continue
                cell = lambda d: f"{pct(g.at[d, f'gain_{m}'])} [{pct(g.at[d, f'gain_{m}_lo'])}, {pct(g.at[d, f'gain_{m}_hi'])}]"
                hurt = 100 * (rep[(rep.exp == exp) & (rep.bin.astype(str) == b) & (rep.decoder == "hmm")][f"gain_{m}"] < 0).mean()
                L.append(f"| {b} | {g.at['none', 'configs']} | {g.at['none', 'replicates']} | {100 * g.at['none', m]:.1f} | "
                         f"{cell('mode')} | {cell('hmm')} | {cell('hsmm')} | {hurt:.0f} |")
            L.append("")
        L += ["Best decoder per bin (participant-mean seg F1@25 level): " + "; ".join(
            f"{b}: {NAME[g.loc[g.seg_f1_25.idxmax(), 'decoder']]}" for b, g in bins[bins.exp == exp].groupby("bin", observed=True)), ""]
        r = rep[rep.exp == exp]
        L += [f"Spearman correlation of the gain in seg F1@25 with classifier quality: " + ", ".join(
            f"{NAME[d]} {r[r.decoder == d][['q', 'gain_seg_f1_25']].corr('spearman').iat[0, 1]:+.2f}" for d in ["mode", "hmm", "hsmm"]), ""]
        L += [f"False G3-rule events per hour on test_external (median over replicates): " + ", ".join(
            f"{NAME[d]} {r[r.decoder == d].g3_rule_per_h.median():.1f}" for d in DEC)
              + f"; HMM score at ≤ 1 FA/h: {r[r.decoder == 'hmm'].score_fa10_per_h.median():.2f}", ""]
        h = r[r.decoder == "hmm"]
        L += [f"HMM score (≤ 1 FA/h on calibration), mean over replicates: sens {100 * h.score_sens.mean():.1f}, spec {100 * h.score_spec.mean():.1f}; "
              f"G3 rule: sens {100 * h.g3_sens.mean():.1f}, spec {100 * h.g3_spec.mean():.1f}.", ""]
    sel = json.loads((D.INDEX / "d2" / "selection.json").read_text())["configs"]
    s = pd.DataFrame([{"config": c, "exp": c[:2], "decoder": d, **v["params"]} for c, ds in sel.items() for d, v in ds.items()])
    L += ["## Selected grid points (per configuration, on validation)", ""]
    for d in ["hmm", "hsmm", "mode"]:
        g = s[s.decoder == d]
        parts = [f"{k}: " + ", ".join(f"{v:g} {100 * n / len(g):.0f} %" for v, n in g[k].value_counts().sort_index().items())
                 for k in ("st", "md", "t", "w") if k in g and g[k].notna().any()]
        L.append(f"- {NAME[d]}: " + "; ".join(parts))
    L += ["", "Grid-edge choices (temperature 4 or mode width 6 s) mean the optimum may lie outside the reduced grid.", ""]
    if len(chk):
        L += ["## Check: reduced grid (D2) versus full grid (D1) on the panel-F replicates", "",
              "| Decoder | replicates | D1 seg F1@25 | D2 seg F1@25 | mean D2 − D1 | max abs |", "|---|---|---|---|---|---|"]
        for d, g in chk.groupby("decoder"):
            L.append(f"| {NAME[d]} | {len(g)} | {100 * g.d1.mean():.1f} | {100 * g.d2.mean():.1f} | {100 * (g.d2 - g.d1).mean():+.2f} | {100 * (g.d2 - g.d1).abs().max():.2f} |")
        L += ["", "`none` must agree exactly (same rule, same data); HMM/HSMM differ by the grid reduction "
              "(D2 decodes only the configuration's own overlap).", ""]
    read = sorted({l.rsplit("/", 1)[-1] for l in D.READ_LOG.read_text().splitlines() if l})
    L += ["## Data check", "", f"Sets read by any stage so far: {', '.join(read)} (allowlist: {', '.join(D.ALLOWED)})."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
