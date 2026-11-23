"""Reports: tables (CSV + Markdown) and figures per experiment, from the indexes only.

    python -m thesis.report

Reads index/runs.csv, index/postproc.csv and index/selection/*.json; writes
reports/<experiment_slug>/{REPORT.md, tables/*.csv, figures/*.png}. Values are mean ± std over
the replicates of a configuration (3 folds × 1 seed in screening stages, 3 folds × 3 seeds in B
and F). "val" columns are the only ones used for selection; every other set is blind.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis import common as C

FACTORS = ["fs_hz", "window_s", "overlap", "signals", "filter"]
RUN_COLS = {
    "val.timeline.seg_f1_25": "val segF1@25",
    "val.window.macro_f1": "val win F1",
    "test_sequences.window.macro_f1": "seq win F1",
    "test_sequences.window.accuracy": "seq win acc",
    "test_sequences.window.recall.G3": "seq G3 recall",
    "test_sequences.timeline.seg_f1_25": "seq segF1@25",
    "test_sequences.order.exact_match": "seq Order EM",
    "test_wild.window.macro_f1": "wild win F1",
    "test_wild.timeline.seg_f1_25": "wild segF1@25",
    "test_external.external.gesture_events_per_hour": "ext gestures/h",
}
POST_COLS = {
    "val.timeline.seg_f1_25": "val segF1@25",
    "test_sequences.window.macro_f1": "seq win F1",
    "test_sequences.window.accuracy": "seq win acc",
    "test_sequences.timeline.seg_f1_25": "seq segF1@25",
    "test_sequences.order.exact_match": "seq Order EM",
    "test_wild.timeline.seg_f1_25": "wild segF1@25",
    "test_wild.order.exact_match": "wild Order EM",
    "test_wild.intake_g3.balanced_accuracy": "wild intake bacc (G3)",
    "test_wild.intake_g2_g3.balanced_accuracy": "wild intake bacc (G2→G3)",
    "test_wild.intake_llr_fa10.balanced_accuracy": "wild intake bacc (LLR ≤1/h)",
    "test_wild.intake_llr_fa10.sensitivity": "wild intake sens (LLR ≤1/h)",
    "test_external.external.g3_events_per_hour": "ext G3 events/h",
    "test_external.external.llr_fa10_events_per_hour": "ext LLR FA/h",
}


def sel(name):
    """A selection file of index/selection, or None."""
    p = C.SELECTION / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def agg(df, by, cols, n=True):
    """Mean and std of the columns per group, with the group size."""
    cols = [c for c in cols if c in df]
    g = df.groupby(by)[cols]
    m, s = g.mean(), g.std()
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = m[c]
        out[c + " std"] = s[c]
    if n:
        out["n"] = g.size()
    return out


def md_table(df, labels, pct=True) -> str:
    """mean ± std cells; percentages except per-hour rates."""
    cols = [c for c in labels if c in df]
    head = [df.index.name or ""] + [labels[c] for c in cols] + (["n"] if "n" in df else [])
    lines = ["| " + " | ".join(map(str, head)) + " |", "|" + "---|" * len(head)]
    for idx, r in df.iterrows():
        cells = [str(idx) if not isinstance(idx, tuple) else " / ".join(map(str, idx))]
        for c in cols:
            v, s = r[c], r.get(c + " std", np.nan)
            if pd.isna(v):
                cells.append("—")
            elif "per_hour" in c:
                cells.append(f"{v:.1f}" + (f" ± {s:.1f}" if pd.notna(s) else ""))
            else:
                cells.append(f"{100 * v:.1f}" + (f" ± {100 * s:.1f}" if pd.notna(s) else ""))
        if "n" in df:
            cells.append(str(int(r["n"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_tables(d, name, df):
    """Write a table as tables/<name>.csv."""
    (d / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "tables" / f"{name}.csv")


def report_e12(exp, runs):
    """REPORT.md, tables and figures of E1 or E2."""
    d = C.REPORTS / C.SLUGS[exp]
    r = runs[(runs.experiment == exp) & (runs.status == "done")]
    md = [f"# {exp} — {C.EXPERIMENTS[exp]['name']}", "",
          "Mean ± std over replicates, in %, except events per hour. Selection uses `val` only "
          "(mean val.timeline.seg_f1_25); every other column is blind.", ""]
    a = sel(f"{exp}_A")
    A = r[r.config_key.isin([x["config_key"] for x in a["ranking"]])] if a else r.iloc[0:0]
    A = A[A.replicate.str.endswith("_s0")]
    md += ["## Stage A — signal configuration (CNN-BiLSTM, default hyperparameters, seed 0)", ""]
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(3.2 * len(FACTORS), 3), sharey=True)
    for ax, f in zip(axes, FACTORS):
        t = agg(A, f, list(RUN_COLS))
        t.index.name = f
        write_tables(d, f"A_factor_{f}", t)
        md += [f"### {f}", "", md_table(t, RUN_COLS), ""]
        for col, lab in (("val.timeline.seg_f1_25", "val"), ("test_sequences.timeline.seg_f1_25", "blind seq")):
            ax.plot(range(len(t)), 100 * t[col], "o-", label=lab)
        ax.set_xticks(range(len(t)), [str(i) for i in t.index])
        ax.set_title(f)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("segmental F1@25 (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    (d / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "figures" / "A_factors.png", dpi=150)
    plt.close(fig)
    md += ["![Stage A factors](figures/A_factors.png)", ""]
    if a:
        top = agg(A[A.config_key.isin(a["advanced"] + [x["config_key"] for x in a["ranking"][:10]])],
                  "config_key", list(RUN_COLS)).sort_values("val.timeline.seg_f1_25", ascending=False)
        write_tables(d, "A_top10", top)
        md += ["### Top 10 signal configurations", "", md_table(top, RUN_COLS), ""]
    for stage, title in (("B", "Stage B — model (3 folds × 3 seeds)"), ("C", "Stage C — hyperparameters (seed 0), top 10"),
                         ("F", "Stage F — finalists (3 folds × 3 seeds)")):
        s = sel(f"{exp}_{stage}")
        if not s:
            continue
        keys = [x["config_key"] for x in s["ranking"]][: (10 if stage == "C" else None)]
        sub = r[r.config_key.isin(keys)]
        if stage == "C":
            sub = sub[sub.replicate.str.endswith("_s0")]
        t = agg(sub, "config_key", list(RUN_COLS)).loc[[k for k in keys if k in set(sub.config_key)]]
        write_tables(d, f"{stage}", t)
        md += [f"## {title}", "", f"Advanced: {', '.join(f'`{k}`' for k in s['advanced'])}", "", md_table(t, RUN_COLS), ""]
    if stage_b := sel(f"{exp}_B"):
        sub = r[r.config_key.isin([x["config_key"] for x in stage_b["ranking"]])]
        t = agg(sub, "model", list(RUN_COLS))
        t.index.name = "model"
        write_tables(d, "B_by_model", t)
        md += ["### Stage B by model (all 5 signals)", "", md_table(t, RUN_COLS), ""]
    (d / "REPORT.md").write_text("\n".join(md) + "\n")
    return d


def report_windows(runs):
    """Short-window tracks: every stage decision and the final configuration of each track."""
    d = C.REPORTS / "short_window_tracks"
    md = ["# Short-window tracks (0.75 s and 1 s)", "",
          "The best stage-A signal configuration at each window, taken through B → C → F like the main "
          "finalists (src/extra_windows.py). Mean ± std in %, blind except `val`.", ""]
    for exp in ("E1", "E2"):
        r = runs[(runs.experiment == exp) & (runs.status == "done")]
        for tag in ("w075", "w100"):
            md += [f"## {exp} {tag}", ""]
            for stage in "ABCF":
                s = sel(f"{exp}_{tag}_{stage}")
                if not s:
                    continue
                keys = [x["config_key"] for x in s["ranking"]][:5]
                sub = r[r.config_key.isin(keys)]
                if stage in "AC":
                    sub = sub[sub.replicate.str.endswith("_s0")]
                t = agg(sub, "config_key", list(RUN_COLS)).loc[[k for k in keys if k in set(sub.config_key)]]
                write_tables(d, f"{exp}_{tag}_{stage}", t)
                md += [f"### Stage {stage} (top 5) — advanced: {', '.join(f'`{k}`' for k in s['advanced'])}", "",
                       md_table(t, RUN_COLS), ""]
    (d / "REPORT.md").write_text("\n".join(md) + "\n")
    return d


def report_e3():
    """REPORT.md, tables and figures of E3."""
    d = C.REPORTS / C.SLUGS["E3"]
    pp = pd.read_csv(C.INDEX / "postproc.csv")
    e3 = sel("E3")
    md = ["# E3 — Temporal post-processing and intake decision", "",
          "Each configuration with the grid point selected on validation for every decoder "
          "(index/selection/E3_grid.json). `none` = E1/E2 timeline rule. Mean ± std over 3 folds × 3 seeds, in %, "
          "except events per hour. LLR thresholds: ≤ 1 false event/h on the external calibration participants.", ""]
    if e3:
        md += ["**Selected on validation:** " + "; ".join(
            f"{k}: `{v['config_key']}` + `{v['decoder_key']}`" for k, v in e3["per_experiment"].items()), ""]
    for role, title in (("finalist", "Stage-F finalists"), ("window_ref", "Short-window references (0.75 s, 1 s)")):
        sub = pp[pp.role == role]
        if not len(sub):
            continue
        t = agg(sub, ["experiment", "config_key", "decoder"], list(POST_COLS))
        write_tables(d, f"E3_{role}", t)
        md += [f"## {title}", "", md_table(t, POST_COLS), ""]
    t = agg(pp, ["experiment", "role", "decoder"], list(POST_COLS))
    write_tables(d, "E3_by_decoder", t)
    md += ["## By decoder (mean over configurations)", "", md_table(t, POST_COLS), ""]
    if e3:
        best = [(k.split("/")[0], k.split("/")[1], v["decoder_key"]) for k, v in e3["per_config"].items()]
        rows = pd.concat([pp[(pp.experiment == e) & (pp.config_key == c) & (pp.decoder_key == dk)] for e, c, dk in best])
        t = agg(rows, ["experiment", "window_s", "config_key"], list(POST_COLS))
        write_tables(d, "E3_by_window", t)
        md += ["## Window length after decoding (best decoder of each configuration)", "", md_table(t, POST_COLS), ""]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for exp, mk in (("E1", "o"), ("E2", "s")):
            x = t.loc[exp] if exp in t.index.get_level_values(0) else None
            if x is None:
                continue
            x = x.groupby(level=0).mean()
            for i, (col, lab) in enumerate((("test_sequences.timeline.seg_f1_25", "blind seq segF1@25"),
                                            ("test_wild.timeline.seg_f1_25", "wild segF1@25"))):
                ax[i].plot(x.index, 100 * x[col], mk + "-", label=exp)
                ax[i].set_title(lab)
                ax[i].set_xlabel("window (s)")
                ax[i].grid(alpha=0.3)
        ax[0].set_ylabel("%")
        ax[0].legend()
        fig.tight_layout()
        (d / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "figures" / "E3_window.png", dpi=150)
        plt.close(fig)
        md += ["![Window length after decoding](figures/E3_window.png)", ""]
    reg = C.MODELS / "registry.csv"
    if reg.exists():
        md += ["## Exported models (models/registry.csv)", "", pd.read_csv(reg).to_markdown(index=False), ""]
    (d / "REPORT.md").write_text("\n".join(md) + "\n")
    return d


def main():
    """Write every report."""
    runs = pd.read_csv(C.INDEX / "runs.csv")
    for exp in ("E1", "E2"):
        print("report:", report_e12(exp, runs))
    print("report:", report_windows(runs))
    if (C.INDEX / "postproc.csv").exists():
        print("report:", report_e3())


if __name__ == "__main__":
    main()
