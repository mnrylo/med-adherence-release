"""Results chapter material: E1 → E2 → comparison, each at window, sequence and intake level.

    python -m thesis.results_chapter

Reads the indexes (runs.csv, postproc.csv, selection/*.json) plus the per-run metrics.json files
(confusion matrices) and the decoded 10 Hz paths of E3. Writes
reports/results_chapter/{RESULTS_DATA.md, tables/*.csv, figures/*.png}.

Replicates: stage A = 3 folds × seed 0 per configuration; finalists and short-window references =
3 folds × 3 seeds. Values are mean ± std over replicates, in %, except events per hour.
"""
from __future__ import annotations

import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from thesis import common as C

OUT = C.REPORTS / "results_chapter"
CLASSES = ["G1", "G2", "G3", "G4", "G5"]
CLASS_NAMES = {"G1": "G1 open", "G2": "G2 pick", "G3": "G3 to mouth", "G4": "G4 drink", "G5": "G5 rest/other"}
SETS = {"val": "validation", "test_sequences": "blind sequences", "test_wild": "sequences in daily activity"}
FACTORS = {"window_s": "window (s)", "overlap": "overlap", "filter": "filter", "fs_hz": "fs (Hz)", "signals": "signals"}
DECODERS = ["none", "hmm", "hsmm"]
DEC_LABEL = {"none": "no decoder", "hmm": "HMM", "hsmm": "HSMM"}
MD = []


def sel(name):
    """A selection file of index/selection."""
    return json.loads((C.SELECTION / f"{name}.json").read_text())


def pct(v, s=None, per_hour=False):
    """Percentage (or per-hour rate) with optional std as text."""
    if pd.isna(v):
        return "—"
    if per_hour:
        return f"{v:.1f}" + (f" ± {s:.1f}" if s is not None and pd.notna(s) else "")
    return f"{100 * v:.1f}" + (f" ± {100 * s:.1f}" if s is not None and pd.notna(s) else "")


def fmt(col, v, s=None):
    """Cell text of a metric column."""
    if "edit_distance" in col:
        return "—" if pd.isna(v) else f"{v:.2f}" + (f" ± {s:.2f}" if s is not None and pd.notna(s) else "")
    return pct(v, s, "per_hour" in col)


def ms_table(df, by, cols, labels=None, index_label=None):
    """mean ± std table (Markdown) and the numeric frame (CSV)."""
    g = df.groupby(by, sort=False)[cols]
    m, s = g.mean(), g.std()
    labels = labels or {c: c for c in cols}
    head = [index_label or (by if isinstance(by, str) else " / ".join(by))] + [labels[c] for c in cols] + ["n"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for idx in m.index:
        name = " / ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        lines.append("| " + " | ".join([name] + [fmt(c, m.loc[idx, c], s.loc[idx, c]) for c in cols]
                                       + [str(g.size().loc[idx])]) + " |")
    num = pd.concat({"mean": m, "std": s}, axis=1)
    num["n"] = g.size()
    return "\n".join(lines), num


def save(name, md_text, num=None, title=None, note=None):
    """Write a numeric table and append its Markdown to RESULTS_DATA.md."""
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    if num is not None:
        num.to_csv(OUT / "tables" / f"{name}.csv")
    MD.extend([f"### {title or name}", ""] + ([note, ""] if note else []) + [md_text, ""])


def savefig(fig, name, title):
    """Save a figure and reference it in RESULTS_DATA.md."""
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    MD.extend([f"![{title}](figures/{name}.png)", ""])


def confusion(paths, part, level="window", root=C.ROOT):
    """Summed confusion matrix of runs on a set and level."""
    cm = None
    for p in paths:
        m = json.loads((root / p / "metrics.json").read_text())
        c = np.array(m[part][level]["confusion"], dtype=float)
        cm = c if cm is None else cm + c
    return cm


def plot_cm(ax, cm, title, rows=None):
    """Row-normalised confusion matrix on an axis."""
    rows = rows if rows is not None else list(range(len(cm)))
    tot = cm.sum(1, keepdims=True)
    norm = np.divide(cm, tot, out=np.zeros_like(cm), where=tot > 0)[rows]
    ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i, j in itertools.product(range(len(rows)), range(5)):
        v = norm[i, j]
        ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else "black")
    ax.set_xticks(range(5), CLASSES, fontsize=7)
    ax.set_yticks(range(len(rows)), [CLASSES[r] for r in rows], fontsize=7)
    ax.set_title(title, fontsize=8)


RUNS = PP = E3 = FINAL = WREF = None


def load_indexes():
    """Read the run and post-processing indexes and the E3 selections into the module globals."""
    global RUNS, PP, E3, FINAL, WREF
    RUNS = pd.read_csv(C.INDEX / "runs.csv")
    RUNS = RUNS[RUNS.status == "done"]
    PP = pd.read_csv(C.INDEX / "postproc.csv")
    E3 = sel("E3")
    FINAL = {e: v["config_key"] for e, v in E3["per_experiment"].items()}
    WREF = sel("E3_windows")


def stage_a(exp):
    """Stage-A runs of an experiment."""
    keys = [x["config_key"] for x in sel(f"{exp}_A")["ranking"]]
    a = RUNS[(RUNS.experiment == exp) & RUNS.config_key.isin(keys) & RUNS.replicate.str.endswith("_s0")].copy()
    a["filter"] = a["filter"].replace({"none": "raw", "hp0.3": "high-pass 0.3 Hz"})
    a["signals"] = a["signals"].replace({"acc": "acc", "acc_gyr": "acc+gyr"})
    a["overlap"] = (100 * a["overlap"]).round().astype(int).astype(str) + " %"
    return a


def seq_configs(exp):
    """(window_s, config_key) for the sequence/intake sections: the short-window references and the finalist."""
    keys = [v["config_key"] for v in WREF["configs"].values() if v["experiment"] == exp]
    sub = PP[(PP.experiment == exp) & (PP.config_key.isin(keys + [FINAL[exp]]))]
    return sub


WIN_COLS = {"val.window.macro_f1": "val macro F1", "test_sequences.window.macro_f1": "seq macro F1",
            "test_wild.window.macro_f1": "wild macro F1", "test_sequences.window.accuracy": "seq accuracy",
            "test_external.external.gesture_events_per_hour": "ext gestures/h",
            "test_external.external.g3_events_per_hour": "ext G3/h"}


def factor_tests(a, col):
    """Friedman (≥ 3 levels) or Wilcoxon signed-rank (2 levels) on config means, blocked by the other factors."""
    cfg = a.groupby(list(FACTORS))[col].mean().reset_index()
    rows = []
    for f in FACTORS:
        others = [o for o in FACTORS if o != f]
        wide = cfg.pivot_table(index=others, columns=f, values=col).dropna()
        lv = list(wide.columns)
        if len(lv) == 2:
            d = wide[lv[1]] - wide[lv[0]]
            p = stats.wilcoxon(d).pvalue
            rows.append({"factor": f, "test": "Wilcoxon", "blocks": len(wide),
                         "effect": f"{lv[1]} − {lv[0]}: {100 * d.mean():+.1f} pts (wins {100 * (d > 0).mean():.0f} %)",
                         "p": p})
        else:
            chi, p = stats.friedmanchisquare(*[wide[c] for c in lv])
            w = chi / (len(wide) * (len(lv) - 1))
            low = "per_hour" in col
            best = wide.rank(axis=1, ascending=low).mean().idxmin()
            rows.append({"factor": f, "test": "Friedman", "blocks": len(wide),
                         "effect": f"Kendall W = {w:.2f}; best mean rank ({'lowest' if low else 'highest'}): {best}", "p": p})
    return pd.DataFrame(rows)


def window_factors(exp, a):
    """Window-level effect of every signal factor in stage A."""
    MD.extend([f"## {exp} — window level: signal factors (stage A, CNN-BiLSTM, default hyperparameters)", "",
               "Marginal means over the other four factors (240 configurations × 3 folds, seed 0). "
               "The tests use one value per configuration (mean over folds), blocked by the other factors.", ""])
    for f, lab in FACTORS.items():
        t, num = ms_table(a.sort_values(f), f, list(WIN_COLS), WIN_COLS, lab)
        save(f"{exp}_win_factor_{f}", t, num, f"{exp} — {lab}")
    for col, lab in (("test_sequences.window.macro_f1", "blind sequences"), ("test_wild.window.macro_f1", "daily activity"),
                     ("test_external.external.g3_events_per_hour", "ext G3 events/h")):
        ft = factor_tests(a, col)
        lines = ["| factor | test | blocks | effect | p |", "|---|---|---|---|---|"]
        for _, r in ft.iterrows():
            eff = r.effect if "per_hour" not in col else r.effect.replace(" pts", "").replace("+", "+")
            if "per_hour" in col and r.test == "Wilcoxon":
                cfg = a.groupby(list(FACTORS))[col].mean().reset_index()
                others = [o for o in FACTORS if o != r.factor]
                wide = cfg.pivot_table(index=others, columns=r.factor, values=col).dropna()
                lv = list(wide.columns)
                d = wide[lv[1]] - wide[lv[0]]
                eff = f"{lv[1]} − {lv[0]}: {d.mean():+.1f} /h (lower in {100 * (d < 0).mean():.0f} %)"
            lines.append(f"| {r.factor} | {r.test} | {r.blocks} | {eff} | {r.p:.1e} |")
        save(f"{exp}_win_tests_{col.split('.')[0]}_{col.split('.')[-1]}", "\n".join(lines), ft,
             f"{exp} — factor tests on {lab} ({col})")
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(13, 2.9), sharey=True)
    for ax, (f, lab) in zip(axes, FACTORS.items()):
        g = a.groupby(f)
        for col, name, mk in (("val.window.macro_f1", "validation", "o"), ("test_sequences.window.macro_f1", "blind sequences", "s"),
                              ("test_wild.window.macro_f1", "daily activity", "^")):
            m = g[col].mean()
            sd = a.groupby(list(FACTORS))[col].mean().groupby(f).std()
            ax.errorbar(range(len(m)), 100 * m, yerr=100 * sd / np.sqrt(a.groupby(list(FACTORS)).ngroups / len(m)),
                        marker=mk, capsize=2, label=name)
        ax.set_xticks(range(len(m)), [str(i) for i in m.index], fontsize=7)
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("window macro F1 (%)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{exp}: marginal effect of each signal factor (mean ± s.e. over configurations)", fontsize=9)
    savefig(fig, f"{exp}_win_factors", f"{exp} factor marginals")
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.9))
    for ax, (f2, col, name) in zip(axes, [("overlap", "test_sequences.window.macro_f1", "blind seq"),
                                          ("overlap", "test_wild.window.macro_f1", "daily activity"),
                                          ("filter", "test_sequences.window.macro_f1", "blind seq"),
                                          ("filter", "test_wild.window.macro_f1", "daily activity")]):
        p = a.pivot_table(index=f2, columns="window_s", values=col, aggfunc="mean")
        ax.imshow(p.values, cmap="viridis", aspect="auto")
        for i, j in itertools.product(range(p.shape[0]), range(p.shape[1])):
            ax.text(j, i, f"{100 * p.values[i, j]:.0f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_xticks(range(p.shape[1]), p.columns, fontsize=7)
        ax.set_yticks(range(p.shape[0]), p.index, fontsize=7)
        ax.set_xlabel("window (s)")
        ax.set_title(f"{f2} × window — {name}", fontsize=8)
    savefig(fig, f"{exp}_win_interactions", f"{exp} interactions")
    MD.extend([f"## {exp} — window level: per-class metrics by window length (blind sequences)", ""])
    for metric in ("precision", "recall", "f1"):
        cols = [f"test_sequences.window.{metric}.{c}" for c in CLASSES]
        if metric == "f1":
            cols.append("test_sequences.window.macro_f1")
        labels = {c: (c.split(".")[-1] if "macro" not in c else "macro F1") for c in cols}
        t, num = ms_table(a.sort_values("window_s"), "window_s", cols, labels, "window (s)")
        save(f"{exp}_win_class_{metric}_by_window", t, num, f"{exp} — {metric} per class by window (blind sequences)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, part in zip(axes, ("test_sequences", "test_wild")):
        rows, names = [], []
        for f, lab in FACTORS.items():
            g = a.groupby(f)[[f"{part}.window.f1.{c}" for c in CLASSES] + [f"{part}.window.macro_f1"]].mean()
            for lv, r in g.iterrows():
                rows.append(r.values)
                names.append(f"{lab}: {lv}")
        h = np.array(rows)
        ax.imshow(h, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        for i, j in itertools.product(range(h.shape[0]), range(h.shape[1])):
            ax.text(j, i, f"{100 * h[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="w" if h[i, j] < 0.6 else "k")
        ax.set_xticks(range(6), CLASSES + ["macro"], fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title(f"window F1 per class — {SETS[part]}", fontsize=8)
    savefig(fig, f"{exp}_win_class_f1_heatmap", f"{exp} per-class F1 by factor level")
    rank = [x["config_key"] for x in sel(f"{exp}_A")["ranking"]]
    best = {}
    for k in rank:
        w = C.parse_key(k)["window_s"]
        best.setdefault(w, k)
    best = dict(sorted(best.items()))
    fig, axes = plt.subplots(2, len(best), figsize=(2.3 * len(best), 4.6))
    rows = []
    for j, (w, k) in enumerate(best.items()):
        sub = a[a.config_key == k]
        for i, part in enumerate(("test_sequences", "test_wild")):
            cm = confusion(sub.path, part)
            plot_cm(axes[i, j], cm, f"{w:g} s" if i == 0 else "")
            if j == 0:
                axes[i, 0].set_ylabel(SETS[part], fontsize=8)
        rows.append(sub.assign(window=f"{w:g} s"))
    fig.suptitle(f"{exp}: window confusion (% of true class), best stage-A configuration per window length "
                 "(3 folds summed)", fontsize=9)
    savefig(fig, f"{exp}_win_confusion_by_window", f"{exp} confusion by window")
    sub = pd.concat(rows)
    cols = ["val.window.macro_f1"] + [f"test_sequences.window.f1.{c}" for c in CLASSES] + ["test_sequences.window.macro_f1",
                                                                                           "test_wild.window.macro_f1"]
    t, num = ms_table(sub, ["window", "config_key"], cols,
                      {c: ("val macro" if c.startswith("val") else ("wild macro" if "wild" in c else
                                                                    ("seq macro" if "macro" in c else "seq F1 " + c[-2:])))
                       for c in cols}, "window / configuration")
    save(f"{exp}_win_best_per_window", t, num, f"{exp} — best stage-A configuration per window length (selected on validation)")


def window_finalist(exp):
    """Window-level metrics of the finalists."""
    key = FINAL[exp]
    sub = RUNS[(RUNS.experiment == exp) & (RUNS.config_key == key)]
    sub = sub[sub.replicate.str.match(r"f\d_s\d")]
    MD.extend([f"## {exp} — window level: selected model `{key}` (3 folds × 3 seeds)", ""])
    lines = ["| stage | advanced (first) | val seg F1@25 (selection score) |", "|---|---|---|"]
    for st in "ABCF":
        s = sel(f"{exp}_{st}")
        top = s["ranking"][0]
        lines.append(f"| {st} | `{top['config_key']}` | {pct(top['score'], top.get('score_std'))} |")
    save(f"{exp}_selection_path", "\n".join(lines), None, f"{exp} — selection path")
    rows = []
    for part in ("val", "test_sequences", "test_wild"):
        for c in CLASSES + ["macro"]:
            for m in ("precision", "recall", "f1"):
                col = f"{part}.window.{m}.{c}" if c != "macro" else (f"{part}.window.macro_f1" if m == "f1" else None)
                if col is None:
                    continue
                rows.append({"set": SETS[part], "class": CLASS_NAMES.get(c, "macro"), "metric": m,
                             "mean": sub[col].mean(), "std": sub[col].std()})
        rows.append({"set": SETS[part], "class": "all", "metric": "accuracy", "mean": sub[f"{part}.window.accuracy"].mean(),
                     "std": sub[f"{part}.window.accuracy"].std()})
    df = pd.DataFrame(rows)
    lines = ["| set | class | precision | recall | F1 |", "|---|---|---|---|---|"]
    for (s, c), g in df[df.metric.isin(["precision", "recall", "f1"])].groupby(["set", "class"], sort=False):
        v = {r.metric: pct(r["mean"], r["std"]) for _, r in g.iterrows()}
        lines.append(f"| {s} | {c} | {v.get('precision', '—')} | {v.get('recall', '—')} | {v.get('f1', '—')} |")
    for _, r in df[df.metric == "accuracy"].iterrows():
        lines.append(f"| {r.set} | accuracy | | | {pct(r['mean'], r['std'])} |")
    save(f"{exp}_win_final_per_class", "\n".join(lines), df, f"{exp} — selected model, window metrics per class")
    ext = sub["test_external.external.gesture_events_per_hour"]
    MD.extend([f"External participants (daily activity only, no protocol gestures): "
               f"{ext.mean():.1f} ± {ext.std():.1f} gesture events/h, "
               f"{sub['test_external.external.g3_events_per_hour'].mean():.1f} ± "
               f"{sub['test_external.external.g3_events_per_hour'].std():.1f} G3 events/h (timeline rule, no decoder).", ""])
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, part in zip(axes, ("val", "test_sequences", "test_wild")):
        plot_cm(ax, confusion(sub.path, part), SETS[part])
    plot_cm(axes[3], confusion(sub.path, "test_external"), "daily activity only (external)", rows=[3, 4])
    fig.suptitle(f"{exp} selected model: window confusion (% of true class, 9 replicates summed)", fontsize=9)
    savefig(fig, f"{exp}_win_final_confusion", f"{exp} selected model confusion")


SEQ_COLS = {"timeline.macro_f1": "frame macro F1", "timeline.seg_f1_10": "seg F1@10", "timeline.seg_f1_25": "seg F1@25",
            "timeline.seg_f1_50": "seg F1@50", "order.exact_match": "Order EM", "order.edit_distance": "Order edit dist."}


def pp_label(df):
    """Post-processing rows with window and decoder labels and a flag for the selected configuration, sorted."""
    df = df.copy()
    df["window"] = df.window_s.map(lambda w: f"{w:g} s")
    df["dec"] = df.decoder.map(DEC_LABEL)
    df["selected"] = df.config_key == df.experiment.map(FINAL)
    return df.sort_values(["window_s", "decoder"], key=lambda s: s.map({d: i for i, d in enumerate(DECODERS)})
                          if s.name == "decoder" else s)


def paired(df, col, a="none", b="hmm"):
    """Mean paired difference b - a over replicates and its Wilcoxon signed-rank p value."""
    x = df[df.decoder == a].set_index("replicate")[col]
    y = df[df.decoder == b].set_index("replicate")[col]
    d = (y - x).dropna()
    p = stats.wilcoxon(d).pvalue if len(d) and (d != 0).any() else 1.0
    return d.mean(), p


def sequence(exp):
    """Sequence-level tables and figures (timeline, order) of an experiment."""
    sub = pp_label(seq_configs(exp))
    MD.extend([f"## {exp} — sequence level: Gesture Timeline and Gesture Order, before and after decoding", "",
               "Configurations: the selected model and the best 0.75 s and 1 s models, each with its decoder "
               "parameters chosen on validation (E3 grid). `no decoder` = averaged 10 Hz timeline + minimum-duration rule.", ""])
    for part in ("val", "test_sequences", "test_wild"):
        cols = [f"{part}.{c}" for c in SEQ_COLS]
        t, num = ms_table(sub, ["window", "dec"], cols, {f"{part}.{c}": v for c, v in SEQ_COLS.items()}, "window / decoder")
        save(f"{exp}_seq_{part}", t, num, f"{exp} — {SETS[part]}")
    lines = ["| window | set | metric | HMM − none | p | HSMM − none | p |", "|---|---|---|---|---|---|---|"]
    rows = []
    for w, g in sub.groupby("window", sort=False):
        for part in ("test_sequences", "test_wild"):
            for c in ("timeline.seg_f1_25", "order.exact_match", "timeline.macro_f1"):
                col = f"{part}.{c}"
                dh, ph = paired(g, col)
                ds, ps = paired(g, col, b="hsmm")
                rows.append({"window": w, "set": part, "metric": c, "hmm_delta": dh, "hmm_p": ph, "hsmm_delta": ds, "hsmm_p": ps})
                lines.append(f"| {w} | {SETS[part]} | {SEQ_COLS[c]} | {100 * dh:+.1f} | {ph:.3f} | {100 * ds:+.1f} | {ps:.3f} |")
    save(f"{exp}_seq_paired", "\n".join(lines), pd.DataFrame(rows),
         f"{exp} — paired differences vs. no decoder (Wilcoxon, 9 replicates; smallest attainable p = 0.004)")
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9), sharey=False)
    wins = list(dict.fromkeys(sub.window))
    for ax, (part, c) in zip(axes, [("test_sequences", "timeline.seg_f1_25"), ("test_wild", "timeline.seg_f1_25"),
                                    ("test_sequences", "order.exact_match"), ("test_wild", "order.exact_match")]):
        for k, d in enumerate(DECODERS):
            g = sub[sub.decoder == d].groupby("window", sort=False)[f"{part}.{c}"]
            m, s = g.mean().reindex(wins), g.std().reindex(wins)
            ax.bar(np.arange(len(wins)) + (k - 1) * 0.26, 100 * m, 0.26, yerr=100 * s, capsize=2, label=DEC_LABEL[d])
        ax.set_xticks(range(len(wins)), wins)
        ax.set_title(f"{SEQ_COLS[c]} — {SETS[part]}", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=7)
    savefig(fig, f"{exp}_seq_decoders", f"{exp} decoders")
    fin = sub[sub.selected]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for j, (part, d) in enumerate([("test_sequences", "none"), ("test_sequences", "hmm"), ("test_wild", "none"), ("test_wild", "hmm")]):
        plot_cm(axes[j], confusion(fin[fin.decoder == d].path, part, "timeline"), f"{SETS[part]} — {DEC_LABEL[d]}")
    fig.suptitle(f"{exp} selected model: 10 Hz Gesture Timeline confusion (% of true frames)", fontsize=9)
    savefig(fig, f"{exp}_seq_timeline_confusion", f"{exp} timeline confusion")


def timeline_example():
    """One session in daily activity: truth, no decoder, HMM — E1 and E2 selected models, fold 0 seed 0."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)
    fig.subplots_adjust(hspace=0.35)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(0), cmap(1), cmap(3), cmap(2), (0.9, 0.9, 0.9)]
    from matplotlib.colors import ListedColormap
    lc = ListedColormap(colors)
    sid = None
    for ax, exp in zip(axes, ("E1", "E2")):
        rows = PP[(PP.experiment == exp) & (PP.config_key == FINAL[exp]) & (PP.replicate == "f0_s0")]
        cfg_dir = C.RUNS / C.SLUGS[exp] / FINAL[exp]
        frames = np.load(cfg_dir / "artifacts" / "frames.npz")
        paths = {d: np.load(C.ROOT / rows[rows.decoder == d].path.iloc[0] / "paths.npz") for d in ("none", "hmm")}
        if sid is None:
            sess = pd.read_csv(C.ROOT / rows[rows.decoder == "hmm"].path.iloc[0] / "sessions.csv")
            sess = sess[(sess.set == "test_wild") & (sess.exact == 1)]
            sid = sess.sid.iloc[len(sess) // 3]
        y = frames[f"y10/{sid}"]
        strips = [y, paths["none"][f"test_wild/{sid}"], paths["hmm"][f"test_wild/{sid}"]]
        n = min(len(s) for s in strips)
        ax.imshow(np.vstack([s[:n] for s in strips]), aspect="auto", cmap=lc, vmin=0, vmax=4, interpolation="nearest",
                  extent=[0, n / 10, 2.5, -0.5])
        ax.set_yticks([0, 1, 2], ["annotation", "no decoder", "HMM"], fontsize=7)
        ax.set_title(f"{exp} selected model", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, [CLASS_NAMES[c] for c in CLASSES], ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Gesture Timeline of one blind session embedded in daily activity ({sid})", fontsize=9)
    MD.extend(["## Example Gesture Timeline", ""])
    savefig(fig, "timeline_example", "timeline example")


def intake(exp):
    """Intake tables and figures of an experiment."""
    sub = pp_label(seq_configs(exp))
    MD.extend([f"## {exp} — intake decision per session", "",
               "Sensitivity = sessions with an annotated intake detected; specificity = sessions without intake "
               "left undetected. False alarms per hour: intake events on the external test participants (daily "
               "activity only, never trained on). Rules on the decoded Gesture Timeline: `G3 present` (any G3 "
               "segment) and `G2 → G3` (a G3 segment preceded by G2). The LLR detector scores 40 s episodes from the "
               "timeline probabilities (independent of the decoder); its threshold is set per replicate on validation "
               "+ external calibration participants for ≤ 0.5 and ≤ 1 false alarm/h, so the rate on the test "
               "participants checks that calibration.", ""])
    rules = {"intake_g3": "G3 present", "intake_g2_g3": "G2 → G3"}
    cols, labels = [], {}
    for part, short in (("test_sequences", "seq"), ("test_wild", "wild")):
        for r, name in rules.items():
            for m in ("sensitivity", "specificity", "balanced_accuracy"):
                c = f"{part}.{r}.{m}"
                cols.append(c)
                labels[c] = f"{short} {name} {dict(sensitivity='sens', specificity='spec', balanced_accuracy='bacc')[m]}"
    cols.append("test_external.external.g3_events_per_hour")
    labels[cols[-1]] = "ext G3 present FA/h"
    t, num = ms_table(sub, ["window", "dec"], cols, labels, "window / decoder")
    save(f"{exp}_intake_rules", t, num, f"{exp} — intake from the Gesture Timeline (rules)")
    cols, labels = [], {}
    for fa, fname in (("fa05", "≤ 0.5/h"), ("fa10", "≤ 1/h")):
        for part, short in (("test_sequences", "seq"), ("test_wild", "wild")):
            for m in ("sensitivity", "specificity"):
                c = f"{part}.intake_llr_{fa}.{m}"
                cols.append(c)
                labels[c] = f"{fname}: {short} {m[:4]}"
        c = f"test_external.external.llr_{fa}_events_per_hour"
        cols.append(c)
        labels[c] = f"{fname}: ext FA/h"
    t, num = ms_table(sub[sub.decoder == "none"], "window", cols, labels, "window")
    save(f"{exp}_intake_llr", t, num, f"{exp} — intake from the LLR detector (calibrated false-alarm budget)")
    t, num = ms_table(sub, ["window", "dec"], ["test_external.external.gesture_events_per_hour",
                                               "test_external.external.g3_events_per_hour"],
                      {"test_external.external.gesture_events_per_hour": "any gesture events/h",
                       "test_external.external.g3_events_per_hour": "G3 events/h (= G3 present FA/h)"}, "window / decoder")
    save(f"{exp}_intake_false_alarms", t, num, f"{exp} — events per hour on the external participants (no gestures performed)")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    operating_points(ax, exp, sub)
    savefig(fig, f"{exp}_intake_operating_points", f"{exp} intake operating points")


def operating_points(ax, exp, sub, mk="o", label_exp=False):
    """Sensitivity in daily activity vs false alarms per hour on the external participants."""
    colors = {"0.75 s": "tab:blue", "1 s": "tab:orange", "2 s": "tab:green", "5 s": "tab:red"}
    for w, g in sub.groupby("window", sort=False):
        pts = []
        for d in DECODERS:
            q = g[g.decoder == d]
            pts.append((q["test_external.external.g3_events_per_hour"].mean(), q["test_wild.intake_g3.sensitivity"].mean(),
                        {"none": "o", "hmm": "s", "hsmm": "D"}[d]))
        q = g[g.decoder == "none"]
        for fa, m in (("fa05", "v"), ("fa10", "^")):
            pts.append((q[f"test_external.external.llr_{fa}_events_per_hour"].mean(),
                        q[f"test_wild.intake_llr_{fa}.sensitivity"].mean(), m))
        for x, y, m in pts:
            ax.scatter(x, 100 * y, marker=m, color=colors.get(w, "k"), s=36 if not label_exp else 28,
                       edgecolor="k" if label_exp and exp == "E2" else None, alpha=0.85)
        ax.plot([p[0] for p in pts[:3]], [100 * p[1] for p in pts[:3]], color=colors.get(w, "k"), lw=0.8,
                ls="-" if exp == "E2" else "--")
        if label_exp:
            ax.annotate(f"{exp} {w}", (pts[0][0], 100 * pts[0][1]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("false alarms per hour, external participants (log)")
    ax.set_ylabel("intake sensitivity in daily activity (%)")
    ax.grid(alpha=0.3, which="both")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=m, ls="", color="gray", label=l) for m, l in
         (("o", "G3 present, no decoder"), ("s", "G3 present, HMM"), ("D", "G3 present, HSMM"),
          ("v", "LLR ≤ 0.5/h"), ("^", "LLR ≤ 1/h"))]
    h += [Line2D([], [], color=c, label=w) for w, c in colors.items()]
    ax.legend(handles=h, fontsize=6, loc="lower right")
    ax.set_title(f"{exp if not label_exp else 'E1 (dashed) vs E2 (solid)'}: intake operating points", fontsize=8)


def comparison():
    """E1 vs E2 comparison tables and figures."""
    MD.extend(["## E1 vs E2", ""])
    a1, a2 = stage_a("E1"), stage_a("E2")
    cols = {"val.window.macro_f1": "val window macro F1", "test_sequences.window.macro_f1": "seq window macro F1",
            "test_wild.window.macro_f1": "wild window macro F1", "test_sequences.timeline.seg_f1_25": "seq seg F1@25",
            "test_wild.timeline.seg_f1_25": "wild seg F1@25", "test_sequences.window.f1.G5": "seq G5 F1",
            "test_wild.window.f1.G5": "wild G5 F1",
            "test_external.external.gesture_events_per_hour": "ext gesture events/h",
            "test_external.external.g3_events_per_hour": "ext G3 events/h"}
    m1 = a1.groupby(list(FACTORS))[list(cols)].mean()
    m2 = a2.groupby(list(FACTORS))[list(cols)].mean()
    j = m1.join(m2, lsuffix="_E1", rsuffix="_E2", how="inner")
    lines = ["| metric | E1 mean | E2 mean | E2 − E1 | E2 better in | Wilcoxon p |", "|---|---|---|---|---|---|"]
    rows = []
    for c, lab in cols.items():
        d = j[c + "_E2"] - j[c + "_E1"]
        hour = "per_hour" in c
        better = (d < 0).mean() if hour else (d > 0).mean()
        p = stats.wilcoxon(d).pvalue
        rows.append({"metric": c, "E1": j[c + "_E1"].mean(), "E2": j[c + "_E2"].mean(), "delta": d.mean(), "better": better, "p": p})
        f = (lambda v: f"{v:.1f}") if hour else (lambda v: f"{100 * v:.1f}")
        lines.append(f"| {lab} | {f(j[c + '_E1'].mean())} | {f(j[c + '_E2'].mean())} | {f(d.mean()) if hour else f'{100 * d.mean():+.1f}'} "
                     f"| {100 * better:.0f} % of {len(j)} | {p:.1e} |")
    save("cmp_stage_a_paired", "\n".join(lines), pd.DataFrame(rows),
         "Same 240 signal configurations trained under E1 and under E2 (stage A, paired by configuration)")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, c in zip(axes, ["test_sequences.window.macro_f1", "test_wild.window.macro_f1",
                            "test_external.external.g3_events_per_hour"]):
        x, y = j[c + "_E1"], j[c + "_E2"]
        hour = "per_hour" in c
        sc = 1 if hour else 100
        for w, mk in zip(sorted(j.index.get_level_values("window_s").unique()), "ov^sD"):
            msk = j.index.get_level_values("window_s") == w
            ax.scatter(sc * x[msk], sc * y[msk], s=9, marker=mk, alpha=0.7, label=f"{w:g} s")
        lo, hi = min(sc * x.min(), sc * y.min()), max(sc * x.max(), sc * y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        if hour:
            ax.set_xscale("symlog")
            ax.set_yscale("symlog")
        ax.set_xlabel("E1")
        ax.set_ylabel("E2")
        ax.set_title(cols[c], fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, title="window", title_fontsize=6)
    fig.suptitle("Each point = one signal configuration (stage A, mean of 3 folds)", fontsize=9)
    savefig(fig, "cmp_stage_a_scatter", "E1 vs E2 per configuration")
    rows = []
    for exp in ("E1", "E2"):
        r = RUNS[(RUNS.experiment == exp) & (RUNS.config_key == FINAL[exp]) & RUNS.replicate.str.match(r"f\d_s\d")]
        p = PP[(PP.experiment == exp) & (PP.config_key == FINAL[exp])]
        for lab, s in [("window macro F1 — blind seq", r["test_sequences.window.macro_f1"]),
                       ("window macro F1 — daily activity", r["test_wild.window.macro_f1"]),
                       ("window G5 precision — daily activity", r["test_wild.window.precision.G5"]),
                       ("window G3 precision — daily activity", r["test_wild.window.precision.G3"])]:
            rows.append({"exp": exp, "level": "window", "metric": lab, "mean": s.mean(), "std": s.std()})
        for d in ("none", "hmm"):
            q = p[p.decoder == d]
            for part in ("test_sequences", "test_wild"):
                for c, lab in (("timeline.seg_f1_25", "seg F1@25"), ("order.exact_match", "Order EM")):
                    rows.append({"exp": exp, "level": "sequence", "metric": f"{lab} — {SETS[part]} — {DEC_LABEL[d]}",
                                 "mean": q[f"{part}.{c}"].mean(), "std": q[f"{part}.{c}"].std()})
        q = p[p.decoder == "hmm"]
        for part in ("test_sequences", "test_wild"):
            for m in ("sensitivity", "specificity"):
                c = f"{part}.intake_llr_fa10.{m}"
                rows.append({"exp": exp, "level": "intake", "metric": f"LLR ≤1 FA/h {m} — {SETS[part]}",
                             "mean": q[c].mean(), "std": q[c].std()})
        for d in ("none", "hmm"):
            c = "test_external.external.g3_events_per_hour"
            q = p[p.decoder == d]
            rows.append({"exp": exp, "level": "intake", "metric": f"G3 rule FA/h — {DEC_LABEL[d]}", "mean": q[c].mean(), "std": q[c].std()})
        c = "test_external.external.llr_fa10_events_per_hour"
        rows.append({"exp": exp, "level": "intake", "metric": "LLR ≤1 FA/h — FA/h", "mean": q[c].mean(), "std": q[c].std()})
    df = pd.DataFrame(rows)
    w = df.pivot_table(index=["level", "metric"], columns="exp", values=["mean", "std"], sort=False)
    lines = ["| level | metric | E1 | E2 |", "|---|---|---|---|"]
    for (lv, m), r in w.iterrows():
        h = m.startswith("G3 rule FA/h") or m.endswith("— FA/h")
        lines.append(f"| {lv} | {m} | {pct(r[('mean', 'E1')], r[('std', 'E1')], h)} | {pct(r[('mean', 'E2')], r[('std', 'E2')], h)} |")
    save("cmp_selected_models", "\n".join(lines), df,
         f"Selected models: E1 `{FINAL['E1']}` vs E2 `{FINAL['E2']}` (3 folds × 3 seeds)")
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6))
    for i, exp in enumerate(("E1", "E2")):
        r = RUNS[(RUNS.experiment == exp) & (RUNS.config_key == FINAL[exp]) & RUNS.replicate.str.match(r"f\d_s\d")]
        plot_cm(axes[i, 0], confusion(r.path, "test_wild"), f"{exp} — daily activity (window)")
        plot_cm(axes[i, 1], confusion(r.path, "test_external"), f"{exp} — external, no gestures (window)", rows=[3, 4])
    savefig(fig, "cmp_confusion", "E1 vs E2 confusion")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for exp in ("E1", "E2"):
        operating_points(ax, exp, pp_label(seq_configs(exp)), label_exp=True)
    savefig(fig, "cmp_intake_operating_points", "E1 vs E2 intake operating points")


def main():
    """Write reports/results_chapter."""
    load_indexes()
    MD.extend(["# Results chapter — data", "",
               "Generated by src/thesis/results_chapter.py. Mean ± std over replicates in %, except per-hour rates. "
               "Sets: validation = held-out gesture subjects composed into daily activity of external calibration "
               "participants (the only set used for selection); blind sequences = annotated lab protocol sessions "
               "(different people); sequences in daily activity = the same blind sessions embedded in daily activity "
               "of unseen external participants; external = unseen external participants, daily activity only.", ""])
    for exp in ("E1", "E2"):
        a = stage_a(exp)
        window_factors(exp, a)
        window_finalist(exp)
        sequence(exp)
        intake(exp)
    timeline_example()
    comparison()
    (OUT / "RESULTS_DATA.md").write_text("\n".join(MD) + "\n")
    print("wrote", OUT / "RESULTS_DATA.md")


if __name__ == "__main__":
    main()
