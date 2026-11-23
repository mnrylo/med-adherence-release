"""Short-window tracks: the best 0.75 s and 1 s configuration of E1 and of E2, taken through the
whole pipeline (B -> C -> F -> E3) like the main finalists.

    python -m thesis.extra_windows

The main sweep continued only with the best windows (5 s for E1, 2 s for E2), so these windows
were screened only in stage A. For each experiment and window (a "track", e.g. E1_w075):

  A  the stage-A signal configuration with the best validation score at that window
  B  cnn / cnn_lstm / cnn_bilstm on that signal, 3 folds x 3 seeds -> best model
  C  the hyperparameter grid on it, seed 0 (default hyperparameters as a candidate) -> top 3
  F  the top 3 with 3 folds x 3 seeds -> best configuration of the track

Selection rule and replicates as in thesis.sweep (mean val.timeline.seg_f1_25, tie-break
val.window.macro_f1). Decisions: index/selection/<experiment>_w<win>_<stage>.json. When every
track is finished, index/selection/E3_windows.json lists the best configuration of each track,
and thesis.postproc decodes those next to the main finalists (role 'window_ref').
"""
from __future__ import annotations

from thesis import common as C
from thesis import sweep as SW

WINDOWS = [0.75, 1.0]
N_TOP_C = 3


def tracks() -> list[tuple[str, float, str]]:
    """(experiment, window, tag) of every short-window track."""
    return [(e, w, f"w{int(round(w * 100)):03d}") for e in ("E1", "E2") for w in WINDOWS]


def main():
    """Run the short-window tracks A -> B -> C -> F and write E3_windows.json."""
    SW.log("short-window tracks start")
    A = SW.stage_a()
    df = SW.build_index()
    top_a = {}
    for e, w, tag in tracks():
        cand = [k for k in A if C.parse_key(k)["window_s"] == w]
        top_a[(e, tag)] = SW.select(df, e, f"{tag}_A", cand, 1, note=f"stage-A configurations with {w} s windows")[0]
    B = {t: SW.stage_b([k]) for t, k in top_a.items()}
    SW.run_jobs([(t[0], k, SW.FULL_SEEDS) for t, ks in B.items() for k in ks], "windows-B")
    df = SW.build_index()
    top_b = {t: SW.select(df, t[0], f"{t[1]}_B", B[t], 1, note="3 folds x 3 seeds")[0] for t in B}
    Cc = {t: SW.stage_c([k]) for t, k in top_b.items()}
    SW.run_jobs([(t[0], k, SW.SCREEN_SEEDS) for t, ks in Cc.items() for k in ks], "windows-C")
    df = SW.build_index()
    df0 = df[df.replicate.str.endswith("_s0")]
    top_c = {t: SW.select(df0, t[0], f"{t[1]}_C", Cc[t] + [top_b[t]], N_TOP_C,
                          note="hyperparameter grid plus the default hyperparameters; scored on seed 0")
             for t in Cc}
    SW.run_jobs([(t[0], k, SW.FULL_SEEDS) for t, ks in top_c.items() for k in ks], "windows-F")
    df = SW.build_index()
    out = {"stage": "E3 short-window references", "created": C.now(), "windows": WINDOWS,
           "rule": "per track: A -> B -> C -> F as in sweep.py; the best stage-F configuration goes to E3",
           "configs": {}}
    for t, ks in top_c.items():
        best = SW.select(df, t[0], f"{t[1]}_F", ks, 1, note="3 folds x 3 seeds; the best goes to E3")[0]
        sc = SW.config_scores(df, t[0], [best]).iloc[0]
        out["configs"][f"{t[0]}_{t[1]}"] = {"experiment": t[0], "window_s": C.parse_key(best)["window_s"],
                                            "config_key": best, "stage_a_signal": top_a[t],
                                            "val.timeline.seg_f1_25": float(sc.score),
                                            "val.window.macro_f1": float(sc.tie)}
    C.write_json(C.SELECTION / "E3_windows.json", out)
    SW.log("short-window tracks finished")


if __name__ == "__main__":
    main()
