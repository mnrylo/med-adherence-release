"""Phase 0 evaluation: what the E2 finalist loses when it runs causally, as on a device.

    python -m embedded.phase0_eval [--workers 6]

Inputs: the predictions of phase0_predict.py (both variants) and, from the E2 run, the
artifacts, train_info and the decoder parameters E3 selected for the finalist (registry
postprocessing.json). Nothing is selected on a test set. Thresholds are calibrated as in E3: best validation
balanced accuracy under the false-event budget on the external calibration participants.

Part A, variants (offline decoding, exactly as in E3):
  * ``zerophase``: trained and run zero-phase (as in E2);
  * ``causal``: trained zero-phase, run causal;
  * ``causal_trained``: retrained with the causal filter (scripts/run_embedded.sh);
  * ``causal_trained_<precision>``: its ONNX fp32 / fp16 / int8 exports (phase0_quant.py).
  Metrics:
  * window macro F1 (argmax);
  * HMM timeline seg F1@25 and Order Exact Match;
  * LLR intake at fa05/fa10 on test_sequences / test_wild and FA/h on test_external.
  Threshold modes:
  * ``recalibrated``: on the variant's own validation streams;
  * ``from_zerophase`` / ``from_causal_trained``: the thresholds of the model the variant is
    derived from, i.e. deployed without recalibrating.
  fall_s0 (deployment model) uses the registry thresholds (zero-phase model) or the median of the
  fold thresholds (retrained model).

Parts B and C run on the deployable model: causal_trained when it exists, else causal.

Part B, fixed-lag Viterbi:
  the label of frame u is the state reached by backtracking L frames from the best state at
  u + L. The last L frames of a stream take the offline path, as a flush at the end. Lags are in
  seconds; "inf" is the offline Viterbi.

Part C, streaming LLR:
  40 s episodes ending every ``step`` seconds (2 / 5 / 10), with no episode aligned to the stream
  end (a live stream has none). Thresholds are recalibrated for each step.
  * Intake detection on test_wild: an alarm after the session starts.
  * Alarm latency = alarm time - intake end, where intake end is the end of the last G3/G4 frame
    of the embedded session.
  * Alarm time = episode end + look-ahead. The look-ahead is the data still needed for every
    window covering that frame: half a window plus one stride.
  * Rules: ``first`` alarms at the first positive episode; ``run_end`` waits for the first
    negative episode after the run (one event per run, as count_events counts them).

Output: outputs/embedded/phase0/tables/{filter,fixed_lag,llr_streaming}[_summary].csv,
        llr_latency_sessions.csv, outputs/embedded/phase0/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from embedded import emb as E
from thesis import metrics as M
from thesis import postproc as PP
from core.hmm import PostProcessor, calibrate_threshold, count_events
from core.sources import LABELS

TABLES = E.PHASE0 / "tables"
HZ, LLR_HZ = PP.HZ, PP.LLR_HZ
FA_BUDGETS = PP.FA_BUDGETS
LAGS_S = [0, 0.5, 1, 2, 3, 5, 10, 20, 30, 60, "inf"]
STEPS_S = [2, 5, 10]
G3, G4 = LABELS.index("G3"), LABELS.index("G4")


def fa_name(fa):
    """Budget name, e.g. fa10."""
    return f"fa{PP._fa(fa)}"


class VRep(PP.Replicate):
    """A finalist replicate whose window probabilities come from a phase-0 variant."""

    def __init__(self, rep: str, variant: str):
        self.rep, self.variant = rep, variant
        rep_dir = (E.CAUSAL_DIR if variant.startswith("causal_trained") else E.CONFIG_DIR) / rep
        if rep != E.DEPLOY_REP:
            super().__init__(rep_dir)
        else:
            self.dir, self.cfg_dir = rep_dir, rep_dir.parent
            self.man = json.loads((self.cfg_dir / "artifacts" / "manifest.json").read_text())
            self.info = json.loads((rep_dir / "train_info.json").read_text())
            self.frames = np.load(self.cfg_dir / "artifacts" / "frames.npz")
            self.fold = "all"
            self.prior = np.array(self.info["prior"])
            self.ov = int(round(100 * (1 - self.man["stride"] / self.man["window"])))
            self.meta = {"val": {}, "test_sequences": self.man["lab"], "test_wild": self.man["wild"]}
            self.probs, self.tl_cache = {}, {}
            self.y10_val = [self.frames[f"y10/{s}"].astype(np.int64) for f in self.man["folds"].values()
                            for s in f["val"]]

    def load(self, part):
        """Window probabilities of a set from the variant's predictions."""
        if self.variant == "causal_trained":
            return super().load(part)
        if part not in self.probs:
            with np.load(E.pred_dir(self.variant, self.rep) / f"{part}.npz") as z:
                self.probs[part] = {k: z[k].astype(np.float32) for k in z.files}
        return self.probs[part]

    def lookahead_s(self):
        """Half a window plus one stride, in seconds."""
        return self.man["window_s"] / 2 + self.man["stride_s"]

    def llr_pp(self, p):
        """PostProcessor and 2 Hz timelines of the LLR, as Replicate.llr_scores builds them."""
        r = HZ // LLR_HZ
        pp = PostProcessor(self.durations(p["md"], LLR_HZ), self.prior, self.post_cfg(p["st"]), LLR_HZ)
        a = self.alpha(p) * r
        pp.emissions = lambda post: (np.log(np.clip(post, PP.POST_CFG["emission_floor"], 1)) - np.log(self.prior)) * a
        return pp, r

    def llr_streaming(self, part, p, step_s) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Episode scores and episode end frames (2 Hz) of every stream: episodes end every step_s."""
        pp, r = self.llr_pp(p)
        tls = [t[:len(t) // r * r].reshape(-1, r, PP.K).mean(1) if len(t) >= r else t.mean(0, keepdims=True)
               for t in self.timelines(part, p["ov"])]
        W, s = int(PP.POST_CFG["episode_s"] * LLR_HZ), int(step_s * LLR_HZ)
        eps, owner, ends = [], [], []
        for i, t in enumerate(tls):
            e = list(range(s, len(t) + 1, s)) or [len(t)]
            eps += [t[max(0, x - W):x] for x in e]
            owner += [i] * len(e)
            ends.append(np.array(e))
        owner = np.array(owner)
        sc = pp.llrs_batch(eps)["llr_phased"]
        return [sc[owner == i] for i in range(len(tls))], ends


def thresholds(val_max, yv, cal_eps, cal_h):
    """LLR thresholds for every false-event budget."""
    return {fa: calibrate_threshold(val_max, yv, cal_eps, cal_h, fa) for fa in FA_BUDGETS}


def smax(scores):
    """Maximum episode score per stream (-inf for streams without episodes)."""
    return np.array([s.max() if len(s) else -np.inf for s in scores])


def part_a(R: VRep, p: dict, fixed: dict[str, dict]) -> tuple[list[dict], dict]:
    """Offline metrics of one replicate: thresholds recalibrated on its validation streams (fold
    replicates) plus every named threshold set in ``fixed``."""
    parts = ["test_sequences", "test_wild"] + ([] if R.rep == E.DEPLOY_REP else ["val"])
    sc = {part: R.llr_scores(part, p) for part in [*parts, "test_external"]
          + ([] if R.rep == E.DEPLOY_REP else ["external_calibration"])}
    test_h = R.external_hours("test_external")
    modes = {}
    if R.rep != E.DEPLOY_REP:
        yv = np.array([R.meta["val"][s]["intake"] for s in R.sids("val")])
        modes["recalibrated"] = thresholds(smax(sc["val"]), yv, sc["external_calibration"],
                                           R.external_hours("external_calibration"))
    modes.update(fixed)
    base = {"variant": R.variant, "replicate": R.rep}
    for part in parts:
        q = R.load(part)
        y = np.concatenate([R.frames[f"y/{s}"].astype(np.int64)[:len(q[s])] for s in R.sids(part)])
        pw = np.concatenate([q[s].argmax(1) for s in R.sids(part)])
        res, rows = PP.timeline_metrics(R, part, R.decode(part, p))
        base.update({f"{part}.window.macro_f1": M.classification(y, pw)["macro_f1"],
                     f"{part}.hmm.seg_f1_25": res["timeline"]["seg_f1_25"],
                     f"{part}.hmm.timeline_macro_f1": res["timeline"]["macro_f1"],
                     f"{part}.hmm.order_exact_match": res["order"]["exact_match"],
                     f"{part}.g3.events_sens": res["intake_g3"]["sensitivity"]})
    out = []
    for mode, taus in modes.items():
        row = {**base, "thresholds": mode}
        for fa, tau in taus.items():
            row[f"tau_{fa_name(fa)}"] = tau
            for part in parts:
                yi = np.array([R.meta[part][s]["intake"] for s in R.sids(part)])
                b = M.binary(yi, (smax(sc[part]) > tau).astype(int))
                row[f"{part}.llr_{fa_name(fa)}.sensitivity"] = b["sensitivity"]
                row[f"{part}.llr_{fa_name(fa)}.specificity"] = b["specificity"]
                row[f"{part}.llr_{fa_name(fa)}.balanced_accuracy"] = b["balanced_accuracy"]
            row[f"test_external.llr_{fa_name(fa)}.events_per_hour"] = \
                sum(count_events(s, tau) for s in sc["test_external"]) / test_h
        out.append(row)
    return out, modes.get("recalibrated")


def fixed_lag_paths(hmm, logE: np.ndarray, lags: list[int]) -> dict[int, np.ndarray]:
    """Fixed-lag Viterbi paths (class labels) for every lag in frames, plus -1 = offline."""
    e = logE[:, hmm.state_class]
    T, S = e.shape
    back = np.zeros((T, S), np.int32)
    best = np.zeros(T, np.int64)
    delta = hmm.log_start + e[0]
    best[0] = delta.argmax()
    for t in range(1, T):
        cand = delta[:, None] + hmm.logA
        back[t] = cand.argmax(0)
        delta = cand.max(0) + e[t]
        best[t] = delta.argmax()
    full = np.empty(T, np.int64)
    full[-1] = np.where(hmm.final_mask, delta, -np.inf).argmax()
    for t in range(T - 1, 0, -1):
        full[t - 1] = back[t, full[t]]
    out = {-1: hmm.state_class[full]}
    for L in lags:
        if L >= T:
            out[L] = out[-1]
            continue
        cur = best[L:].copy()
        t_idx = np.arange(L, T)
        for j in range(L):
            cur = back[t_idx - j, cur]
        out[L] = np.concatenate([hmm.state_class[cur], out[-1][T - L:]])
    return out


def part_b(R: VRep, p: dict) -> list[dict]:
    """Fixed-lag Viterbi metrics of one replicate at every lag."""
    lags = {lag: (-1 if lag == "inf" else int(round(lag * HZ))) for lag in LAGS_S}
    frames = sorted({v for v in lags.values() if v >= 0})
    pp = PostProcessor(R.durations(p["md"]), R.prior, R.post_cfg(p["st"]), HZ)
    rows = []
    for part in ["val", "test_sequences", "test_wild"]:
        per = [fixed_lag_paths(pp.seg, e, frames) for e in R.logE(R.timelines(part, p["ov"]), p)]
        for lag, L in lags.items():
            res, _ = PP.timeline_metrics(R, part, [d[L] for d in per])
            rows.append({"variant": R.variant, "replicate": R.rep, "set": part, "lag_s": str(lag),
                         "record_latency_s": np.inf if L < 0 else L / HZ + R.lookahead_s(),
                         "seg_f1_25": res["timeline"]["seg_f1_25"], "timeline_macro_f1": res["timeline"]["macro_f1"],
                         "order_exact_match": res["order"]["exact_match"],
                         "order_edit_distance": res["order"]["edit_distance"]})
    return rows


def intake_times(R: VRep, streams: dict) -> dict[str, tuple[float, float, float]]:
    """(session start, intake end, G3 end) in seconds of every intake session of test_wild."""
    out = {}
    for sid in R.sids("test_wild"):
        if not R.meta["test_wild"][sid]["intake"]:
            continue
        off, n = streams[sid]["offset"], streams[sid]["length"]
        fs = R.man["fs"]
        a, b = int(np.floor(off / fs * HZ)), int(np.ceil((off + n) / fs * HZ))
        y = R.frames[f"y10/{sid}"][a:b]
        idx, i3 = np.flatnonzero((y == G3) | (y == G4)), np.flatnonzero(y == G3)
        if len(idx):
            out[sid] = (off / fs, (a + idx[-1] + 1) / HZ, (a + i3[-1] + 1) / HZ if len(i3) else np.nan)
    return out


def part_c(R: VRep, p: dict, streams: dict) -> tuple[list[dict], list[dict]]:
    """Streaming-LLR detection, false events and alarm latency of one replicate at every step."""
    rows, lat_rows = [], []
    yv = np.array([R.meta["val"][s]["intake"] for s in R.sids("val")])
    cal_h, test_h = R.external_hours("external_calibration"), R.external_hours("test_external")
    times = intake_times(R, streams)
    starts = {s: streams[s]["offset"] / R.man["fs"] for s in R.sids("test_wild")}
    look = R.lookahead_s()
    for step in STEPS_S:
        sc = {part: R.llr_streaming(part, p, step) for part in
              ["val", "external_calibration", "test_wild", "test_external"]}
        taus = thresholds(smax(sc["val"][0]), yv, sc["external_calibration"][0], cal_h)
        for fa, tau in taus.items():
            row = {"variant": R.variant, "replicate": R.rep, "step_s": step, "budget": fa_name(fa), "tau": tau,
                   "test_external.events_per_hour": sum(count_events(s, tau) for s in sc["test_external"][0]) / test_h}
            det = {}
            for sid, s, e in zip(R.sids("test_wild"), *sc["test_wild"]):
                t_end = e / LLR_HZ
                pos = (s > tau) & (t_end >= starts[sid])
                det[sid] = bool(pos.any())
                if sid not in times:
                    continue
                if not pos.any():
                    lat_rows += [{"replicate": R.rep, "step_s": step, "budget": fa_name(fa), "sid": sid,
                                  "detected": 0, "rule": rule, "latency_s": np.nan, "latency_g3_s": np.nan}
                                 for rule in ("first", "run_end")]
                    continue
                i = int(np.flatnonzero(pos)[0])
                j = i
                while j + 1 < len(s) and s[j + 1] > tau:
                    j += 1
                t_run_end = t_end[j + 1] if j + 1 < len(s) else t_end[j]
                for rule, t_al in (("first", t_end[i]), ("run_end", t_run_end)):
                    lat_rows.append({"replicate": R.rep, "step_s": step, "budget": fa_name(fa), "sid": sid,
                                     "detected": 1, "rule": rule, "latency_s": t_al + look - times[sid][1],
                                     "latency_g3_s": t_al + look - times[sid][2]})
            yi = np.array([R.meta["test_wild"][s]["intake"] for s in R.sids("test_wild")])
            b = M.binary(yi, np.array([det[s] for s in R.sids("test_wild")], int))
            row.update({"test_wild.sensitivity": b["sensitivity"], "test_wild.specificity": b["specificity"],
                        "test_wild.balanced_accuracy": b["balanced_accuracy"]})
            for rule in ("first", "run_end"):
                sel = [r for r in lat_rows if r["step_s"] == step and r["budget"] == fa_name(fa)
                       and r["rule"] == rule and r["detected"]]
                for name, key in (("", "latency_s"), ("_g3", "latency_g3_s")):
                    lat = np.array([r[key] for r in sel])
                    lat = lat[np.isfinite(lat)]
                    row[f"latency{name}_{rule}_median_s"] = float(np.median(lat)) if len(lat) else np.nan
                    row[f"latency{name}_{rule}_p95_s"] = float(np.percentile(lat, 95)) if len(lat) else np.nan
            rows.append(row)
    return rows, lat_rows


def dynamic_parts(R: VRep, p: dict, streams: dict, out: dict):
    """Parts B and C (streaming decoders) of one replicate."""
    out["b"] += part_b(R, p)
    c, lat = part_c(R, p, streams)
    out["c"] += c
    out["lat"] += [{"variant": R.variant, **r} for r in lat]


def fold_job(args):
    """Parts A, B and C for one fold replicate."""
    rep, p, streams, variants, streaming = args
    out = {"a": [], "b": [], "c": [], "lat": [], "taus": {}}
    rows, out["taus"]["zerophase"] = part_a(VRep(rep, "zerophase"), p, {})
    out["a"] += rows
    for v in variants[1:]:
        R = VRep(rep, v)
        fixed = ({"from_zerophase": out["taus"]["zerophase"]} if v == "causal" else
                 {"from_causal_trained": out["taus"]["causal_trained"]} if v.startswith("causal_trained_") else {})
        rows, out["taus"][v] = part_a(R, p, fixed)
        out["a"] += rows
        if v == streaming:
            dynamic_parts(R, p, streams, out)
    print(f"[{E.now()}] {rep} done", flush=True)
    return out


def deploy_job(args):
    """fall_s0 on the test sets: registry thresholds for the zero-phase model, fold medians for the retrained one."""
    p, variants, fixed = args
    out = {"a": [], "b": [], "c": [], "lat": []}
    for v in variants:
        rows, _ = part_a(VRep(E.DEPLOY_REP, v), p, fixed[v])
        out["a"] += rows
    print(f"[{E.now()}] {E.DEPLOY_REP} done", flush=True)
    return out


def summarise(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean ± std text per group."""
    g = df.groupby(keys)[metrics]
    m, s = g.mean(), g.std()
    out = m.copy()
    for c in metrics:
        out[c] = [f"{a:.4f} ± {b:.4f}" if pd.notna(b) else f"{a:.4f}" for a, b in zip(m[c], s[c])]
    return out.reset_index()


def paired(df: pd.DataFrame, metrics: list[str]) -> list[dict]:
    """Every (variant, thresholds) against the zero-phase model (recalibrated), paired over the fold replicates."""
    z = df[(df.variant == "zerophase") & (df.thresholds == "recalibrated")].set_index("replicate")
    out = []
    for (v, mode), c in df.groupby(["variant", "thresholds"]):
        if v == "zerophase":
            continue
        c = c.set_index("replicate").loc[z.index]
        for m in metrics:
            d = (c[m] - z[m]).to_numpy(float)
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            out.append({"variant": v, "thresholds": mode, "metric": m, "zerophase": float(z[m].mean()),
                        "value": float(c[m].mean()), "diff_mean": float(d.mean()), "diff_min": float(d.min()),
                        "diff_max": float(d.max()), "wilcoxon_p": p, "n": len(d)})
    return out


def main():
    """Run the phase-0 evaluation and write its tables and summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reps", nargs="+", default=E.FOLD_REPS, help="fold replicates (fall_s0 is always added)")
    a = ap.parse_args()
    p = {k: v for k, v in E.postprocessing()["decoder"].items() if k in ("decoder", "ov", "st", "md", "t")}
    streams = json.loads((E.PHASE0 / "predictions" / "causal" / "streams.json").read_text())["test_wild"]
    done = lambda r: (E.CAUSAL_DIR / r / "manifest.json").exists() and \
        json.loads((E.CAUSAL_DIR / r / "manifest.json").read_text()).get("status") == "done"
    retrained = all(done(r) for r in [*a.reps, E.DEPLOY_REP])
    precisions = [d.name for d in sorted((E.PHASE0 / "predictions").glob("causal_trained_*"))
                  if all((d / r / "test_external.npz").exists() for r in [*a.reps, E.DEPLOY_REP])]
    variants = ["zerophase", "causal", *(["causal_trained", *precisions] if retrained else [])]
    streaming = variants[-1]
    print(f"variants {variants}; streaming decoders on {streaming}", flush=True)
    with Pool(min(a.workers, len(a.reps))) as pool:
        res = pool.map(fold_job, [(r, p, streams, variants, streaming) for r in a.reps])
    reg = {fa: E.postprocessing()["intake"]["llr"]["thresholds"][fa_name(fa)] for fa in FA_BUDGETS}
    fixed = {"zerophase": {"registry": reg}, "causal": {"registry": reg}}
    for v in variants[2:]:
        fixed[v] = {"fold_median": {fa: float(np.median([x["taus"][v][fa] for x in res])) for fa in FA_BUDGETS}}
    res.append(deploy_job((p, variants, fixed)))
    TABLES.mkdir(parents=True, exist_ok=True)
    A = pd.DataFrame([r for x in res for r in x["a"]])
    B = pd.DataFrame([r for x in res for r in x["b"]])
    Cc = pd.DataFrame([r for x in res for r in x["c"]])
    L = pd.DataFrame([r for x in res for r in x["lat"]])
    A.to_csv(TABLES / "filter.csv", index=False)
    B.to_csv(TABLES / "fixed_lag.csv", index=False)
    Cc.to_csv(TABLES / "llr_streaming.csv", index=False)
    L.to_csv(TABLES / "llr_latency_sessions.csv", index=False)

    folds = A[A.replicate != E.DEPLOY_REP]
    am = [c for c in A.columns if "." in c]
    summarise(A, ["variant", "thresholds"], am).to_csv(TABLES / "filter_summary.csv", index=False)
    summarise(B, ["variant", "set", "lag_s", "record_latency_s"],
              ["seg_f1_25", "timeline_macro_f1", "order_exact_match"]).to_csv(TABLES / "fixed_lag_summary.csv",
                                                                               index=False)
    cm = [c for c in Cc.columns if c.startswith(("test_", "latency"))]
    summarise(Cc, ["variant", "step_s", "budget"], cm).to_csv(TABLES / "llr_streaming_summary.csv", index=False)
    det = L[L.detected == 1]
    pooled = pd.concat([det.groupby(["variant", "step_s", "budget", "rule"])[k].agg(
        median="median", p95=lambda v: np.nanpercentile(v, 95), max="max", n="count").reset_index().assign(reference=ref)
        for k, ref in (("latency_s", "intake_end"), ("latency_g3_s", "g3_end"))])
    pooled.to_csv(TABLES / "llr_latency_pooled.csv", index=False)
    pd.DataFrame(paired(folds, am)).to_csv(TABLES / "filter_paired.csv", index=False)
    E.write_json(E.PHASE0 / "summary.json", {
        "created": E.now(), "config_key": E.CONFIG_KEY, "causal_key": E.CAUSAL_KEY, "variants": variants,
        "streaming_variant": streaming, "decoder_params": p, "replicates": [*a.reps, E.DEPLOY_REP],
        "lags_s": [str(x) for x in LAGS_S], "llr_steps_s": STEPS_S, "fixed_thresholds": fixed,
        "lookahead_s": VRep(a.reps[0], "causal").lookahead_s()})
    print(pooled.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
