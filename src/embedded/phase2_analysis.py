"""Phase 2 analysis: cost and energy tables of the placement comparison.

    python -m embedded.phase2_analysis [--confirmatory 3,4,...]

Reads outputs/embedded/phase2/energy_runs.csv and
outputs/embedded/phase1/devices/e2_rep*_*/{watch,phone}/summary.json, analysis.json; writes
outputs/embedded/phase2/analysis/{cost.csv, energy_runs.csv, energy_model_*.json}.

* cost.csv: per device and condition, mean [min, max] over all repetitions of the quantities that
  do not depend on battery state (CPU s/h, inference p50/p95 in real time, LLR per step, PSS peak,
  bytes and messages per hour, alarms).
* energy_runs.csv: watch charge per hour of every run, with repetition, sequence, position,
  temperature and starting charge, and the status of the repetition (pilot / off-protocol /
  confirmatory).
* energy_model.json: least-squares fit of charge/h ~ condition + position + repetition (fixed
  effects; with a balanced design this gives the same condition contrasts as the planned mixed
  model with a random repetition intercept), on the confirmatory repetitions only; contrasts
  P_k - C1 and P_i - P_j with t-based 95 % CIs. Written only when the model has residual degrees
  of freedom.
"""
from __future__ import annotations

import argparse
import itertools
import json
import numpy as np
import pandas as pd
from scipy import stats

from embedded import emb as E

DEV = E.PHASE1 / "devices"
OUT = E.PHASE2 / "analysis"
STATUS = {int(k): v for k, v in E.CONF["campaign"]["repetition_status"].items()}
CONDS = ["C0", "C1", "P1", "P2", "P3"]


def cost_table() -> pd.DataFrame:
    """Per run and role, the quantities that do not depend on battery state."""
    rows = []
    for d in sorted(DEV.glob("e2_rep*_*")):
        an = json.loads((d / "analysis.json").read_text()) if (d / "analysis.json").exists() else {}
        for role in ("watch", "phone"):
            f = d / role / "summary.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            h = s["elapsed_s"] / 3600
            inf = s.get("inference", {})
            c = an.get(role, {}).get("cost", {})
            rows.append({"rep": int(d.name.split("_")[1][3:]), "condition": d.name.split("_")[2], "role": role,
                         "cpu_s_per_h": s["cpu_ms"] / 1000 / h,
                         "inference_p50_ms": inf.get("p50_us", np.nan) / 1000 if s["windows"] else np.nan,
                         "inference_p95_ms": inf.get("p95_us", np.nan) / 1000 if s["windows"] else np.nan,
                         "llr_ms_per_step": s["stages"]["llr"]["mean_us"] / 1000 if s["stages"]["llr"]["count"] else np.nan,
                         "windows_per_h": s["windows"] / h,
                         "pss_mb_max": c.get("pss_mb_max", np.nan),
                         "tx_kib_per_h": s["tx_bytes"] / 1024 / h, "rx_kib_per_h": s["rx_bytes"] / 1024 / h,
                         "tx_msgs_per_h": s["tx_msgs"] / h, "alarms": s["alarms"]})
    return pd.DataFrame(rows)


def phone_current_mah_per_h(run: str) -> float:
    """Phone charge from the average current reported by the battery service, integrated over the
    unplugged samples (trapezoid rule). Unlike the charge counter, it has no 4.65 mAh steps and does
    not stay constant at full charge. Samsung reports this field in mA, not the documented uA."""
    m = pd.read_csv(DEV / run / "phone" / "metrics.csv")
    m = m[(m.plugged == 0) & m.current_avg_ua.notna()]
    i = m.current_avg_ua.abs() / (1.0 if m.current_avg_ua.abs().median() < 5000 else 1000.0)
    t = m.elapsed_s.to_numpy()
    return float(np.trapezoid(i.to_numpy(), t) / (t[-1] - t[0]))


def fit(e: pd.DataFrame, covariate: str | None = None) -> dict | None:
    """OLS of charge/h on condition (C1 reference), position and repetition dummies (+ optional covariate)."""
    reps = sorted(e.rep.unique())
    X, names = [np.ones(len(e))], ["intercept"]
    if covariate:
        X.append(e[covariate].to_numpy(float) - e[covariate].mean()); names.append(covariate)
    for c in CONDS:
        if c != "C1":
            X.append((e.condition == c).to_numpy(float)); names.append(f"cond_{c}")
    for p in sorted(e.position.unique())[1:]:
        X.append((e.position == p).to_numpy(float)); names.append(f"pos_{p}")
    for r in reps[1:]:
        X.append((e.rep == r).to_numpy(float)); names.append(f"rep_{r}")
    X = np.column_stack(X)
    y = e.charge_per_h.to_numpy()
    dof = len(y) - np.linalg.matrix_rank(X)
    if dof <= 0:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    tq = stats.t.ppf(0.975, dof)
    idx = {n: i for i, n in enumerate(names)}

    def contrast(a, b):
        v = np.zeros(len(names))
        if a != "C1":
            v[idx[f"cond_{a}"]] += 1
        if b != "C1":
            v[idx[f"cond_{b}"]] -= 1
        est, se = v @ beta, np.sqrt(v @ cov @ v)
        p = 2 * stats.t.sf(abs(est / se), dof)
        return {"contrast": f"{a} - {b}", "estimate": est, "ci_lo": est - tq * se, "ci_hi": est + tq * se, "p": p}

    cs = [contrast(p, "C1") for p in ("P1", "P2", "P3")] + [contrast(a, b) for a, b in itertools.combinations(["P1", "P2", "P3"], 2)]
    order = sorted(range(len(cs)), key=lambda i: cs[i]["p"])
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, cs[i]["p"] * (len(cs) - k)))
        cs[i]["p_holm"] = run
    def coef(n):
        i = idx[n]
        se = np.sqrt(cov[i, i])
        return {"estimate": float(beta[i]), "ci_lo": float(beta[i] - tq * se), "ci_hi": float(beta[i] + tq * se),
                "p": float(2 * stats.t.sf(abs(beta[i] / se), dof))}

    positions = {"1": {"estimate": 0.0}} | {n.split("_")[1]: coef(n) for n in names if n.startswith("pos_")}
    out = {"repetitions": [int(r) for r in reps], "runs": int(len(y)), "residual_dof": int(dof),
           "residual_sd_mah_per_h": float(np.sqrt(s2)), "contrasts": cs, "position_vs_first": positions,
           "raw_mean_mah_per_h": {c: float(e[e.condition == c].charge_per_h.mean()) for c in CONDS}}
    if covariate:
        out["covariate"] = {"name": covariate} | coef(covariate)
    return out


def main():
    """Write the cost table, energy runs and energy models."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmatory", default="", help="comma-separated repetitions (default: every rep >= 3)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cost = cost_table()
    agg = cost.drop(columns="rep").groupby(["role", "condition"]).agg(["mean", "min", "max"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.reset_index().to_csv(OUT / "cost.csv", index=False)

    e = pd.read_csv(E.PHASE2 / "energy_runs.csv")
    e = e[e.role == "watch"].copy()
    conf = [int(x) for x in a.confirmatory.split(",") if x] or [r for r in e.rep.unique() if r >= 3]
    e["status"] = e.rep.map(lambda r: "confirmatory" if r in conf else STATUS.get(r, "excluded"))
    e[["rep", "seq", "position", "condition", "status", "charge_per_h", "pct_start", "pct_end", "current_ma_mean",
       "temp_c", "cpu_s_per_h", "tx_kib_per_h"]].to_csv(OUT / "energy_runs.csv", index=False)
    print(e.pivot(index="condition", columns="rep", values="charge_per_h").round(2))
    ph = pd.read_csv(E.PHASE2 / "energy_runs.csv")
    ph = ph[ph.role == "phone"].copy()
    ph["counter_per_h"] = ph.charge_per_h
    ph["charge_per_h"] = [phone_current_mah_per_h(f"e2_rep{r}_{c}") for r, c in zip(ph.rep, ph.condition)]
    ph[["rep", "seq", "position", "condition", "charge_per_h", "counter_per_h", "pct_start", "temp_c",
        "cpu_s_per_h"]].to_csv(OUT / "energy_runs_phone.csv", index=False)
    m = fit(ph[ph.rep.isin(conf)])
    if m is not None:
        good = ph[ph.rep.isin(conf) & (ph.pct_start < 100)]
        m["counter_vs_current"] = {"corr": float(np.corrcoef(good.counter_per_h, good.charge_per_h)[0, 1]),
                                   "ratio": float(good.charge_per_h.mean() / good.counter_per_h.mean()),
                                   "runs": int(len(good))}
        (OUT / "energy_model_phone_confirmatory.json").write_text(json.dumps(m, indent=1))
        print("phone (current integral)", f"residual sd {m['residual_sd_mah_per_h']:.2f} mAh/h, dof {m['residual_dof']}")
        for c in m["contrasts"]:
            print(f"  {c['contrast']}: {c['estimate']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}] p_holm={c['p_holm']:.3f}")

    analyses = (("confirmatory", e[e.rep.isin(conf)], None),
                ("sensitivity_temperature", e[e.rep.isin(conf)], "temp_c"),
                ("sensitivity_with_rep2", e[e.rep.isin(conf + [2])], None),
                ("all_reps_sensitivity", e, None))
    for name, sub, cov in analyses:
        m = fit(sub, cov)
        if m is None:
            print(f"{name}: no residual degrees of freedom yet ({sub.rep.nunique()} repetition(s))")
            continue
        (OUT / f"energy_model_{name}.json").write_text(json.dumps(m, indent=1))
        print(name, f"reps {m['repetitions']}, residual sd {m['residual_sd_mah_per_h']:.2f} mAh/h, dof {m['residual_dof']}")
        for c in m["contrasts"]:
            print(f"  {c['contrast']}: {c['estimate']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}] p_holm={c['p_holm']:.3f}")


if __name__ == "__main__":
    main()
