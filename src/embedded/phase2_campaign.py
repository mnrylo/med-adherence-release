"""Phase 2: energy campaign. One repetition = every condition for `--duration-s`, one after the other.

    python -m embedded.phase2_campaign run --rep 1 --phone <serial>  # needs adb to the phone only
    python -m embedded.phase2_campaign pull-watch --rep 1 --watch <serial> --phone <serial>  # afterwards, watch on the charger
    python -m embedded.phase2_campaign summary --reps 1,2,3  # outputs/embedded/phase2/energy_runs.csv

Each run is started on the phone with `pair=true`, so the phone sends the same configuration to the
watch over the Data Layer. Off the charger the watch drops Wi-Fi adb, so it is never contacted
during the campaign. Both devices end the run by its duration.

Conditions (the watch side; the phone runs the matching role):
  C0  idle, metrics only
  C1  IMU sampled at 100 Hz and discarded
  P3  replay -> whole pipeline on the watch, alarms to the phone      (IMU on)
  P2  replay -> filter + classifier on the watch, probabilities sent  (IMU on)
  P1  replay -> raw IMU sent, whole pipeline on the phone              (IMU on)
In P1-P3 the IMU is sampled and discarded while the replay feeds the pipeline, so every placement
pays the sensor cost and C1 is the baseline for the cost of the processing and the link.
All placements replay the same data: test_external from stream 0, in real time.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import time

from embedded import emb as E
from embedded import phase1_bench as B

PHASE2 = E.PHASE2
CONDITIONS = ["C0", "C1", "P3", "P2", "P1"]
WILLIAMS_SEED = E.CONF["campaign"]["williams_seed"]


def williams() -> list[list[str]]:
    """Ten sequences: each condition once per position, each ordered pair of conditions adjacent twice."""
    rng = random.Random(WILLIAMS_SEED)
    labels = rng.sample(CONDITIONS, len(CONDITIONS))
    n = len(labels)
    base = [0, 1, n - 1, 2, n - 2]
    seqs = [[labels[(b + i) % n] for b in base] for i in range(n)]
    seqs += [s[::-1] for s in seqs]
    pairs = [p for s in seqs for p in zip(s, s[1:])]
    assert all(pairs.count((x, y)) == 2 for x in labels for y in labels if x != y)
    assert all(len({s[i] for s in seqs}) == n for i in range(n))
    rng.shuffle(seqs)
    return seqs


def device_state(serial) -> dict:
    """Battery, radios and display settings, so that every repetition records what the scripts do not set."""
    out = {}
    for k in ("wifi_on", "bluetooth_on", "airplane_mode_on", "mobile_data", "stay_on_while_plugged_in"):
        out[k] = B.adb(serial, "shell", "settings", "get", "global", k, check=False, timeout=20).strip()
    for line in B.adb(serial, "shell", "dumpsys", "battery", check=False, timeout=20).splitlines():
        line = line.strip()
        if line.startswith(("level:", "temperature:", "AC powered:", "USB powered:", "Wireless powered:")):
            k, v = line.split(":", 1)
            out[k] = v.strip()
    return out


def cmd_wait_start(a):
    """Record the state, wait until the watch has left the charger and rested, then run the sequence."""
    a.conditions = ",".join(williams()[a.seq - 1])
    state = {"rep": a.rep, "seq": a.seq, "conditions": a.conditions, "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
             "watch": device_state(a.watch), "phone": device_state(a.phone), "watch_location": a.watch_location}
    log(a.rep, f"waiting for the watch to leave the charger; sequence {a.seq}: {a.conditions}")
    off, last = 0, None
    while off < a.off_polls:
        time.sleep(60)
        st = device_state(a.watch)
        if st.get("level"):
            last = st
        off = 0 if st.get("Wireless powered") == "true" or st.get("AC powered") == "true" else off + 1
    state["watch_last_seen"] = last
    log(a.rep, f"watch off the charger, last seen {last and last.get('level')} %; resting {a.rest_s} s")
    time.sleep(a.rest_s)
    if device_state(a.watch).get("Wireless powered") == "true":
        raise SystemExit("the watch is back on the charger: not started")
    state["phone_at_start"] = device_state(a.phone)
    PHASE2.mkdir(parents=True, exist_ok=True)
    (PHASE2 / f"state_rep{a.rep}.json").write_text(json.dumps(state, indent=1))
    cmd_run(a)


def log(rep, msg):
    """Time-stamped line on stdout and in the repetition's campaign log."""
    PHASE2.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(PHASE2 / f"campaign_rep{rep}.log", "a") as f:
        f.write(line + "\n")


def config(rep, cond, a) -> dict:
    """Benchmark config of one condition of a repetition."""
    cfg = {"run_id": f"e2_rep{rep}_{cond}", "condition": cond, "duration_s": a.duration_s,
           "metrics_period_s": a.metrics_period_s, "steps": [a.step]}
    if cond.startswith("P"):
        cfg.update({"source": "replay", "set": "test_external", "pacing": "realtime", "sensors_on": True})
    return cfg


def plugged(phone) -> str:
    """Power sources of the phone, or 'unplugged'."""
    out = B.adb(phone, "shell", "dumpsys", "battery", check=False)
    return ",".join(l.split(":")[0].strip() for l in out.splitlines() if "powered: true" in l) or "unplugged"


def cmd_run(a):
    """Run every condition of a repetition in the Williams order."""
    conds = a.conditions.split(",")
    for cond in conds:
        cfg = config(a.rep, cond, a)
        rid = cfg["run_id"]
        if (B.DEVICES / rid / "phone" / "summary.json").exists():
            log(a.rep, f"{rid}: already done, skipped")
            continue
        log(a.rep, f"{rid}: start ({a.duration_s} s), phone {plugged(a.phone)}")
        b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
        B.adb(a.phone, "shell", "am", "start", "-n", f"{B.PKG}/com.medmon.emb.bench.ControlActivity",
              "--es", "action", "start", "--es", "config_b64", b64, "--ez", "pair", "true")
        t0 = time.time()
        done = False
        while time.time() - t0 < a.duration_s + 900:
            time.sleep(60)
            r = B.adb(a.phone, "shell", "ls", f"{B.DEV_FILES}/runs/{rid}/phone/summary.json", check=False)
            if r.endswith("summary.json"):
                done = True
                break
        log(a.rep, f"{rid}: {'finished' if done else 'TIMEOUT'} after {time.time() - t0:.0f} s")
        B.pull({"phone": a.phone}, ["phone"], rid)
        if cond != conds[-1]:
            time.sleep(a.gap_s)
    log(a.rep, "campaign done")


def cmd_pull_watch(a):
    """Pull the watch files of a repetition and analyse every run."""
    for cond in a.conditions.split(","):
        rid = f"e2_rep{a.rep}_{cond}"
        B.pull({"watch": a.watch}, ["watch"], rid)
        if (B.DEVICES / rid).exists():
            B.analyze(rid)


def energy(d) -> dict:
    """Energy of one device over one run, from the charge counter and the battery percentage."""
    import pandas as pd
    m = pd.read_csv(d / "metrics.csv")
    m = m[m.plugged == 0]
    s = json.loads((d / "summary.json").read_text())
    h = (m.elapsed_s.iloc[-1] - m.elapsed_s.iloc[0]) / 3600
    scale = B.current_scale(m.current_now_ua)
    return {"hours": h, "charge_per_h": (m.charge_uah.iloc[0] - m.charge_uah.iloc[-1]) / 1000 / h,
            "pct_start": int(m.battery_pct.iloc[0]), "pct_end": int(m.battery_pct.iloc[-1]),
            "pct_per_h": (m.battery_pct.iloc[0] - m.battery_pct.iloc[-1]) / h,
            "current_ma_mean": float(-m.current_now_ua.mean() / scale),
            "temp_c": float(m.temp_dc.mean() / 10), "cpu_s_per_h": s["cpu_ms"] / 1000 / h,
            "tx_kib_per_h": s["tx_bytes"] / 1024 / h, "rx_kib_per_h": s["rx_bytes"] / 1024 / h,
            "imu_events": s["extra"].get("imu_events", 0), "stop": s["extra"].get("reason")}


def run_order(rep) -> tuple[int, list[str]]:
    """Williams sequence and order of one repetition. Repetition 1 was the pilot, in the fixed order (sequence 0)."""
    f = PHASE2 / f"state_rep{rep}.json"
    if not f.exists():
        return 0, CONDITIONS
    s = json.loads(f.read_text())
    return s["seq"], s["conditions"].split(",")


def cmd_summary(a):
    """Write phase2/energy_runs.csv from the analysed runs."""
    import pandas as pd
    rows = []
    for rep in a.reps.split(","):
        seq, order = run_order(rep)
        for cond in CONDITIONS:
            for role in ("watch", "phone"):
                d = B.DEVICES / f"e2_rep{rep}_{cond}" / role
                if (d / "summary.json").exists():
                    rows.append({"rep": int(rep), "seq": seq, "position": order.index(cond) + 1,
                                 "condition": cond, "role": role, **energy(d)})
    df = pd.DataFrame(rows)
    PHASE2.mkdir(parents=True, exist_ok=True)
    df.to_csv(PHASE2 / "energy_runs.csv", index=False)
    for role in ("watch", "phone"):
        t = df[df.role == role].copy()
        base = t[t.condition == "C1"].set_index("rep").charge_per_h
        t["minus_C1_per_h"] = t.charge_per_h - t.rep.map(base)
        print(f"\n{role}\n" + t[["rep", "condition", "charge_per_h", "minus_C1_per_h", "pct_per_h", "current_ma_mean",
                                  "cpu_s_per_h", "tx_kib_per_h", "temp_c"]].round(2).to_string(index=False))


def main():
    """Command-line interface."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    w = sub.add_parser("wait-start")
    w.add_argument("--seq", type=int, required=True, help="Williams sequence, 1 to 10")
    w.add_argument("--off-polls", type=int, default=5, help="minutes off the charger or out of adb reach")
    w.add_argument("--rest-s", type=int, default=600)
    w.add_argument("--watch-location", default="table")
    for q in (p, w):
        q.add_argument("--duration-s", type=int, default=3600)
        q.add_argument("--gap-s", type=int, default=120)
        q.add_argument("--metrics-period-s", type=int, default=30)
        q.add_argument("--step", type=int, default=2)
    w.add_argument("--rep", type=int, required=True)
    w.add_argument("--phone", required=True)
    w.add_argument("--watch", required=True)
    for q in (p, sub.add_parser("pull-watch")):
        q.add_argument("--rep", type=int, required=True)
        q.add_argument("--phone", required=True)
        q.add_argument("--watch")
        q.add_argument("--conditions", default=",".join(CONDITIONS))
    p = sub.add_parser("summary")
    p.add_argument("--reps", default="1")
    a = ap.parse_args()
    {"run": cmd_run, "wait-start": cmd_wait_start, "pull-watch": cmd_pull_watch, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
