"""Phase 1/2: install, run and analyse the embedded benchmark on the watch and the phone (adb).

    python -m embedded.phase1_bench devices  # serials, roles, model, SoC, battery
    python -m embedded.phase1_bench install [--emulator]  # build and install the watch / phone apps
    python -m embedded.phase1_bench push  # replay files -> both devices
    python -m embedded.phase1_bench run --on watch --set condition=LOCAL pacing=fast steps=2,5,10 log_probs=true --wait
    python -m embedded.phase1_bench run --on both --set condition=P3 duration_s=3600 --wait
    python -m embedded.phase1_bench pull <run_id>
    python -m embedded.phase1_bench analyze <run_id>

Watch and phone are told apart by ro.build.characteristics ("watch"); --watch / --phone serials
override. Runs are started through ControlActivity (config as base64 JSON) and written by the apps
to <external files>/runs/<run_id>/<role>/; they are pulled to outputs/embedded/phase1/devices/<run_id>/.

analyze:
  * fidelity (a role that ran the whole pipeline: LOCAL, P3 watch, P1 phone): alarms, Gesture
    Timeline and (if logged) window probabilities against phase1_reference.py;
  * latency at the wrist (watch, replay): alarm time - moment the last intake sample was delivered,
    first alarm of the deployed step after the session start;
  * cost: inference time percentiles, CPU time, memory, link bytes, battery charge per hour.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from core import paths
from embedded import emb as E

PKG = "com.medmon.embbench"
DEV_FILES = f"/sdcard/Android/data/{PKG}/files"
ANDROID = paths.ANDROID
REPLAY = E.PHASE1 / "replay"
REFERENCE = E.PHASE1 / "reference"
DEVICES = E.PHASE1 / "devices"
LOOKAHEAD_S = 1.5


def adb(serial: str | None, *args, check=True, capture=True, timeout=None) -> str:
    """Run an adb command (on one device when serial is given) and return its output."""
    cmd = ["adb"] + (["-s", serial] if serial else []) + list(args)
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if check:
            raise SystemExit(f"{' '.join(cmd)}: timed out")
        return ""
    if check and r.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)}: {r.stderr.strip()}")
    return (r.stdout or "").strip()


def prop(serial, name):
    """A system property of a device."""
    return adb(serial, "shell", "getprop", name, check=False)


def devices(a) -> dict:
    """{"watch": serial, "phone": serial} of the connected devices (--watch / --phone override)."""
    out = adb(None, "devices").splitlines()[1:]
    serials = [l.split()[0] for l in out if l.strip().endswith("device")]
    roles = {}
    for s in serials:
        role = "watch" if "watch" in prop(s, "ro.build.characteristics") else "phone"
        roles.setdefault(role, s)
    if getattr(a, "watch", None):
        roles["watch"] = a.watch
    if getattr(a, "phone", None):
        roles["phone"] = a.phone
    return roles


def cmd_devices(a):
    """Print model, SoC, Android version and battery of each device."""
    for role, s in devices(a).items():
        info = {k: prop(s, p) for k, p in [("model", "ro.product.model"), ("soc", "ro.soc.model"),
                                            ("soc_vendor", "ro.soc.manufacturer"), ("abi", "ro.product.cpu.abi"),
                                            ("android", "ro.build.version.release"), ("sdk", "ro.build.version.sdk")]}
        bat = adb(s, "shell", "dumpsys", "battery", check=False)
        level = next((l.split(":")[1].strip() for l in bat.splitlines() if "level" in l), "?")
        print(f"{role:5s} {s}: {info}, battery {level} %")


def cmd_install(a):
    """Build the apps and install them, with the power-management exemptions the benchmark needs."""
    task = ["./gradlew", ":wear:assembleDebug", ":phone:assembleDebug", "--console=plain"]
    if a.emulator:
        task.append("-Pemulator")
    subprocess.run(task, cwd=ANDROID, check=True)
    for role, s in devices(a).items():
        apk = ANDROID / ("wear" if role == "watch" else "phone") / "build/outputs/apk/debug" / \
            f"{'wear' if role == 'watch' else 'phone'}-debug.apk"
        print(f"{role}: installing {apk.name}")
        adb(s, "install", "-r", "-g", str(apk))
        adb(s, "shell", "dumpsys", "deviceidle", "whitelist", f"+{PKG}")
        adb(s, "shell", "cmd", "appops", "set", PKG, "RUN_ANY_IN_BACKGROUND", "allow")


def cmd_push(a):
    """Replay files -> the app's internal storage (via /data/local/tmp and run-as: the app cannot read
    files that adb pushes into its Android/data directory)."""
    tmp = "/data/local/tmp/emb_replay"
    for role, s in devices(a).items():
        if a.only and role != a.only:
            continue
        adb(s, "shell", "mkdir", "-p", tmp)
        for f in sorted(REPLAY.glob("*.json")) + sorted(REPLAY.glob("*.f32")):
            print(f"{role}: {f.name}")
            adb(s, "push", str(f), f"{tmp}/{f.name}", capture=False)
        adb(s, "shell", f"chmod 644 {tmp}/* && run-as {PKG} sh -c 'mkdir -p files/replay && cp {tmp}/* files/replay/'"
            f" && rm -r {tmp}")
        print(f"{role}: {adb(s, 'shell', 'run-as', PKG, 'ls', '-l', 'files/replay')}")


def parse_value(v: str):
    """A --set value as bool, int list, int, float or string."""
    if v in ("true", "false"):
        return v == "true"
    if "," in v:
        return [int(x) for x in v.split(",")]
    try:
        return int(v)
    except ValueError:
        return v


def cmd_run(a):
    """Start a benchmark run on the chosen devices."""
    cfg = {k: parse_value(v) for k, v in (x.split("=", 1) for x in a.set)}
    if "steps" in cfg and isinstance(cfg["steps"], int):
        cfg["steps"] = [cfg["steps"]]
    cfg.setdefault("run_id", time.strftime("%Y%m%d-%H%M%S") + "_" + cfg.get("condition", "X"))
    b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
    roles = devices(a)
    targets = ["watch", "phone"] if a.on == "both" else [a.on]
    for role in sorted(targets, key=lambda r: r != "phone"):
        adb(roles[role], "shell", "am", "start", "-n", f"{PKG}/com.medmon.emb.bench.ControlActivity",
            "--es", "action", "start", "--es", "config_b64", b64)
        print(f"{role}: started {cfg['run_id']}")
    if a.wait:
        wait(roles, targets, cfg["run_id"], a.timeout)
        pull(roles, targets, cfg["run_id"])
        analyze(cfg["run_id"])


def wait(roles, targets, run_id, timeout):
    """Wait until every target device has written its run summary."""
    t0 = time.time()
    pending = set(targets)
    while pending and time.time() - t0 < timeout:
        for role in list(pending):
            if adb(roles[role], "shell", "ls", f"{DEV_FILES}/runs/{run_id}/{role}/summary.json", check=False) \
                    .endswith("summary.json"):
                pending.discard(role)
        if pending:
            time.sleep(10)
    if pending:
        print(f"timeout waiting for {pending}")


def pull(roles, targets, run_id):
    """Copy a run's files from the devices to phase1/devices/<run_id>/."""
    for role in targets:
        dst = DEVICES / run_id
        dst.mkdir(parents=True, exist_ok=True)
        adb(roles[role], "pull", f"{DEV_FILES}/runs/{run_id}/{role}", str(dst), check=False)
        print(f"{role}: pulled to {dst / role}")


def cmd_pull(a):
    """Pull a run from both devices."""
    roles = devices(a)
    pull(roles, [r for r in ("watch", "phone") if r in roles], a.run_id)


def cmd_stop(a):
    """Stop the running benchmark on every device."""
    for role, s in devices(a).items():
        adb(s, "shell", "am", "start", "-n", f"{PKG}/com.medmon.emb.bench.ControlActivity", "--es", "action", "stop",
            check=False)


def labels_to_paths(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Gesture Timeline per stream from the logged label runs."""
    out = {}
    for s, g in df.groupby("stream"):
        p = np.full(int(g.last_frame.max()) + 1, -1, np.int8)
        for r in g.itertuples():
            p[r.first_frame:r.last_frame + 1] = r.label
        out[int(s)] = p
    return out


def fidelity(d, cfg, index) -> dict:
    """Alarms, timeline and probabilities of a device run against the PC reference."""
    prec, name = cfg["precision"], cfg["set"]
    ref_al = json.loads((REFERENCE / prec / "alarms.json").read_text())[name]
    al = pd.read_csv(d / "alarms.csv")
    al = al[~al["where"].astype(str).str.startswith("rx_")]
    sids = [s["sid"] for s in index["streams"]]
    done = pd.read_csv(d / "events.csv").query("type == 'stream_end'").stream.astype(int).tolist()
    same = total = 0
    for i in done:
        for st in cfg.get("steps", [2]):
            for bud, want in ref_al[sids[i]][str(st)].items():
                got = al[(al.stream == i) & (al.step_s == st) & (al.budget == bud)].stream_time_s.astype(float).round(3).tolist()
                total += 1
                same += [round(x, 3) for x in want] == got
    res = {"streams": len(done), "alarm_lists_identical": same, "alarm_lists": total}
    refp = np.load(REFERENCE / prec / f"{name}_paths.npz")
    paths = labels_to_paths(pd.read_csv(d / "labels.csv"))
    agree = [np.mean(paths[i][:len(refp[sids[i]])] == refp[sids[i]]) for i in done if i in paths]
    res["timeline_frame_agreement"] = float(np.mean(agree)) if agree else None
    if (d / "probs.f32").exists():
        P = np.fromfile(d / "probs.f32", dtype="<f4").reshape(-1, 5)
        idx = pd.read_csv(d / "probs_index.csv")
        refq = np.load(REFERENCE / prec / f"{name}.npz")
        diffs, agr = [], []
        for r in idx.itertuples():
            q, w = P[r.first_row:r.first_row + r.n_windows], refq[sids[r.stream]]
            n = min(len(q), len(w))
            diffs.append(np.abs(q[:n] - w[:n]).max())
            agr.append(q[:n].argmax(1) == w[:n].argmax(1))
        res.update({"probs_max_abs_diff": float(max(diffs)), "argmax_agreement": float(np.mean(np.concatenate(agr)))})
    return res


def latency(d, cfg, index) -> dict:
    """Alarm latency at the wrist of a device run."""
    ev = pd.read_csv(d / "events.csv")
    al = pd.read_csv(d / "alarms.csv")
    step, bud = cfg.get("steps", [2])[0], "fa10"
    al = al[(al.step_s == step) & (al.budget == bud)]
    ends = ev[ev.type == "intake_end"].set_index("stream").elapsed_s
    wall, alg = [], []
    for i, t_end in ends.items():
        s = index["streams"][int(i)]
        a = al[(al.stream == i) & (al.stream_time_s >= s["session_start_s"])]
        if len(a):
            first = a.sort_values("elapsed_s").iloc[0]
            wall.append(first.elapsed_s - t_end)
            alg.append(first.stream_time_s + LOOKAHEAD_S - s["intake_end_s"])
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else None
    return {"budget": bud, "step_s": step, "detected": len(wall), "intakes_delivered": len(ends),
            "wall_median_s": q(wall, 50), "wall_p95_s": q(wall, 95),
            "algorithmic_median_s": q(alg, 50), "algorithmic_p95_s": q(alg, 95),
            "overhead_median_s": q(np.array(wall) - np.array(alg), 50) if wall else None}


def current_scale(col) -> float:
    """BatteryManager documents the current in µA, but Samsung devices report mA. A wearable or a
    phone never draws < 5 mA while running, so a median |value| below 5000 means mA."""
    return 1.0 if col.abs().median() < 5000 else 1000.0


def cost(d) -> dict:
    """Inference time, CPU time, memory, link traffic and battery of a device run."""
    s = json.loads((d / "summary.json").read_text())
    m = pd.read_csv(d / "metrics.csv")
    h = s["elapsed_s"] / 3600
    out = {"elapsed_h": h, "samples": s["samples"], "windows": s["windows"], "alarms": s["alarms"],
           "cpu_s_per_h": s["cpu_ms"] / 1000 / h if h else None,
           "inference_us": s.get("inference", {}), "stage_mean_us": {k: v["mean_us"] for k, v in s["stages"].items()},
           "pss_mb_max": float(m.pss_kb.max() / 1024), "pss_mb_mean": float(m.pss_kb.mean() / 1024),
           "tx_kib_per_h": s["tx_bytes"] / 1024 / h if h else None, "rx_kib_per_h": s["rx_bytes"] / 1024 / h if h else None}
    unplugged = m[m.plugged == 0]
    if len(unplugged) > 1 and h > 0:
        dq = unplugged.charge_uah.iloc[0] - unplugged.charge_uah.iloc[-1]
        dh = (unplugged.elapsed_s.iloc[-1] - unplugged.elapsed_s.iloc[0]) / 3600
        out.update({"charge_mah_per_h": dq / 1000 / dh if dh else None,
                    "battery_pct_per_h": (unplugged.battery_pct.iloc[0] - unplugged.battery_pct.iloc[-1]) / dh if dh else None,
                    "current_avg_ma": float(-unplugged.current_avg_ua.mean() / current_scale(unplugged.current_now_ua)),
                    "current_now_ma_median": float(-unplugged.current_now_ua.median()
                                                   / current_scale(unplugged.current_now_ua))})
    else:
        out["energy"] = "device was plugged in (no energy figures)"
    return out


def analyze(run_id) -> dict:
    """Fidelity, latency and cost of every role of a run -> analysis.json."""
    res = {}
    for d in sorted(p for p in (DEVICES / run_id).iterdir() if p.is_dir()):
        cfg = json.loads((d / "config.json").read_text())
        index = json.loads((REPLAY / f"{cfg['set']}.json").read_text())
        r = {"device": json.loads((d / "device.json").read_text()), "cost": cost(d)}
        role, cond = d.name, cfg["condition"]
        full = cond == "LOCAL" or (role == "watch" and cond == "P3") or (role == "phone" and cond in ("P1", "P2"))
        if full and cfg.get("source", "replay") == "replay":
            r["fidelity"] = fidelity(d, cfg, index)
        if role == "watch" and cfg.get("source", "replay") == "replay" and cfg.get("pacing") == "realtime" \
                and cond in ("P1", "P2", "P3", "LOCAL"):
            r["latency_at_wrist"] = latency(d, cfg, index)
        res[role] = r
    E.write_json(DEVICES / run_id / "analysis.json", res)
    print(json.dumps(res, indent=1, default=str))
    return res


def main():
    """Command-line interface."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch")
    ap.add_argument("--phone")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices")
    p = sub.add_parser("install")
    p.add_argument("--emulator", action="store_true")
    p = sub.add_parser("push")
    p.add_argument("--only", choices=["watch", "phone"])
    p = sub.add_parser("run")
    p.add_argument("--on", choices=["watch", "phone", "both"], required=True)
    p.add_argument("--set", nargs="+", default=[], help="config key=value (BenchConfig JSON names)")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--timeout", type=int, default=24 * 3600)
    p = sub.add_parser("pull")
    p.add_argument("run_id")
    sub.add_parser("stop")
    p = sub.add_parser("analyze")
    p.add_argument("run_id")
    a = ap.parse_args()
    {"devices": cmd_devices, "install": cmd_install, "push": cmd_push, "run": cmd_run, "pull": cmd_pull,
     "stop": cmd_stop, "analyze": lambda a: analyze(a.run_id)}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
