"""Annotated data and composition of continuous streams.

* ``load_gesture_trials``: data/labeled_output_gestures/<Subject>/<G1..G5>/*.csv, one isolated
  gesture per file, recorded at 100 Hz.
* ``load_lab_sessions``: data/labeled_output/<Subject>/labeled_*_<intake>_<protocol>.csv,
  continuous lab protocol sessions (S01-S05, and S02M = G3 G4). They are a test set only: never
  trained on and never used for a choice. The true sequence is the one in the labels (a few
  sessions deviate from their protocol).
* ``compose_world``: synthetic streams built from the gesture trials of a set of subjects,
  placed in daily-activity background from the external datasets (WISDM, InGesture) according
  to a "world" configuration (scenario recipes, gaps, context length, repeats, episodes).
* ``embed_in_background``: lab sessions placed inside unseen background (test_wild).

Everything is composed after the preprocessing filter (0.3 Hz high-pass on the accelerometer),
so the junctions between pieces of different recordings carry no gravity steps.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core import sources as S
from core.sources import LABEL_TO_ID, MASK, REST, SENSORS, Stream

FS_RAW = 100.0
LAB_RE = re.compile(r"labeled_(subject\d+)_(\d{8}-\d{6})_([01])_(S\d{2}M?)\.csv$")
PROTOCOLS = {**S.PROTOCOLS, "S02M": (["G3", "G4"], 1)}

RECIPES = {
    "full": (["G1", "G2", "G3", "G4"], 1),
    "water_first": (["G1", "G4", "G2", "G3"], 1),
    "no_water": (["G1", "G2", "G3"], 1),
    "pill_out": (["G2", "G3", "G4"], 1),
    "merged": (["G3", "G4"], 1),
    "dry": (["G3"], 1),
    "open_drink": (["G1", "G4"], 0),
    "open_pick": (["G1", "G2"], 0),
    "drink_only": (["G4"], 0),
    "open_only": (["G1"], 0),
    "pick_only": (["G2"], 0),
}
"""Scenario recipes: gesture order and intake (a pill reaches the mouth, i.e. G3 is present)."""
LAB_RECIPES = ["full", "water_first", "open_drink", "open_pick", "no_water"]
"""The five lab protocols S01-S05 as recipes."""


@dataclass
class Trial:
    """One isolated gesture: x is (T, 6), acc in g and gyro in deg/s."""

    subject: str
    gesture: str
    x: np.ndarray


def load_gesture_trials(root: Path, fs: float, cutoff: float) -> list[Trial]:
    """Every isolated gesture trial, resampled to fs (files with NaN or < 10 rows are skipped)."""
    out = []
    for f in sorted(root.glob("Subject*/G*/*.csv")):
        subj, g = f.parent.parent.name, f.parent.name
        if g not in LABEL_TO_ID:
            continue
        df = pd.read_csv(f, usecols=SENSORS)
        if df.isna().any().any() or len(df) < 10:
            continue
        raw = df.to_numpy(np.float64)
        x, _ = S.resample(S.to_units(raw[:, :3], raw[:, 3:]), [], FS_RAW, fs, cutoff)
        out.append(Trial(f"g_{subj}", g, x))
    return out


def load_lab_sessions(root: Path, fs: float, cutoff: float) -> list[Stream]:
    """Lab protocol sessions with their sample labels; meta["protocol_match"] tells whether the
    labelled sequence equals the protocol in the file name."""
    out = []
    for f in sorted(root.glob("Subject*/labeled_*.csv")):
        m = LAB_RE.search(f.name)
        if not m:
            continue
        subj, ts, intake, proto = m.groups()
        df = pd.read_csv(f, usecols=SENSORS + ["Gesto"])
        df = df.dropna(subset=SENSORS)
        y = df["Gesto"].map(LABEL_TO_ID).fillna(MASK).astype(np.int64).to_numpy()
        raw = df[SENSORS].to_numpy(np.float64)
        x, (yr,) = S.resample(S.to_units(raw[:, :3], raw[:, 3:]), [y], FS_RAW, fs, cutoff)
        s = Stream(f"lab_{subj}_{ts}_{proto}", "lab", f"lab_{subj.capitalize()}", x, yr, np.full(len(yr), MASK),
                   protocol=proto, intake=int(intake), meta={"file": f.name})
        s.meta["protocol_match"] = s.gesture_sequence == PROTOCOLS[proto][0]
        out.append(s)
    return out


def filter_trials(trials: list[Trial], spec: dict | None, fs: float) -> list[Trial]:
    """The preprocessing filter applied to every trial (no-op when spec is empty)."""
    if not spec:
        return trials
    st = [Stream(str(i), "t", t.subject, t.x, np.zeros(len(t.x), np.int64), np.zeros(len(t.x), np.int64))
          for i, t in enumerate(trials)]
    return [dataclasses.replace(t, x=s.x) for t, s in zip(trials, S.apply_post_filter(st, spec, fs))]


def join(pieces: list[tuple[np.ndarray, np.ndarray]], cf: int):
    """Concatenate (x, y) pieces with a linear cross-fade of cf samples at each junction."""
    x, y = pieces[0][0].astype(np.float32), pieces[0][1].astype(np.int64)
    for p, lab in pieces[1:]:
        c = max(1, min(cf, len(x) - 1, len(p) - 1))
        w = np.linspace(0, 1, c + 2)[1:-1, None].astype(np.float32)
        x = np.concatenate([x[:-c], x[-c:] * (1 - w) + p[:c] * w, p[c:]])
        y = np.concatenate([y[:-c], y[-c:][: c // 2], lab[: c - c // 2], lab[c:]])
    return x, y


class BackgroundSampler:
    """Random excerpts (with their own labels) of external daily-activity streams, drawn with
    probability proportional to the stream length."""

    def __init__(self, streams: list[Stream], rng):
        self.streams = [s for s in streams if len(s.x) > 50]
        self.w = np.array([len(s.x) for s in self.streams], float)
        self.w /= self.w.sum()
        self.rng = rng

    def take(self, n: int):
        """(x, y) excerpt of n samples, or a whole stream if it is shorter."""
        s = self.streams[self.rng.choice(len(self.streams), p=self.w)]
        if len(s.x) <= n:
            return s.x.copy(), s.y.copy()
        a = int(self.rng.integers(0, len(s.x) - n))
        return s.x[a:a + n], s.y[a:a + n]


def compose_world(trials: list[Trial], background: list[Stream], world: dict, fs: float, seed: int,
                  prefix: str) -> list[Stream]:
    """Synthetic streams following the world configuration.

    world keys:
      recipes: "lab" (the 5 lab protocols) or "all" (RECIPES); recipe_weights: optional {name: w}
      n_passes: each gesture trial is used about this many times
      context_s: [lo, hi] background before and after each stream, or null for rest (0.5-2 s)
      gap: {"rest": p, "background": p, "none": p} between consecutive gestures
      gap_rest_s, gap_background_s: duration ranges of the gaps
      repeat_g4: probability of 1-2 extra sips (G4) after a G4
      episodes: [lo, hi] gesture episodes per stream, separated by episode_gap_s of background
      stretch, amplitude, rotation_deg, crossfade_s: augmentation
    The composition is a deterministic function of the seed and of the stream lengths.
    """
    rng = np.random.default_rng(seed)
    bg = BackgroundSampler(background, rng) if background else None
    by_subj: dict[str, dict[str, list[np.ndarray]]] = {}
    for t in trials:
        by_subj.setdefault(t.subject, {}).setdefault(t.gesture, []).append(t.x)
    names = LAB_RECIPES if world.get("recipes", "lab") == "lab" else list(RECIPES)
    rw = world.get("recipe_weights") or {}
    weights = np.array([rw.get(n, 1.0) for n in names], float)
    cf = max(2, int(round(world.get("crossfade_s", 0.1) * fs)))
    gap = world.get("gap", {"rest": 0.85, "background": 0.0, "none": 0.15})
    gap_kinds, gap_p = list(gap), np.array(list(gap.values()), float) / sum(gap.values())
    n_gest = sum(len(v) for d in by_subj.values() for g, v in d.items() if g != "G5")
    mean_len = np.mean([len(RECIPES[n][0]) for n in names])
    ep_lo, ep_hi = world.get("episodes", [1, 1])
    n_streams = int(np.ceil(world.get("n_passes", 3) * n_gest / mean_len / ((ep_lo + ep_hi) / 2)))
    subjects = sorted(by_subj)

    def rest(subj, lo, hi):
        pool = by_subj[subj].get("G5") or [t.x for t in trials if t.gesture == "G5"]
        tr = pool[rng.integers(len(pool))]
        n = min(len(tr), max(2, int(rng.uniform(lo, hi) * fs)))
        a = int(rng.integers(0, len(tr) - n + 1))
        return tr[a:a + n], np.full(n, REST)

    def background_piece(lo, hi):
        n = max(2, int(rng.uniform(lo, hi) * fs))
        return bg.take(n)

    def context():
        if world.get("context_s") and bg is not None:
            return background_piece(*world["context_s"])
        return rest(subj, 0.5, 2.0)

    def gesture(subj, g):
        tr = by_subj[subj][g][rng.integers(len(by_subj[subj][g]))]
        tr = S.stretch(tr, float(np.exp(rng.uniform(*np.log(world.get("stretch", [0.8, 1.25]))))))
        amp = rng.uniform(*world.get("amplitude", [0.85, 1.2]))
        return (tr * amp).astype(np.float32), np.full(len(tr), LABEL_TO_ID[g])

    out = []
    for k in range(n_streams):
        subj = subjects[rng.integers(len(subjects))]
        have = set(by_subj[subj])
        ok = [i for i, n in enumerate(names) if set(RECIPES[n][0]) <= have]
        if not ok:
            continue
        pieces, intake, recipes = [context()], 0, []
        n_ep = int(rng.integers(ep_lo, ep_hi + 1))
        for e in range(n_ep):
            if e > 0:
                pieces.append(background_piece(*world.get("episode_gap_s", [10, 60])) if bg is not None
                              else rest(subj, 0.5, 2.0))
            w = weights[ok] / weights[ok].sum()
            name = names[ok[rng.choice(len(ok), p=w)]]
            seq = list(RECIPES[name][0])
            if "G4" in seq and rng.random() < world.get("repeat_g4", 0.0):
                i = seq.index("G4")
                seq[i + 1:i + 1] = ["G4"] * int(rng.integers(1, 3))
            recipes.append(name)
            intake |= RECIPES[name][1]
            for i, g in enumerate(seq):
                if i > 0:
                    kind = gap_kinds[rng.choice(len(gap_kinds), p=gap_p)]
                    if kind == "rest":
                        pieces.append(rest(subj, *world.get("gap_rest_s", [0.5, 2.0])))
                    elif kind == "background" and bg is not None:
                        pieces.append(background_piece(*world.get("gap_background_s", [1, 12])))
                pieces.append(gesture(subj, g))
        pieces.append(context())
        x, y = join(pieces, cf)
        R = S.rotation(rng, world.get("rotation_deg", 25)).astype(np.float32)
        x = np.concatenate([x[:, :3] @ R.T, x[:, 3:] @ R.T], 1).astype(np.float32)
        out.append(Stream(f"{prefix}_{k:05d}", "world", subj, x, y, np.full(len(y), MASK),
                          protocol="+".join(recipes), intake=int(intake)))
    return out


def embed_in_background(sessions: list[Stream], background: list[Stream], fs: float, seed: int,
                        context_s=(10, 60), crossfade_s=0.2) -> list[Stream]:
    """Lab sessions placed between two background excerpts (fixed seed, unseen participants).

    meta holds the session id, the offset of the session inside the stream and its length.
    """
    rng = np.random.default_rng(seed)
    bg = BackgroundSampler(background, rng)
    cf = max(2, int(round(crossfade_s * fs)))
    out = []
    for s in sessions:
        a = bg.take(int(rng.uniform(*context_s) * fs))
        b = bg.take(int(rng.uniform(*context_s) * fs))
        x, y = join([a, (s.x, s.y), b], cf)
        out.append(Stream(f"wild_{s.sid}", "lab_wild", s.subject, x.astype(np.float32), y, np.full(len(y), MASK),
                          protocol=s.protocol, intake=s.intake,
                          meta={"session": s.sid, "offset": len(a[0]), "length": len(s.x)}))
    return out
