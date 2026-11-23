"""Sensor streams of every data source, resampled to a common rate in common units.

Each stream holds the accelerometer in g and the gyroscope in deg/s, a main label (gesture ids
G1..G5, or MASK where the main label is unknown) and an auxiliary label specific to its source
(MASK where undefined).

External sources:
  ingesture  InGesture (Mendeley Data, doi:10.17632/fdxst56tcj.5, CC BY 4.0): continuous ~10 min
             recordings at 200 Hz. Code 1 = fluid intake, 2 = answering phone, 3..7 = hand-to-face
             confounders, 0 = other.
  wisdm      WISDM 2019 (UCI, doi:10.24432/C5HK59, CC BY 4.0): 51 subjects x 18 activities x 3 min,
             smartwatch at 20 Hz, acc m/s^2, gyro rad/s.
The annotated gesture trials and lab sessions are loaded by ``core.streams``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

LABELS = ["G1", "G2", "G3", "G4", "G5"]
LABEL_TO_ID = {g: i for i, g in enumerate(LABELS)}
REST = LABEL_TO_ID["G5"]
MASK = -1
G = 9.80665
SENSORS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

PROTOCOLS = {
    "S01": (["G1", "G2", "G3", "G4"], 1),
    "S02": (["G1", "G4", "G2", "G3"], 1),
    "S03": (["G1", "G4"], 0),
    "S04": (["G1", "G2"], 0),
    "S05": (["G1", "G2", "G3"], 1),
}
INGESTURE_CODES = ["other", "fluid_intake", "answering_phone", "confounder_3", "confounder_4",
                   "confounder_5", "confounder_6", "confounder_7"]
WISDM_ACTIVITIES = {"A": "walking", "B": "jogging", "C": "stairs", "D": "sitting", "E": "standing",
                    "F": "typing", "G": "teeth", "H": "soup", "I": "chips", "J": "pasta",
                    "K": "drinking", "L": "sandwich", "M": "kicking", "O": "catch",
                    "P": "dribbling", "Q": "writing", "R": "clapping", "S": "folding"}
WISDM_CODES = list(WISDM_ACTIVITIES)


@dataclass
class Stream:
    """One continuous recording or composed stream.

    x: (T, 6) float32, acc in g and gyro in deg/s; y: (T,) main label id or MASK;
    aux: (T,) source-specific label or MASK.
    """

    sid: str
    source: str
    subject: str
    x: np.ndarray
    y: np.ndarray
    aux: np.ndarray
    protocol: str = ""
    intake: int = -1
    meta: dict = field(default_factory=dict)

    @property
    def gesture_sequence(self) -> list[str]:
        """Ordered non-rest gestures of the stream (see ``label_runs_to_sequence``)."""
        return label_runs_to_sequence(self.y)


def label_runs_to_sequence(y: np.ndarray) -> list[str]:
    """Ordered non-rest gestures, consecutive duplicates merged (masked samples ignored)."""
    seq: list[str] = []
    for lab in y:
        if lab >= 0 and lab != REST and (not seq or seq[-1] != LABELS[lab]):
            seq.append(LABELS[lab])
    return seq


def resample(x: np.ndarray, labels: list[np.ndarray], fs_in: float, fs_out: float,
             cutoff: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Zero-phase anti-alias low-pass and linear interpolation; labels by nearest sample."""
    n = len(x)
    if fs_in != fs_out and n > 30:
        sos = butter(4, cutoff, btype="low", fs=fs_in, output="sos")
        x = sosfiltfilt(sos, x, axis=0)
    t_in = np.arange(n) / fs_in
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fs_out)
    xo = np.stack([np.interp(t_out, t_in, x[:, c]) for c in range(x.shape[1])], 1)
    idx = np.clip(np.rint(t_out * fs_in).astype(int), 0, n - 1)
    return xo.astype(np.float32), [lab[idx].astype(np.int64) for lab in labels]


def to_units(acc_ms2: np.ndarray, gyr_dps: np.ndarray) -> np.ndarray:
    """(T, 6) array with the accelerometer converted from m/s^2 to g."""
    return np.concatenate([acc_ms2 / G, gyr_dps], 1)


def stretch(x: np.ndarray, factor: float) -> np.ndarray:
    """Resample a trial to factor x its length (factor > 1 gives a slower gesture)."""
    n = max(2, int(round(len(x) * factor)))
    t = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], 1).astype(np.float32)


def rotation(rng, max_deg: float) -> np.ndarray:
    """Random 3 x 3 rotation about a uniformly drawn axis by at most max_deg degrees."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.radians(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def load_ingesture(root: Path, fs: float, cutoff: float, drink_label: str) -> list[Stream]:
    """Continuous InGesture recordings.

    drink_label: 'G4' maps fluid intake to G4, 'mask' leaves it unlabelled. Every other sample
    is rest (G5); the original code is kept as the auxiliary label. The recordings are already
    in g and deg/s.
    """
    meta = pd.read_excel(root / "ingesture_metadata.xlsx")
    by_file = {r.file: r for r in meta.itertuples()}
    out = []
    for path in sorted(root.glob("subject_*/*p10min*.csv")):
        df = pd.read_csv(path, usecols=["accX", "accY", "accZ", "asX", "asY", "asZ", "label"])
        code = df["label"].to_numpy()
        y = np.full(len(code), REST)
        y[code == 1] = LABEL_TO_ID["G4"] if drink_label == "G4" else MASK
        raw = df[["accX", "accY", "accZ", "asX", "asY", "asZ"]].to_numpy(np.float64)
        x, (yr, ar) = resample(raw, [y, code], 200.0, fs, cutoff)
        r = by_file.get(path.name)
        subj = f"ingesture_p{int(r.participant_id):03d}" if r is not None else path.parent.name
        out.append(Stream(f"ingesture_{path.stem}", "ingesture", subj, x, yr, ar,
                          meta={"file": str(path), "hand": path.name.split("_")[2]}))
    return out


def _read_wisdm(path: Path) -> pd.DataFrame:
    """One raw WISDM sensor file (subject, activity, time in ns, x, y, z)."""
    df = pd.read_csv(path, header=None, names=["subject", "act", "t", "x", "y", "z"],
                     dtype={"act": str}, sep=",", lineterminator="\n")
    df["z"] = df["z"].astype(str).str.rstrip(";").astype(float)
    return df


def load_wisdm(root: Path, fs: float, cutoff: float, eating_as_rest: bool) -> list[Stream]:
    """One stream per subject x activity of the WISDM watch data.

    The gyroscope is interpolated onto the accelerometer clock and both onto a uniform grid at
    fs on the device clock. Streams with a gap longer than 1 s are dropped. Drinking is
    unlabelled (MASK); eating is rest when eating_as_rest, otherwise unlabelled; every other
    activity is rest.
    """
    out = []
    eating = {"H", "I", "J", "L"}
    for acc_path in sorted((root / "raw/watch/accel").glob("data_*_accel_watch.txt")):
        subj = acc_path.name.split("_")[1]
        acc = _read_wisdm(acc_path)
        gyr = _read_wisdm(root / f"raw/watch/gyro/data_{subj}_gyro_watch.txt")
        for act, a in acc.groupby("act"):
            g = gyr[gyr.act == act]
            if act not in WISDM_ACTIVITIES or len(a) < 100 or len(g) < 100:
                continue
            a = a.sort_values("t").drop_duplicates("t")
            g = g.sort_values("t").drop_duplicates("t")
            t = a["t"].to_numpy(np.float64) * 1e-9
            tg = g["t"].to_numpy(np.float64) * 1e-9
            keep = (t >= tg[0]) & (t <= tg[-1])
            t = t[keep]
            if len(t) < 100:
                continue
            accv = a[["x", "y", "z"]].to_numpy(np.float64)[keep]
            gyrv = np.stack([np.interp(t, tg, g[c].to_numpy(np.float64)) for c in "xyz"], 1)
            grid = np.arange(t[0], t[-1], 1.0 / fs)
            if np.max(np.diff(t)) > 1.0:
                continue
            xs = np.stack([np.interp(grid, t, v) for v in np.concatenate([accv, gyrv], 1).T], 1)
            x = to_units(xs[:, :3], np.degrees(xs[:, 3:])).astype(np.float32)
            if act == "K" or (act in eating and not eating_as_rest):
                y = np.full(len(x), MASK)
            else:
                y = np.full(len(x), REST)
            aux = np.full(len(x), WISDM_CODES.index(act))
            out.append(Stream(f"wisdm_{subj}_{act}", "wisdm", f"wisdm_{subj}", x, y, aux,
                              meta={"activity": WISDM_ACTIVITIES[act]}))
    return out


def split_external(streams: list[Stream], test_fraction: float, seed: int) -> tuple[set, set]:
    """Participant-level split of an external dataset into (train, held-out) subject sets."""
    subjects = sorted({s.subject for s in streams})
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(subjects) * test_fraction)))
    test = set(rng.choice(subjects, n_test, replace=False).tolist())
    return set(subjects) - test, test


def window_and_stride(data_cfg: dict) -> tuple[int, int]:
    """Window length and inference stride in samples.

    ``frame_stride`` (samples) fixes the posterior frame rate independently of the window
    length; otherwise the stride follows from ``overlap``.
    """
    win = int(round(data_cfg["window_s"] * data_cfg["fs"]))
    if data_cfg.get("frame_stride"):
        return win, int(data_cfg["frame_stride"])
    return win, max(1, int(round(win * (1 - data_cfg["overlap"]))))


def causal_sos(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Forward-only filtering of (T, C), initial state = steady state for the first sample."""
    zi = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    return sosfilt(sos, x, axis=0, zi=zi)[0]


def apply_post_filter(streams: list[Stream], spec: dict, fs: float) -> list[Stream]:
    """Preprocessing filter applied to whole streams.

    spec:
      {"type": "lowpass", "cutoff": Hz}: low-pass on all channels;
      {"type": "highpass_acc", "cutoff": Hz}: high-pass on the accelerometer only (removes
          gravity, keeps the dynamic acceleration; the gyroscope is left unchanged);
      {"type": "bandpass_acc", "low": Hz, "high": Hz}: band-pass on the accelerometer and the
          low-pass alone on the gyroscope.
    The filter is zero-phase (forward and backward) unless "causal": true, which filters
    forward only, as a device must, starting in the steady state of the first sample so that a
    constant input gives no transient.
    """
    filt = causal_sos if spec.get("causal") else (lambda sos, x: sosfiltfilt(sos, x, axis=0))
    if spec["type"] == "bandpass_acc":
        lo, hi = float(spec["low"]), float(spec["high"])
        if hi >= fs / 2:
            raise ValueError(f"band-pass upper edge {hi} Hz must be below Nyquist ({fs / 2} Hz)")
        sos_a = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
        sos_g = butter(4, hi, btype="low", fs=fs, output="sos")
        out = []
        for s in streams:
            x = s.x.astype(np.float64)
            if len(x) > 30:
                x[:, :3], x[:, 3:] = filt(sos_a, x[:, :3]), filt(sos_g, x[:, 3:])
            out.append(dataclasses.replace(s, x=x.astype(np.float32)))
        return out
    kind, fc = spec["type"], float(spec["cutoff"])
    if kind not in ("lowpass", "highpass_acc"):
        raise KeyError(kind)
    if kind == "lowpass" and fc >= fs / 2:
        raise ValueError(f"low-pass cut-off {fc} Hz must be below Nyquist ({fs / 2} Hz)")
    sos = butter(4, fc, btype="low" if kind == "lowpass" else "high", fs=fs, output="sos")
    out = []
    for s in streams:
        x = s.x.astype(np.float64)
        if len(x) > 30:
            if kind == "lowpass":
                x = filt(sos, x)
            else:
                x[:, :3] = filt(sos, x[:, :3])
        out.append(dataclasses.replace(s, x=x.astype(np.float32)))
    return out
