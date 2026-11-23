"""Decoders of stage D1 on a frozen classifier .

All work on the 10 Hz Gesture Timeline probabilities of a replicate (dcommon.Rep.timelines) and
learn only from the validation streams of that replicate.

* ``mode``: majority vote of the per-frame argmax over a centred window (grid MODE_S, chosen on
  validation per configuration, like the E3 grids).
* ``crf`` / ``crfdur``: the E3 HMM graph trained discriminatively, i.e. a hidden-state
  CRF: the segmentation loss -log P(label path | x) of core.hmm_torch.HMMLayer, with a
  learned emission scale. ``crf`` has no minimum-duration chains (one state per gesture);
  ``crfdur`` keeps the chains of the configuration's selected HMM. Initialised at the selected HMM.
* ``refiner``: a dilated temporal CNN over the log-probabilities (MS-TCN single stage), trained
  with cross-entropy + truncated MSE smoothing, decoded by argmax.

Learned decoders split the validation streams in two halves (seeded by the replicate): fit on one,
choose the number of steps on the other, then refit on all validation streams for that number of
steps. Fixed choices (pre-registered here): chunks of 60 s starting at rest frames, batch 32,
Adam, CRF lr 0.03 for up to 200 steps, refiner lr 1e-3 for up to 600 steps. ``crfdur`` (about 70
states instead of 5) uses 30 s chunks and up to 100 steps, to keep it near 5 min per replicate.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from torch import nn

from decoders import dcommon as D
from decoders.dcommon import PP

from core import hmm_torch as HT
from core.hmm import Durations, PostProcessor, viterbi_batch

HZ, K, REST = PP.HZ, PP.K, PP.LABELS.index("G5")
MODE_S = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
CHUNK, BATCH = 600, 32
CRF_LR, CRF_STEPS, CRF_EVAL, CRF_REG = 0.03, 200, 25, 1e-3
CRFDUR_CHUNK, CRFDUR_STEPS = 300, 100
REF_LR, REF_STEPS, REF_EVAL, TMSE_W = 1e-3, 600, 50, 0.15
torch.set_num_threads(1)


def mode_filter(tl: np.ndarray, win_s: float) -> np.ndarray:
    """Majority vote of the per-frame argmax over a centred window of win_s seconds."""
    n = max(1, int(round(win_s * HZ)) | 1)
    onehot = np.eye(K)[tl.argmax(1)]
    return uniform_filter1d(onehot, n, axis=0, mode="nearest").argmax(1)


def chunks(tls, ys, rng, n, size=CHUNK):
    """n random chunks starting at a rest frame (so the label path can start in the graph's start)."""
    xs, yy, ls = [], [], []
    cand = [(i, s) for i, y in enumerate(ys) for s in np.flatnonzero(y == REST)[::10]]
    for j in rng.integers(0, len(cand), n):
        i, s = cand[j]
        e = min(len(ys[i]), s + size)
        xs.append(tls[i][s:e])
        yy.append(ys[i][s:e])
        ls.append(e - s)
    T = max(ls)
    X = np.zeros((n, T, K), np.float32)
    Y = np.full((n, T), -1, np.int64)
    for b, (x, y) in enumerate(zip(xs, yy)):
        X[b, :len(x)], Y[b, :len(y)] = x, y
    return torch.from_numpy(X), torch.from_numpy(Y), torch.tensor(ls)


def split(n, seed):
    """Two halves of n indices, permuted with the seed."""
    idx = np.random.default_rng(seed).permutation(n)
    return idx[: n // 2], idx[n // 2:]


def fit_early_stopped(make, step_fn, eval_fn, tls, ys, seed, max_steps, every, size=CHUNK):
    """Fit on half A, pick the step count by the loss on half B, refit on all for that count."""
    a, b = split(len(tls), seed)
    rng = np.random.default_rng(seed + 1)
    ev = chunks([tls[i] for i in b], [ys[i] for i in b], rng, 4 * BATCH, size)
    model = make()
    best = (math.inf, every)
    for s in range(1, max_steps + 1):
        step_fn(model, chunks([tls[i] for i in a], [ys[i] for i in a], rng, BATCH, size))
        if s % every == 0:
            best = min(best, (eval_fn(model, ev), s))
    model = make()
    rng = np.random.default_rng(seed + 2)
    for _ in range(best[1]):
        step_fn(model, chunks(tls, ys, rng, BATCH, size))
    return model, {"steps": best[1], "heldout_loss": best[0]}


class CRF(nn.Module):
    """variant: "base" (pre-registered); post-hoc after D1 (the emission scale collapsed on E1):
    "bias" adds a per-class additive emission bias outside the scale; "fixs" freezes the scale at
    the selected HMM's 1/temperature."""

    def __init__(self, R: D.Rep, hp: dict, chains: bool, variant: str = "base"):
        super().__init__()
        self.variant = variant
        self.dur = R.durations(hp["md"] if chains else 0.0)
        self.layer = HT.HMMLayer(R.post_cfg(hp["st"]), self.dur, R.prior, HZ)
        self.layer.bias.requires_grad_(False)
        for k in ("log_g3_entry", "log_g4_after_g3", "log_repeat_g3", "log_eps_g3"):
            getattr(self.layer, k).requires_grad_(False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / hp["t"])))
        self.register_buffer("log_scale_0", self.log_scale.detach().clone())
        self.cbias = nn.Parameter(torch.zeros(K), requires_grad=variant == "bias")
        if variant == "fixs":
            self.log_scale.requires_grad_(False)
        self.opt = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=CRF_LR)

    def e(self, X):
        """Learned emissions: scale x scaled likelihoods + class bias."""
        lp = torch.log(X.clamp(min=self.layer.floor))
        return torch.exp(self.log_scale) * (lp - torch.log_softmax(self.layer.log_prior, 0)) + self.cbias

    def loss(self, X, Y, L):
        """Mean segmentation negative log-likelihood of a batch."""
        nll, ok = self.layer.seg_nll(self.e(X), L, Y)
        return nll[ok].mean() if ok.any() else nll.sum() * 0

    def reg(self):
        """L2 regulariser towards the initial graph, scale and zero bias."""
        return self.layer.reg() + (self.log_scale - self.log_scale_0) ** 2 + (self.cbias ** 2).sum()


def crf_step(m: CRF, batch):
    """One optimiser step of a CRF."""
    m.opt.zero_grad()
    (m.loss(*batch) + CRF_REG * m.reg()).backward()
    m.opt.step()


@torch.no_grad()
def crf_eval(m: CRF, batch):
    """Loss of a CRF on a batch."""
    return float(m.loss(*batch))


def train_crf(R: D.Rep, hp: dict, chains: bool, seed: int, variant: str = "base"):
    """Train a CRF on the replicate's validation streams with early stopping on a held-out half."""
    tls = [t.astype(np.float32) for t in R.timelines("val", hp["ov"])]
    ys = R.truths("val")
    torch.manual_seed(seed)
    steps, size = (CRFDUR_STEPS, CRFDUR_CHUNK) if chains else (CRF_STEPS, CHUNK)
    return fit_early_stopped(lambda: CRF(R, hp, chains, variant), crf_step, crf_eval, tls, ys, seed, steps, CRF_EVAL, size)


@torch.no_grad()
def crf_export(m: CRF) -> dict:
    """Learned parameters of a CRF as post-processing settings."""
    ex = m.layer.export(m.dur)
    return {**ex, "scale": float(torch.exp(m.log_scale)), "cbias": m.cbias.cpu().numpy().tolist()}


def crf_post(R: D.Rep, hp: dict, ex: dict) -> tuple[dict, np.ndarray]:
    """Post-processing config and prior of an exported CRF."""
    cfg = HT.merge_post(R.post_cfg(hp["st"]), ex["post_overrides"])
    return cfg, np.asarray(ex["prior"])


def crf_decode(R: D.Rep, part: str, hp: dict, ex: dict):
    """Viterbi paths of a set with the exported CRF parameters."""
    cfg, prior = crf_post(R, hp, ex)
    dur = Durations(np.asarray(ex["durations"]["min_frames"]), np.asarray(ex["durations"]["p_stay"]))
    pp = PostProcessor(dur, prior, cfg, HZ)
    cb = np.asarray(ex.get("cbias", np.zeros(K)))
    logs = [(np.log(np.clip(t, cfg["emission_floor"], 1)) - np.log(prior)) * ex["scale"] + cb
            for t in R.timelines(part, hp["ov"])]
    return viterbi_batch(pp.seg, logs)


def crf_llr_scores(R: D.Rep, part: str, hp: dict, ex: dict) -> list[np.ndarray]:
    """The E3 phased LLR (Replicate.llr_scores) with the CRF's learned edge weights, background,
    start, prior and emission scale. The 2 Hz chains keep the E3 durations (the learned 10 Hz
    self-loops do not transfer to another frame rate)."""
    cfg, prior = crf_post(R, hp, ex)
    r = HZ // PP.LLR_HZ
    tls = [t[:len(t) // r * r].reshape(-1, r, K).mean(1) if len(t) >= r else t.mean(0, keepdims=True)
           for t in R.timelines(part, hp["ov"])]
    a, cb = ex["scale"] * r, np.asarray(ex.get("cbias", np.zeros(K))) * r
    pp = PostProcessor(R.durations(hp["md"], PP.LLR_HZ), prior, cfg, PP.LLR_HZ)
    pp.emissions = lambda post: (np.log(np.clip(post, cfg["emission_floor"], 1)) - np.log(prior)) * a + cb
    W, step = int(cfg["episode_s"] * PP.LLR_HZ), int(cfg["episode_step_s"] * PP.LLR_HZ)
    eps, owner = [], []
    for i, t in enumerate(tls):
        starts = list(range(0, max(1, len(t) - W + 1), step))
        if len(t) > W and starts[-1] + W < len(t):
            starts.append(len(t) - W)
        eps += [t[s:s + W] for s in starts]
        owner += [i] * len(starts)
    owner = np.array(owner)
    sc = pp.llrs_batch(eps)["llr_phased"]
    return [sc[owner == i] for i in range(len(tls))]


class Refiner(nn.Module):
    """Single-stage dilated temporal CNN over log-probabilities (MS-TCN style)."""
    def __init__(self, ch=32, layers=6):
        super().__init__()
        self.inp = nn.Conv1d(K, ch, 1)
        self.layers = nn.ModuleList(nn.Conv1d(ch, ch, 3, padding=2 ** i, dilation=2 ** i) for i in range(layers))
        self.out = nn.Conv1d(ch, K, 1)
        self.opt = torch.optim.Adam(self.parameters(), lr=REF_LR)

    def forward(self, X):
        """(B, T, K) probabilities -> (B, T, K) logits."""
        h = self.inp(torch.log(X.clamp(min=1e-6)).transpose(1, 2))
        for l in self.layers:
            h = h + torch.relu(l(h))
        return self.out(h).transpose(1, 2)

    def loss(self, X, Y, L):
        """Cross-entropy plus truncated MSE smoothing."""
        z = self(X)
        ce = nn.functional.cross_entropy(z.reshape(-1, K), Y.reshape(-1), ignore_index=-1)
        lp = torch.log_softmax(z, -1)
        d = (lp[:, 1:] - lp[:, :-1].detach()) ** 2
        mask = (Y[:, 1:] >= 0)[..., None]
        tmse = (d.clamp(max=16) * mask).sum() / mask.sum().clamp(min=1) / K
        return ce + TMSE_W * tmse


def ref_step(m: Refiner, batch):
    """One optimiser step of the refiner."""
    m.opt.zero_grad()
    m.loss(*batch).backward()
    m.opt.step()


@torch.no_grad()
def ref_eval(m: Refiner, batch):
    """Loss of the refiner on a batch."""
    return float(m.loss(*batch))


def train_refiner(R: D.Rep, ov: int, seed: int):
    """Train the refiner on the replicate's validation streams with early stopping."""
    tls = [t.astype(np.float32) for t in R.timelines("val", ov)]
    torch.manual_seed(seed)
    return fit_early_stopped(Refiner, ref_step, ref_eval, tls, R.truths("val"), seed, REF_STEPS, REF_EVAL)


@torch.no_grad()
def refiner_decode(m: Refiner, R: D.Rep, part: str, ov: int):
    """Argmax paths of the refiner on a set."""
    return [m(torch.from_numpy(t.astype(np.float32))[None])[0].argmax(-1).numpy() for t in R.timelines(part, ov)]
