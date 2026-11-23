"""Differentiable HMM layer: the post-processing graphs of ``core.hmm`` with learnable parameters.

The graphs are built by the same topology
functions: the ergodic segmentation graph and the phased intake / no-intake pair. Only the
handful of scalars that define them become parameters (log / logit space):

* ``base_weights``: 5 x 5 gesture-to-gesture edge weights shared by every graph (init 1, 3 into rest);
* ``p_stay``: self-loop of the last sub-state of G1-G4 (init: Durations fitted on the training streams);
* ``background_expected_s``, ``start_in_background``;
* phased: ``g3_entry`` (none / g1 / g2), ``g4_after_g3``, ``repeat_g3_weight``, ``no_intake_g3_prob``;
* ``log_prior``: the class prior dividing the posteriors (scaled-likelihood emissions).

The minimum durations (sub-state chains) stay fixed. Structural zeros stay zeros.

Two quantities, both by the forward algorithm (log space, differentiable):

* ``llr``: log P(x | phased intake) - log P(x | phased no-intake), the intake score of ``core.hmm``;
* ``seg_nll``: -log P(label path | x) on the ergodic graph = -(forward with the emissions
  restricted to the labelled classes - unrestricted forward), the MMI / CRF segmentation loss.

``export`` turns the parameters back into the post-processing config (cfg["post"] overrides,
durations, prior), so post-processing runs the unchanged numpy code with the learned values.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from core.hmm import Durations, ergodic_topology, phased_intake_topology, phased_no_intake_topology
from core.sources import LABELS

NEG = -1e4
K = len(LABELS)
PHASES = ("none", "g1", "g2")


def _logit(p):
    """log(p / (1 - p))."""
    return math.log(p / (1 - p))


class TorchGraph:
    """An expanded topology whose transition log-probabilities are torch tensors."""

    def __init__(self, logA, log_start, final, state_class):
        self.logA, self.log_start, self.final, self.state_class = logA, log_start, final, state_class


def expand_torch(top, min_frames: np.ndarray, p_stay, device) -> TorchGraph:
    """core.hmm.expand with tensor-valued weights (same state layout, same normalisation)."""
    first, last, cls_of, node_of = {}, {}, [], []
    for name, k in top.nodes:
        first[name] = len(cls_of)
        for _ in range(int(min_frames[k])):
            cls_of.append(k)
            node_of.append(name)
        last[name] = len(cls_of) - 1
    S = len(cls_of)
    entries: dict[tuple[int, int], object] = {}

    def add(i, j, v):
        entries[(i, j)] = entries[(i, j)] + v if (i, j) in entries else v

    one = torch.ones((), device=device)
    for name, k in top.nodes:
        for s in range(first[name], last[name]):
            add(s, s + 1, one)
        p = top.stay.get(name, p_stay[k])
        out = top.edges.get(name, {})
        tot = sum(out.values()) if out else 0.0
        if not out:
            add(last[name], last[name], one)
            continue
        add(last[name], last[name], p)
        for dst, w in out.items():
            add(last[name], first[dst], (1 - p) * w / tot)
    ij = torch.tensor(list(entries), device=device, dtype=torch.long)
    vals = torch.stack([torch.as_tensor(v, device=device, dtype=torch.float32).reshape(()) for v in entries.values()])
    logA = torch.full((S, S), NEG, device=device).index_put((ij[:, 0], ij[:, 1]), torch.log(vals))
    st_tot = sum(top.start.values())
    ls = torch.full((S,), NEG, device=device)
    idx = torch.tensor([first[n] for n in top.start], device=device)
    sv = torch.stack([torch.as_tensor(w, device=device, dtype=torch.float32).reshape(()) for w in top.start.values()])
    log_start = ls.index_put((idx,), torch.log(sv / st_tot))
    final = torch.ones(S, dtype=torch.bool, device=device) if top.final is None else \
        torch.tensor(np.isin(node_of, top.final), device=device)
    return TorchGraph(logA, log_start, final, torch.tensor(cls_of, device=device))


def forward_loglik(g: TorchGraph, e: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """log P(x | graph) per sequence. e: (B, T, C) log emissions per class, lengths: (B,)."""
    es = e[:, :, g.state_class]
    alpha = g.log_start[None] + es[:, 0]
    for t in range(1, es.shape[1]):
        nxt = torch.logsumexp(alpha[:, :, None] + g.logA[None], dim=1) + es[:, t]
        alpha = torch.where((t < lengths)[:, None], nxt, alpha)
    return torch.logsumexp(alpha.masked_fill(~g.final[None], NEG), dim=1)


class HMMLayer(nn.Module):
    """Learnable HMM graphs (segmentation and phased intake pair) with differentiable forward passes."""
    def __init__(self, post_cfg: dict, dur: Durations, prior, frame_rate: float, learn: bool = True):
        super().__init__()
        self.cfg0, self.fr, self.min_frames = post_cfg, frame_rate, np.asarray(dur.min_frames)
        self.floor = float(post_cfg["emission_floor"])
        bw = post_cfg.get("base_weights")
        if bw is None:
            bw = np.ones((K, K))
            bw[:4, 4] = 3.0
        ph = post_cfg["phased"]
        init = {
            "log_bw": torch.tensor(np.log(np.asarray(bw, float)), dtype=torch.float32),
            "logit_stay": torch.tensor([_logit(min(max(p, 1e-3), 1 - 1e-3)) for p in dur.p_stay], dtype=torch.float32),
            "log_bg_s": torch.tensor(math.log(post_cfg["background_expected_s"])),
            "logit_start_bg": torch.tensor(_logit(post_cfg["start_in_background"])),
            "log_g3_entry": torch.tensor([math.log(ph["g3_entry"][p]) for p in PHASES]),
            "log_g4_after_g3": torch.tensor(math.log(ph["g4_after_g3"])),
            "log_repeat_g3": torch.tensor(math.log(ph["repeat_g3_weight"])),
            "log_eps_g3": torch.tensor(math.log(post_cfg["no_intake_g3_prob"])),
            "log_prior": torch.tensor(np.log(np.asarray(prior, float)), dtype=torch.float32),
        }
        for k, v in init.items():
            self.register_parameter(k, nn.Parameter(v.clone(), requires_grad=learn))
            self.register_buffer(k + "_0", v.clone())
        self.bias = nn.Parameter(torch.zeros(()), requires_grad=learn)

    def _cfg(self):
        """cfg["post"] with tensor values (consumed by the core.hmm topology functions)."""
        c = dict(self.cfg0)
        g3 = torch.exp(self.log_g3_entry)
        c.update({
            "base_weights": torch.exp(self.log_bw),
            "background_expected_s": torch.exp(self.log_bg_s),
            "start_in_background": torch.sigmoid(self.logit_start_bg),
            "no_intake_g3_prob": torch.exp(self.log_eps_g3),
            "phased": {**self.cfg0["phased"], "g3_entry": {p: g3[i] for i, p in enumerate(PHASES)},
                       "g4_after_g3": torch.exp(self.log_g4_after_g3),
                       "repeat_g3_weight": torch.exp(self.log_repeat_g3)},
        })
        return c

    def graphs(self):
        c, dev = self._cfg(), self.log_bw.device
        stay = torch.sigmoid(self.logit_stay)
        return {name: expand_torch(fn(c, self.fr), self.min_frames, stay, dev)
                for name, fn in (("seg", ergodic_topology), ("intake", phased_intake_topology),
                                 ("no_intake", phased_no_intake_topology))}

    def emissions(self, logits: torch.Tensor) -> torch.Tensor:
        """Scaled likelihoods, exactly core.hmm.log_emissions: log(clip(p, floor)) - log prior."""
        lp = torch.log_softmax(logits.float(), -1).clamp(min=math.log(self.floor))
        prior = torch.log_softmax(self.log_prior, 0)
        return lp - prior

    def llr(self, e, lengths, graphs=None):
        g = graphs or self.graphs()
        return forward_loglik(g["intake"], e, lengths) - forward_loglik(g["no_intake"], e, lengths)

    def seg_nll(self, e, lengths, y, graphs=None):
        """Per-frame -log P(path | x) on the ergodic graph. y: (B, T) frame labels, -1 = any class.
        A frame may also take the label of its neighbours (blurred boundaries). Returns (loss per
        sequence, feasible mask): sequences whose label path the graph cannot produce are dropped."""
        g = graphs or self.graphs()
        allowed = torch.zeros(*y.shape, K, dtype=torch.bool, device=y.device)
        for sh in (-1, 0, 1):
            ys = torch.roll(y, sh, 1)
            ok = ys >= 0
            allowed |= torch.nn.functional.one_hot(ys.clamp(min=0), K).bool() & ok[..., None]
        allowed |= (y < 0)[..., None]
        num = forward_loglik(g["seg"], e.masked_fill(~allowed, NEG), lengths)
        den = forward_loglik(g["seg"], e, lengths)
        feasible = num > NEG / 2
        return (den - num) / lengths.float(), feasible

    def reg(self):
        """Squared distance of the graph parameters from their initial values."""
        return sum(((getattr(self, k) - getattr(self, k + "_0")) ** 2).sum()
                   for k, _ in self.named_parameters() if k != "bias")

    @torch.no_grad()
    def export(self, dur: Durations) -> dict:
        """Learned values as cfg["post"] overrides + durations + prior (plain floats)."""
        c = self._cfg()
        f = lambda v: float(v)
        p_stay = np.array(dur.p_stay, float)
        p_stay[:4] = torch.sigmoid(self.logit_stay).cpu().numpy()[:4]
        return {
            "post_overrides": {
                "base_weights": c["base_weights"].cpu().numpy().tolist(),
                "background_expected_s": f(c["background_expected_s"]),
                "start_in_background": f(c["start_in_background"]),
                "no_intake_g3_prob": f(c["no_intake_g3_prob"]),
                "phased": {**copy.deepcopy(self.cfg0["phased"]),
                           "g3_entry": {p: f(v) for p, v in c["phased"]["g3_entry"].items()},
                           "g4_after_g3": f(c["phased"]["g4_after_g3"]),
                           "repeat_g3_weight": f(c["phased"]["repeat_g3_weight"])},
            },
            "durations": {"min_frames": np.asarray(dur.min_frames).tolist(), "p_stay": p_stay.tolist()},
            "prior": torch.softmax(self.log_prior, 0).cpu().numpy().tolist(),
            "llr_bias": f(self.bias),
        }


def merge_post(post_cfg: dict, overrides: dict | None) -> dict:
    """cfg["post"] with the learned overrides of ``HMMLayer.export`` applied."""
    out = copy.deepcopy(post_cfg)
    for k, v in (overrides or {}).items():
        out[k] = v
    return out
