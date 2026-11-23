"""HMM post-processing: gesture segmentation (Viterbi) and the intake decision.

All models share one construction. A *topology* is a graph of nodes; each node
emits one gesture class. Every node expands into a chain of ``m_k`` sub-states
(the class's minimum duration in frames), and its last sub-state has a
self-loop ``p_k``. Leaving the node follows the graph's edges. Emissions are
scaled likelihoods p(class | window) / prior(class).

* ``ergodic_topology``: any gesture can follow any other. Viterbi on it gives
  the gesture segmentation (the classic HMM + Viterbi post-processing).
* ``intake_topology``: a left-to-right template of one medicine intake,
  ``[G1]? -> [G4]? -> [G2]? -> G3 -> [G4]?``, separated by optional rests.
  G3 (pill to mouth) is mandatory; G1, G2 and G4 are supporting evidence that
  may be skipped at a cost (``skip``). It must end after G3.
* ``no_intake_topology``: background behaviour. Rest, G1, G2 and G4 in any
  order; G3 is reachable only with a small probability.

The intake score is log P(obs | intake) - log P(obs | no intake), computed
with the forward algorithm. Every observed supporting gesture that fits the
template raises the score; a missing one only costs its skip penalty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from core.sources import LABELS, REST, label_runs_to_sequence

G1, G2, G3, G4, G5 = range(5)


@dataclass
class Durations:
    """Minimum duration (sub-state chain length, frames) and last-sub-state self-loop of each class."""
    min_frames: np.ndarray
    p_stay: np.ndarray

    @classmethod
    def fit(cls, label_streams: list[np.ndarray], frame_stride: int, cfg: dict, frame_rate: float):
        """Durations from labelled streams: chain length = min_duration_fraction x 10th percentile of the segment
        length, self-loop matched to the mean remaining length.
        """
        segs: dict[int, list[int]] = {k: [] for k in range(5)}
        for y in label_streams:
            f = y[::frame_stride]
            f = f[f >= 0]
            if len(f) == 0:
                continue
            b = np.r_[0, np.flatnonzero(np.diff(f)) + 1, len(f)]
            for i in range(len(b) - 1):
                segs[int(f[b[i]])].append(b[i + 1] - b[i])
        m = np.ones(5, int)
        p = np.zeros(5)
        for k in range(5):
            d = np.array(segs[k] or [4])
            m[k] = max(1, int(np.floor(cfg["min_duration_fraction"] * np.percentile(d, 10))))
            p[k] = 1 - 1 / max(1.5, d.mean() - m[k] + 1)
        return cls(m, p)


@dataclass
class Topology:
    """Graph of nodes, each emitting one class; edges with weights, start weights and final nodes."""
    nodes: list[tuple[str, int]]
    edges: dict[str, dict[str, float]]
    start: dict[str, float]
    final: list[str] | None
    stay: dict[str, float]


@dataclass
class ExpandedHMM:
    """A topology expanded into sub-state chains: log transitions, log start, final mask, class of every state."""
    logA: np.ndarray
    log_start: np.ndarray
    final_mask: np.ndarray
    state_class: np.ndarray
    state_node: list[str]


def expand(top: Topology, dur: Durations) -> ExpandedHMM:
    """Expand a topology into chains of ``dur.min_frames`` sub-states per node, the last one with self-loop
    ``dur.p_stay``.
    """
    first, last, cls_of, node_of = {}, {}, [], []
    for name, k in top.nodes:
        first[name] = len(cls_of)
        for _ in range(int(dur.min_frames[k])):
            cls_of.append(k)
            node_of.append(name)
        last[name] = len(cls_of) - 1
    S = len(cls_of)
    A = np.zeros((S, S))
    for name, k in top.nodes:
        for s in range(first[name], last[name]):
            A[s, s + 1] = 1.0
        p = top.stay.get(name, dur.p_stay[k])
        A[last[name], last[name]] = p
        out = top.edges.get(name, {})
        tot = sum(out.values())
        if tot <= 0:
            A[last[name], last[name]] = 1.0
            continue
        for dst, w in out.items():
            A[last[name], first[dst]] += (1 - p) * w / tot
    start = np.zeros(S)
    tot = sum(top.start.values())
    for name, w in top.start.items():
        start[first[name]] = w / tot
    final = np.ones(S, bool) if top.final is None else np.isin(node_of, top.final)
    with np.errstate(divide="ignore"):
        return ExpandedHMM(np.log(A), np.log(start), final, np.array(cls_of), node_of)


def log_emissions(post: np.ndarray, prior: np.ndarray, floor: float) -> np.ndarray:
    """Scaled-likelihood emissions log(max(p, floor)) - log prior, per frame and class."""
    return np.log(np.clip(post, floor, 1)) - np.log(prior)[None, :]


def viterbi(hmm: ExpandedHMM, logE: np.ndarray) -> np.ndarray:
    """Most likely class path (T,) for log emissions per class (T, 5)."""
    e = logE[:, hmm.state_class]
    T, S = e.shape
    delta = hmm.log_start + e[0]
    back = np.zeros((T, S), np.int32)
    for t in range(1, T):
        cand = delta[:, None] + hmm.logA
        back[t] = cand.argmax(0)
        delta = cand.max(0) + e[t]
    delta = np.where(hmm.final_mask, delta, -np.inf)
    path = np.empty(T, np.int64)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return hmm.state_class[path]


def forward_loglik(hmm: ExpandedHMM, logE: np.ndarray) -> float:
    """Log-likelihood of one emission sequence (forward algorithm, final states only)."""
    e = logE[:, hmm.state_class]
    alpha = hmm.log_start + e[0]
    for t in range(1, len(e)):
        alpha = logsumexp(alpha[:, None] + hmm.logA, axis=0) + e[t]
    return float(logsumexp(alpha[hmm.final_mask]))


def _pad(logE_list, state_class):
    """Stack variable-length log-emission arrays (T_i, C) into (B, Tmax, S) and lengths."""
    lengths = np.array([len(e) for e in logE_list])
    out = np.zeros((len(logE_list), lengths.max(), len(state_class)))
    for i, e in enumerate(logE_list):
        out[i, :len(e)] = e[:, state_class]
    return out, lengths


def forward_loglik_batch(hmm: ExpandedHMM, logE_list: list[np.ndarray], chunk: int = 512) -> np.ndarray:
    """forward_loglik for many sequences at once (identical results, vectorised over sequences)."""
    out = np.empty(len(logE_list))
    order = np.argsort([len(e) for e in logE_list])
    for c in range(0, len(order), chunk):
        idx = order[c:c + chunk]
        e, lengths = _pad([logE_list[i] for i in idx], hmm.state_class)
        res = np.empty(len(idx))
        alpha = hmm.log_start + e[:, 0]
        fin = lengths == 1
        if fin.any():
            res[fin] = logsumexp(np.where(hmm.final_mask, alpha[fin], -np.inf), axis=1)
        for t in range(1, e.shape[1]):
            alpha = logsumexp(alpha[:, :, None] + hmm.logA[None], axis=1) + e[:, t]
            fin = lengths == t + 1
            if fin.any():
                res[fin] = logsumexp(np.where(hmm.final_mask, alpha[fin], -np.inf), axis=1)
        out[idx] = res
    return out


def viterbi_batch(hmm: ExpandedHMM, logE_list: list[np.ndarray], chunk: int = 64) -> list[np.ndarray]:
    """viterbi for many sequences at once (identical results, vectorised over sequences)."""
    paths: list = [None] * len(logE_list)
    order = np.argsort([len(e) for e in logE_list])
    for c in range(0, len(order), chunk):
        idx = order[c:c + chunk]
        e, lengths = _pad([logE_list[i] for i in idx], hmm.state_class)
        B, T, S = e.shape
        back = np.zeros((B, T, S), np.int32)
        delta = hmm.log_start + e[:, 0]
        last = np.where(lengths[:, None] == 1, delta, 0.0)
        for t in range(1, T):
            cand = delta[:, :, None] + hmm.logA[None]
            back[:, t] = cand.argmax(1)
            delta = cand.max(1) + e[:, t]
            fin = lengths == t + 1
            if fin.any():
                last[fin] = delta[fin]
        last = np.where(hmm.final_mask, last, -np.inf)
        state = np.zeros((B, T), np.int64)
        rows = np.arange(B)
        state[rows, lengths - 1] = last.argmax(1)
        for t in range(T - 1, 0, -1):
            active = lengths > t
            prev = back[rows, t, state[rows, t]]
            state[active, t - 1] = prev[active]
        for j, i in enumerate(idx):
            paths[i] = hmm.state_class[state[j, :lengths[j]]]
    return paths


def ergodic_topology(cfg: dict, frame_rate: float) -> Topology:
    """Every gesture may follow every other one; rest self-transition from background_expected_s."""
    nodes = [(g, k) for k, g in enumerate(LABELS)]
    edges = _ergodic_weights(cfg)
    stay = {"G5": 1 - 1 / (cfg["background_expected_s"] * frame_rate)}
    return Topology(nodes, edges, {g: 1.0 for g in LABELS}, None, stay)


def intake_topology(cfg: dict, frame_rate: float) -> Topology:
    """Left-to-right template of one intake, [G1]? -> [G4]? -> [G2]? -> G3 -> [G4]?, with optional rests."""
    skip = cfg["skip"]
    stages = [("G1", G1, skip["G1"]), ("G4a", G4, skip["G4_before"]), ("G2", G2, skip["G2"]),
              ("G3", G3, 0.0), ("G4b", G4, skip["G4_after"])]
    nodes = [("bg0", G5), ("bg1", G5)] + [(n, k) for n, k, _ in stages] + \
            [(f"gap_{n}", G5) for n, _, _ in stages[:-1]]

    def entry(i):
        """Distribution over the next node when arriving before stage i."""
        out, carry = {}, 1.0
        for n, _, q in stages[i:]:
            out[n] = out.get(n, 0) + carry * (1 - q)
            carry *= q
        if carry > 0:
            out["bg1"] = out.get("bg1", 0) + carry
        return out

    edges: dict[str, dict[str, float]] = {"bg0": entry(0), "bg1": {}}
    for i, (n, _, _) in enumerate(stages):
        nxt = entry(i + 1) if i + 1 < len(stages) else {"bg1": 1.0}
        if i + 1 < len(stages):
            edges[n] = {f"gap_{n}": cfg["gap_prob"], **{k: (1 - cfg["gap_prob"]) * v for k, v in nxt.items()}}
            edges[f"gap_{n}"] = nxt
        else:
            edges[n] = nxt
    bg_stay = 1 - 1 / (cfg["background_expected_s"] * frame_rate)
    start = {"bg0": cfg["start_in_background"], "G1": 1 - cfg["start_in_background"]}
    final = ["G3", "gap_G3", "G4b", "bg1"]
    return Topology(nodes, edges, start, final, {"bg0": bg_stay, "bg1": bg_stay})


def no_intake_topology(cfg: dict, frame_rate: float) -> Topology:
    """Background behaviour: rest, G1, G2 and G4 in any order; G3 only with probability no_intake_g3_prob."""
    nodes = [(g, k) for k, g in enumerate(LABELS)]
    eps = cfg["no_intake_g3_prob"]
    to = {"G1": 1.0, "G2": 1.0, "G4": 1.0, "G3": eps}
    edges = {"G5": dict(to)}
    for g in ["G1", "G2", "G3", "G4"]:
        edges[g] = {"G5": 3.0, **{h: w for h, w in to.items() if h != g}}
    stay = {"G5": 1 - 1 / (cfg["background_expected_s"] * frame_rate)}
    start = {"G5": cfg["start_in_background"], "G1": (1 - cfg["start_in_background"]) / 3,
             "G2": (1 - cfg["start_in_background"]) / 3, "G4": (1 - cfg["start_in_background"]) / 3}
    return Topology(nodes, edges, start, None, stay)


PHASES = ["none", "g1", "g2", "g3"]


def phased_intake_topology(cfg: dict, frame_rate: float) -> Topology:
    """Phased intake model: an ergodic gesture graph copied over evidence phases.

    Short spurious gestures cost the same as in the no-intake model, so only
    the evidence differs. Entering G1 / G2 moves to the 'g1' / 'g2' phase,
    entering G3 moves to 'g3' and the sequence must end there (G3 is
    mandatory). The weight of entering G3 grows with the support seen before
    it (``g3_entry`` per phase), and G4 after G3 gets ``g4_after_g3``.
    """
    w = cfg["phased"]
    base = _ergodic_weights(cfg)
    nodes, edges = [], {}
    for ph in PHASES:
        for k, g in enumerate(LABELS):
            if g == "G3" and ph != "g3":
                continue
            nodes.append((f"{g}@{ph}", k))
    names = {n for n, _ in nodes}
    for ph in PHASES:
        for g in LABELS:
            src = f"{g}@{ph}"
            if src not in names:
                continue
            out = {}
            for h, bw in base[g].items():
                if h == "G3":
                    nph = "g3"
                    bw = bw * (w["g3_entry"][ph] if ph != "g3" else cfg["no_intake_g3_prob"])
                elif h == "G2" and ph in ("none", "g1"):
                    nph = "g2"
                elif h == "G1" and ph == "none":
                    nph = "g1"
                else:
                    nph = ph
                if h == "G4" and ph == "g3":
                    bw = bw * w["g4_after_g3"]
                out[f"{h}@{nph}"] = out.get(f"{h}@{nph}", 0) + bw
            edges[src] = out
    sb = cfg["start_in_background"]
    start = {"G5@none": sb, "G1@g1": (1 - sb) / 3, "G2@g2": (1 - sb) / 3, "G4@none": (1 - sb) / 3}
    bg = 1 - 1 / (cfg["background_expected_s"] * frame_rate)
    stay = {f"G5@{ph}": bg for ph in PHASES}
    return Topology(nodes, edges, start, [n for n, _ in nodes if n.endswith("@g3")], stay)


def phased_no_intake_topology(cfg: dict, frame_rate: float) -> Topology:
    """Background model: G1, G2, G4 and rest in any order. A first hand-to-mouth
    (G3) is rare, but after one, repeated G3s are cheap ('repetitive' phase):
    eating or similar repetitive behaviour, not a single pill intake."""
    base = _ergodic_weights(cfg)
    eps, rep = cfg["no_intake_g3_prob"], cfg["phased"]["repeat_g3_weight"]
    nodes, edges = [], {}
    for ph in ("quiet", "repetitive"):
        for k, g in enumerate(LABELS):
            if g == "G3" and ph == "quiet":
                continue
            nodes.append((f"{g}@{ph}", k))
    for ph in ("quiet", "repetitive"):
        for g in LABELS:
            if g == "G3" and ph == "quiet":
                continue
            out = {}
            for h, bw in base[g].items():
                if h == "G3":
                    out["G3@repetitive"] = out.get("G3@repetitive", 0) + bw * (eps if ph == "quiet" else rep)
                else:
                    out[f"{h}@{ph}"] = out.get(f"{h}@{ph}", 0) + bw
            edges[f"{g}@{ph}"] = out
    sb = cfg["start_in_background"]
    start = {"G5@quiet": sb, "G1@quiet": (1 - sb) / 3, "G2@quiet": (1 - sb) / 3, "G4@quiet": (1 - sb) / 3}
    bg = 1 - 1 / (cfg["background_expected_s"] * frame_rate)
    return Topology(nodes, edges, start, None, {f"G5@{ph}": bg for ph in ("quiet", "repetitive")})


def _ergodic_weights(cfg):
    """Gesture-to-gesture edge weights shared by every topology: cfg["base_weights"] (5 x 5,
    diagonal unused) when given (e.g. learned by core.hmm_torch), else 1, and 3 into rest."""
    bw = cfg.get("base_weights")
    if bw is not None:
        return {g: {h: bw[i][j] for j, h in enumerate(LABELS) if j != i} for i, g in enumerate(LABELS)}
    out = {g: {h: 1.0 for h in LABELS if h != g} for g in LABELS}
    for g in LABELS[:4]:
        out[g]["G5"] = 3.0
    return out


class PostProcessor:
    """Segmentation (Viterbi) plus two intake scores:

    * ``llr_template``: strict left-to-right template vs ergodic background.
    * ``llr_phased``: phased ergodic intake model vs ergodic background (the intake score of E3).
    """

    def __init__(self, dur: Durations, prior: np.ndarray, cfg: dict, frame_rate: float):
        self.cfg, self.prior, self.fr = cfg, np.asarray(prior), frame_rate
        self.seg = expand(ergodic_topology(cfg, frame_rate), dur)
        self.models = {
            "template": (expand(intake_topology(cfg, frame_rate), dur),
                         expand(no_intake_topology(cfg, frame_rate), dur)),
            "phased": (expand(phased_intake_topology(cfg, frame_rate), dur),
                       expand(phased_no_intake_topology(cfg, frame_rate), dur)),
        }

    def emissions(self, post):
        """Scaled-likelihood emissions of a posterior sequence."""
        return log_emissions(post, self.prior, self.cfg["emission_floor"])

    def segment(self, post) -> np.ndarray:
        """Viterbi path of one posterior sequence on the ergodic graph."""
        return viterbi(self.seg, self.emissions(post))

    def llrs(self, post) -> dict[str, float]:
        """Template and phased intake log-likelihood ratios of one posterior sequence."""
        e = self.emissions(post)
        return {f"llr_{k}": forward_loglik(m1, e) - forward_loglik(m0, e) for k, (m1, m0) in self.models.items()}

    def intake_llr(self, post, model: str = "template") -> float:
        """Intake log-likelihood ratio of one posterior sequence for the given intake model."""
        return self.llrs(post)[f"llr_{model}"]

    def session(self, post) -> dict:
        """Segmentation, gesture order and intake scores of one session."""
        path = self.segment(post)
        seq = label_runs_to_sequence(path)
        llrs = self.llrs(post)
        return {"path": path, "sequence": seq, **llrs, "llr": llrs["llr_template"],
                "rule_g2_g3": rule_g2_g3(seq), "rule_g3": int("G3" in seq)}

    def episode_llrs(self, post) -> dict[str, np.ndarray]:
        """Scores of sliding episodes (episode_s long, every episode_step_s) over a long stream."""
        return self.episode_llrs_batch([post])[0]

    def stream_rules(self, post) -> dict:
        """Rule-based intake decisions (G3, G2 then G3) of one decoded stream."""
        return self.stream_rules_batch([post])[0]

    def llrs_batch(self, posts: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Intake log-likelihood ratios of many posterior sequences (batched forward passes)."""
        e = [self.emissions(p) for p in posts]
        return {f"llr_{k}": forward_loglik_batch(m1, e) - forward_loglik_batch(m0, e)
                for k, (m1, m0) in self.models.items()}

    def segment_batch(self, posts: list[np.ndarray]) -> list[np.ndarray]:
        """Viterbi paths of many posterior sequences."""
        return viterbi_batch(self.seg, [self.emissions(p) for p in posts])

    def sessions_batch(self, posts: list[np.ndarray]) -> list[dict]:
        """``session`` for many posterior sequences."""
        if not posts:
            return []
        paths = self.segment_batch(posts)
        llrs = self.llrs_batch(posts)
        out = []
        for i, path in enumerate(paths):
            seq = label_runs_to_sequence(path)
            r = {k: float(v[i]) for k, v in llrs.items()}
            out.append({"path": path, "sequence": seq, **r, "llr": r["llr_template"],
                        "rule_g2_g3": rule_g2_g3(seq), "rule_g3": int("G3" in seq)})
        return out

    def episode_llrs_batch(self, posts: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """``episode_llrs`` for many streams."""
        W = int(self.cfg["episode_s"] * self.fr)
        step = int(self.cfg["episode_step_s"] * self.fr)
        eps, owner = [], []
        for i, p in enumerate(posts):
            for s0 in range(0, max(1, len(p) - W + 1), step):
                eps.append(p[s0:s0 + W])
                owner.append(i)
        owner = np.array(owner)
        llrs = self.llrs_batch(eps) if eps else {}
        return [{k: v[owner == i] for k, v in llrs.items()} for i in range(len(posts))]

    def stream_rules_batch(self, posts: list[np.ndarray]) -> list[dict]:
        """``stream_rules`` for many streams."""
        out = []
        max_gap = self.cfg["episode_s"] * self.fr
        for path in self.segment_batch(posts) if posts else []:
            segs = _segments(path)
            g3 = [s for s in segs if s[0] == G3]
            g2_ends = [e for k, _, e in segs if k == G2]
            g2g3 = sum(any(0 <= st - e <= max_gap for e in g2_ends) for _, st, _ in g3)
            out.append({"rule_g3_events": len(g3), "rule_g2_g3_events": int(g2g3)})
        return out

    def stream_events(self, post) -> dict:
        """Intake events of one long stream: episode scores and the rule decisions."""
        ep = self.episode_llrs(post)
        out = self.stream_rules(post)
        for k, v in ep.items():
            out[k.replace("llr", "events")] = count_events(v, self.cfg["llr_threshold"])
        out["llr_events"] = out["events_template"]
        return out


def count_events(episode_scores: np.ndarray, threshold: float) -> int:
    """Runs of consecutive positive episodes count as one event."""
    pos = np.asarray(episode_scores) > threshold
    if len(pos) == 0:
        return 0
    return int(pos[0] + np.sum(pos[1:] & ~pos[:-1]))


def calibrate_threshold(session_scores: np.ndarray, session_labels: np.ndarray,
                        negative_streams: list[np.ndarray], hours: float, max_fa_per_h: float) -> float:
    """Threshold with the best balanced accuracy on labelled sessions, subject to a
    false-event budget on negative streams. Falls back to the lowest false-event rate."""
    cands = np.unique(np.concatenate([session_scores] + [s for s in negative_streams if len(s)]))
    cands = np.concatenate([[cands.min() - 1], (cands[:-1] + cands[1:]) / 2, [cands.max() + 1]])
    best, fallback = None, None
    pos, neg = session_labels == 1, session_labels == 0
    for t in cands:
        pred = session_scores > t
        bacc = 0.5 * (pred[pos].mean() + (~pred[neg]).mean())
        fa = sum(count_events(s, t) for s in negative_streams) / max(hours, 1e-9)
        if fa <= max_fa_per_h and (best is None or bacc > best[0] or (bacc == best[0] and t < best[1])):
            best = (bacc, t)
        if fallback is None or fa < fallback[0] or (fa == fallback[0] and bacc > fallback[1]):
            fallback = (fa, bacc, t)
    return float(best[1] if best else fallback[2])


def _segments(path):
    """(class, start, end) runs of a path."""
    b = np.r_[0, np.flatnonzero(np.diff(path)) + 1, len(path)]
    return [(int(path[b[i]]), b[i], b[i + 1]) for i in range(len(b) - 1) if path[b[i]] != REST]


def rule_g2_g3(seq: list[str]) -> int:
    """Legacy rule: a G2 segment followed later by a G3 segment."""
    return int("G2" in seq and "G3" in seq[seq.index("G2") + 1:])
