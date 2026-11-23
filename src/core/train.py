"""Augmentation and multi-source supervised training of a window classifier."""
from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F

from core.bank import StreamBank
from core.models import KERAS_STYLE, MultiHeadClassifier, build_encoder, main_head_for


def random_rotation(n, max_deg, device):
    """n random 3 x 3 rotations about uniform axes, by at most max_deg degrees."""
    axis = F.normalize(torch.randn(n, 3, device=device), dim=1)
    ang = (torch.rand(n, device=device) * 2 - 1) * math.radians(max_deg)
    K = torch.zeros(n, 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(1, 2)
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return torch.eye(3, device=device) + s * K + (1 - c) * (K @ K)


def time_warp(x, lo, hi):
    """Resample every window by a random tempo factor in [lo, hi] around its centre (linear interpolation)."""
    B, C, L = x.shape
    f = torch.empty(B, 1, device=x.device).uniform_(math.log(lo), math.log(hi)).exp()
    c = (L - 1) / 2
    pos = (c + (torch.arange(L, device=x.device)[None] - c) * f).clamp(0, L - 1)
    i0 = pos.floor().long().clamp(max=L - 2)
    w = (pos - i0)[:, None]
    g0 = torch.gather(x, 2, i0[:, None].expand(B, C, L))
    g1 = torch.gather(x, 2, (i0 + 1)[:, None].expand(B, C, L))
    return g0 * (1 - w) + g1 * w


def augment(x, cfg):
    """Rotation, amplitude, tempo, sensor noise, gyro bias and dropped samples."""
    B, _, L = x.shape
    R = random_rotation(B, cfg["rotation_deg"], x.device)
    acc = torch.einsum("bij,bjl->bil", R, x[:, :3])
    gyr = torch.einsum("bij,bjl->bil", R, x[:, 3:])
    mean = acc.mean(-1, keepdim=True)
    s = torch.empty(B, 1, 1, device=x.device).uniform_(*cfg["amplitude"])
    acc, gyr = mean + (acc - mean) * s, gyr * s
    gyr = gyr + torch.randn(B, 3, 1, device=x.device) * cfg["gyro_bias_dps"]
    x = time_warp(torch.cat([acc, gyr], 1), *cfg["time_warp"])
    noise = torch.tensor([cfg["noise_acc_g"]] * 3 + [cfg["noise_gyr_dps"]] * 3, device=x.device)
    x = x + torch.randn_like(x) * noise[None, :, None]
    drop = torch.rand(B, 1, L, device=x.device) < cfg["drop_prob"]
    drop[..., 0] = False
    if drop.any():
        idx = torch.arange(L, device=x.device).expand(B, 1, L).masked_fill(drop, 0)
        idx = torch.cummax(idx, dim=2).values
        x = torch.gather(x, 2, idx.expand(B, 6, L))
    return x


def _rng_state(gen):
    """Torch, CUDA and generator random states (for checkpoints)."""
    st = {"torch": torch.get_rng_state(), "gen": gen.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(gen, st):
    """Restore the random states saved by ``_rng_state``."""
    torch.set_rng_state(st["torch"])
    gen.set_state(st["gen"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_checkpoint(path, **state):
    """Atomic save: a crash while writing never corrupts the previous checkpoint."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    """Checkpoint dict, or None when there is none."""
    if path is None or not Path(path).exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def lr_at(step, total, warm):
    """Learning-rate factor: linear warm-up over ``warm`` steps, then cosine decay to 0 at ``total``."""
    return min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))


@torch.no_grad()
def predict(model, bank: StreamBank, centres: np.ndarray, device, batch=None) -> np.ndarray:
    """Softmax outputs; the default batch shrinks with the window length (long windows on 4 GB GPUs)."""
    batch = batch or min(4096, max(256, 204800 // bank.win))
    model.eval()
    out = []
    c_all = torch.from_numpy(centres).to(device)
    for i in range(0, len(c_all), batch):
        x, _, _ = bank.gather(c_all[i:i + batch])
        amp = device.type == "cuda" and getattr(model, "amp", True)
        with torch.autocast(device_type=device.type, dtype=getattr(model, "amp_dtype", torch.bfloat16), enabled=amp):
            logits, _ = model(x)
        out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 5), np.float32)


def train_supervised(encoder_name: str, bank: StreamBank,
                     pools: dict[str, tuple[np.ndarray, float]], aux_heads: dict[str, int],
                     val_centres: np.ndarray, n_classes: int, cfg: dict, seed: int, device, log=print,
                     progress=None, ckpt_path=None):
    """Train one classifier on windows drawn from several pools.

    pools: {source: (window centres in the bank, sampling weight)}; every batch holds each
    source in proportion to its weight. The main head is trained with class-weighted
    cross-entropy (inverse square-root class frequency of the expected mix, label smoothing);
    auxiliary heads are trained on their own source only. The model with the best validation
    window macro F1 (val_centres, every cfg["eval_every"] steps) is returned, or the final
    weights when cfg["keep_last"] is set. ckpt_path: checkpoint written at every validation; an
    existing checkpoint resumes the run.
    Returns (model, info) with best_step, best_val_macro_f1, history, seconds, prior and n_params.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = build_encoder(encoder_name, bank.win, **cfg.get("encoder_opts", {}))
    model = MultiHeadClassifier(enc, n_classes, aux_heads, main_head=main_head_for(encoder_name)).to(device)
    model.amp = encoder_name not in KERAS_STYLE or bool(cfg.get("fp16"))
    model.amp_dtype = torch.float16 if encoder_name in KERAS_STYLE else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and encoder_name in KERAS_STYLE and model.amp)
    head_l2 = cfg.get("head_l2", 0.01) if encoder_name in KERAS_STYLE else 0.0
    fwd = model
    enc_params = set(id(p) for p in model.encoder.parameters())
    enc_lr = cfg["lr"]
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in enc_params], "base": enc_lr},
        {"params": [p for p in model.parameters() if id(p) not in enc_params], "base": cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])

    names = list(pools)
    gpu_pools = [torch.from_numpy(pools[k][0]).to(device) for k in names]
    weights = np.array([pools[k][1] for k in names], float)
    weights /= weights.sum()
    counts = np.zeros(n_classes)
    for k, w in zip(names, weights):
        y = bank.y[pools[k][0]]
        y = y[y >= 0]
        if len(y):
            counts += w * np.bincount(y, minlength=n_classes) / len(y)
    cw = 1 / np.sqrt(np.maximum(counts, 1e-6))
    cw = cw / (cw * counts).sum() * counts.sum()
    prior = counts * cw / (counts * cw).sum()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device),
                               label_smoothing=cfg["label_smoothing"], ignore_index=-1)
    aux_crit = nn.CrossEntropyLoss(ignore_index=-1)
    steps, bs = cfg["steps"], cfg["batch_size"]
    n_per = np.maximum(1, np.round(weights * bs).astype(int))
    gen = torch.Generator(device=device).manual_seed(seed)
    yv = bank.y[val_centres]
    best = {"f1": -1, "step": -1, "state": None}
    hist, t0, run, start = [], time.time(), 0.0, 0
    ck = load_checkpoint(ckpt_path)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        _set_rng_state(gen, ck["rng"])
        start, run, best, hist = ck["step"], ck["run"], ck["best"], ck["hist"]
        t0 -= ck["seconds"]
        log(f"    resumed from checkpoint at step {start}")
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = g["base"] * lr_at(step, steps, cfg["warmup_steps"])
        model.train()
        cs, srcs = [], []
        for i, (p, n) in enumerate(zip(gpu_pools, n_per)):
            cs.append(p[torch.randint(len(p), (int(n),), device=device, generator=gen)])
            srcs.append(torch.full((int(n),), i, device=device))
        c, src = torch.cat(cs), torch.cat(srcs)
        x, y, a = bank.gather(c)
        x = augment(x, cfg["augment"])
        with torch.autocast(device_type=device.type, dtype=model.amp_dtype, enabled=device.type == "cuda" and model.amp):
            logits, aux = fwd(x)
        loss = crit(logits.float(), y)
        if head_l2:
            loss = loss + head_l2 * model.main.weight.pow(2).sum()
        for k, out in aux.items():
            sel = src == names.index(k)
            if sel.any():
                loss = loss + cfg["aux_weight"] * aux_crit(out.float()[sel], a[sel])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if progress is not None and (step + 1) % 20 == 0:
            progress(step + 1, loss=run)
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == steps:
            pv = predict(model, bank, val_centres, device)
            f1 = f1_score(yv, pv.argmax(1), average="macro")
            hist.append({"step": step + 1, "loss": run, "val_macro_f1": f1})
            if progress is not None:
                progress(step + 1, loss=run, val_macro_f1=f1, best_val_f1=max(f1, best["f1"]))
            log(f"    step {step + 1:5d} loss {run:.4f} val_f1 {f1:.4f} ({time.time() - t0:.0f}s)")
            if f1 > best["f1"]:
                best = {"f1": f1, "step": step + 1, "state": copy.deepcopy(model.state_dict())}
            if step + 1 < steps:
                save_checkpoint(ckpt_path, model=model.state_dict(), opt=opt.state_dict(), rng=_rng_state(gen),
                                step=step + 1, run=run, best=best, hist=hist, seconds=time.time() - t0)
    if cfg.get("keep_last"):
        best = {"f1": hist[-1]["val_macro_f1"] if hist else float("nan"), "step": steps, "state": None}
    else:
        model.load_state_dict(best["state"])
    return model, {"best_step": best["step"], "best_val_macro_f1": best["f1"], "history": hist,
                   "seconds": time.time() - t0, "prior": prior.tolist(),
                   "n_params": sum(p.numel() for p in model.parameters())}
