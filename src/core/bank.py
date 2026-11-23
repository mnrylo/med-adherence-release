"""All streams concatenated into one padded array, windows gathered by centre index.

Each stream is edge-padded by half a window on both sides, so every real sample
can be a window centre without crossing into a neighbouring stream.
"""
from __future__ import annotations

import numpy as np
import torch

from core.sources import MASK, Stream


class StreamBank:
    """Streams concatenated into one padded array; windows are gathered by the index of their centre sample."""
    def __init__(self, streams: list[Stream], win: int):
        self.win, self.half = win, win // 2
        self.index: dict[str, tuple[int, int]] = {}
        xs, ys, auxs = [], [], []
        pos = 0
        for s in streams:
            pad_l, pad_r = self.half, win - self.half
            xs.append(np.pad(s.x, ((pad_l, pad_r), (0, 0)), mode="edge"))
            ys.append(np.pad(s.y, (pad_l, pad_r), constant_values=MASK))
            auxs.append(np.pad(s.aux, (pad_l, pad_r), constant_values=MASK))
            self.index[s.sid] = (pos + pad_l, len(s.x))
            pos += len(s.x) + pad_l + pad_r
        self.x = np.concatenate(xs).astype(np.float32)
        self.y = np.concatenate(ys)
        self.aux = np.concatenate(auxs)
        self.streams = {s.sid: s for s in streams}
        self._gpu = None

    def centres(self, sids, stride: int, phase: int = 0) -> np.ndarray:
        """Bank indices of every stride-th sample (from phase) of the given streams."""
        out = [off + np.arange(phase, T, stride) for off, T in (self.index[s] for s in sids)]
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def frame_centres(self, sid: str, stride: int) -> np.ndarray:
        """Bank indices of the window centres of one stream (sample j * stride)."""
        off, T = self.index[sid]
        return off + np.arange(0, T, stride)

    def to(self, device):
        """Copy the bank to a torch device; required before ``gather``."""
        self._gpu = (torch.from_numpy(self.x).to(device), torch.from_numpy(self.y).to(device),
                     torch.from_numpy(self.aux).to(device))
        self._arange = torch.arange(self.win, device=device) - self.half
        return self

    def gather(self, centres: torch.Tensor):
        """centres: (B,) long on the bank's device -> windows (B, 6, win), main y, aux y."""
        X, Y, A = self._gpu
        idx = centres[:, None] + self._arange[None, :]
        return X[idx].transpose(1, 2), Y[centres], A[centres]
