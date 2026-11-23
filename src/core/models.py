"""Window classifiers: CNN, CNN-LSTM and CNN-BiLSTM encoders with a linear main head.

Every encoder takes raw windows (B, 6, L), acc in g and gyro in deg/s, applies a fixed input
scale (gyro / 100) and optionally keeps only the accelerometer channels. The classifier adds the
main G1..G5 head and one auxiliary head per external dataset used in training.
"""
from __future__ import annotations

import torch
from torch import nn

GYR_SCALE = 1.0 / 100.0
ENCODERS = ("cnn", "cnn_lstm", "cnn_bilstm")


def scale_inputs(x: torch.Tensor) -> torch.Tensor:
    """Fixed per-channel input scale: acc x 1, gyro x 1/100."""
    s = torch.tensor([1.0] * 3 + [GYR_SCALE] * 3, device=x.device, dtype=x.dtype)
    return x * s[None, :, None]


class CNNBiLSTM(nn.Module):
    """Conv1D(filters, kernel, ReLU, same) -> BatchNorm -> MaxPool1D(3) -> (Bi)LSTM -> Dropout.

    The embedding is the concatenated last states of both directions (the last state alone when
    bidirectional is False, the CNN-LSTM). BatchNorm uses eps 1e-3 and momentum 0.01.
    """

    def __init__(self, filters=128, lstm_units=64, dropout=0.15, kernel_size=5, bidirectional=True,
                 channels="acc_gyr", fs=100.0):
        super().__init__()
        self.channels, self.bidirectional = channels, bidirectional
        self.conv = nn.Conv1d(3 if channels == "acc" else 6, filters, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01)
        self.pool = nn.MaxPool1d(3)
        n_dir = 2 if bidirectional else 1
        self.lstm = nn.LSTM(filters, lstm_units, batch_first=True, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.out_dim = n_dir * lstm_units

    def forward(self, x):
        """(B, 6, L) window -> (B, out_dim) embedding."""
        h = scale_inputs(x)
        if self.channels == "acc":
            h = h[:, :3]
        h = self.pool(self.bn(torch.relu(self.conv(h))))
        _, (hn, _) = self.lstm(h.transpose(1, 2))
        z = torch.cat([hn[0], hn[1]], dim=1) if self.bidirectional else hn[0]
        return self.drop(z)


class CNN1D(nn.Module):
    """Convolution-only counterpart of CNNBiLSTM.

    The same stem (Conv1D(k) -> ReLU -> BatchNorm -> MaxPool1D(3)), a second Conv1D(k) block in
    place of the LSTM, then global average pooling over time and dropout.
    """

    def __init__(self, filters=128, kernel_size=5, dropout=0.15, channels="acc_gyr", fs=100.0, **_):
        super().__init__()
        self.channels = channels
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(3 if channels == "acc" else 6, filters, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01)
        self.pool = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01)
        self.drop = nn.Dropout(dropout)
        self.out_dim = filters

    def forward(self, x):
        """(B, 6, L) window -> (B, out_dim) embedding."""
        h = scale_inputs(x)
        if self.channels == "acc":
            h = h[:, :3]
        h = self.pool(self.bn1(torch.relu(self.conv1(h))))
        h = self.bn2(torch.relu(self.conv2(h)))
        return self.drop(h.mean(-1))


class MultiHeadClassifier(nn.Module):
    """Encoder + main head (G1..G5) + one auxiliary head per external dataset.

    forward returns (main logits, {dataset: auxiliary logits}).
    """

    def __init__(self, encoder: nn.Module, n_main: int, aux_sizes: dict[str, int], dropout=0.2,
                 main_head: str = "linear"):
        super().__init__()
        self.encoder = encoder
        d = encoder.out_dim
        if main_head == "linear":
            self.main = nn.Linear(d, n_main)
        else:
            self.main = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n_main))
        self.aux = nn.ModuleDict({k: nn.Linear(d, n) for k, n in aux_sizes.items()})

    def forward(self, x):
        """(B, 6, L) window -> (main logits, {dataset: auxiliary logits})."""
        z = self.encoder(x)
        return self.main(z), {k: h(z) for k, h in self.aux.items()}


def build_encoder(name: str, length: int, **opts) -> nn.Module:
    """Encoder by name ("cnn", "cnn_lstm", "cnn_bilstm"); opts are its constructor arguments."""
    if name == "cnn_bilstm":
        return CNNBiLSTM(**opts)
    if name == "cnn_lstm":
        return CNNBiLSTM(**{**opts, "bidirectional": False})
    if name == "cnn":
        return CNN1D(**opts)
    raise KeyError(name)


KERAS_STYLE = set(ENCODERS)
"""Encoders trained with a linear head with L2 penalty and fp16 mixed precision with loss scaling."""


def main_head_for(name: str) -> str:
    """Main head type of an encoder ("linear" for every encoder of this study)."""
    return "linear" if name in KERAS_STYLE else "mlp"
