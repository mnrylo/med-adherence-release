"""Encoders, the stream bank and supervised training (core.models, core.bank, core.train)."""
import numpy as np
import pytest
import torch

from core import sources as S
from core.bank import StreamBank
from core.models import ENCODERS, MultiHeadClassifier, build_encoder, main_head_for
from core.train import augment, predict, train_supervised

AUG = {"rotation_deg": 30, "amplitude": [0.8, 1.25], "time_warp": [0.8, 1.25], "noise_acc_g": 0.01,
       "noise_gyr_dps": 1.0, "gyro_bias_dps": 2.0, "drop_prob": 0.05}


@pytest.mark.parametrize("name", ENCODERS)
@pytest.mark.parametrize("channels", ["acc", "acc_gyr"])
def test_encoder_shapes(name, channels):
    enc = build_encoder(name, 100, filters=16, lstm_units=8, dropout=0.1, kernel_size=5, channels=channels, fs=50.0)
    m = MultiHeadClassifier(enc, 5, {"wisdm": 18}, main_head=main_head_for(name)).eval()
    logits, aux = m(torch.randn(4, 6, 100))
    assert logits.shape == (4, 5) and aux["wisdm"].shape == (4, 18)


def test_acc_only_ignores_gyro():
    torch.manual_seed(0)
    enc = build_encoder("cnn", 50, filters=8, channels="acc").eval()
    x = torch.randn(2, 6, 50)
    y = x.clone()
    y[:, 3:] = 0
    assert torch.allclose(enc(x), enc(y))


def test_bank_padding_and_gather():
    a = S.Stream("a", "t", "p", np.arange(12, dtype=np.float32).reshape(2, 6).repeat(3, 0), np.array([0, 0, 1, 1, 2, 2]),
                 np.full(6, -1))
    bank = StreamBank([a], 4).to(torch.device("cpu"))
    off, T = bank.index["a"]
    assert T == 6 and off == 2
    x, y, _ = bank.gather(torch.from_numpy(bank.frame_centres("a", 2)))
    assert x.shape == (3, 6, 4) and y.tolist() == [0, 1, 2]
    assert torch.equal(x[0, :, 0], x[0, :, 2])


def test_augment_keeps_shape():
    x = torch.randn(8, 6, 50)
    assert augment(x, AUG).shape == x.shape


def test_train_and_predict_cpu(tmp_path):
    rng = np.random.default_rng(0)
    streams = []
    for i in range(6):
        y = np.repeat(np.arange(5), 40)
        x = rng.normal(size=(200, 6)).astype(np.float32) + y[:, None].astype(np.float32)
        streams.append(S.Stream(f"s{i}", "t", "p", x, y, np.full(200, -1)))
    bank = StreamBank(streams, 20).to(torch.device("cpu"))
    tr = bank.centres([s.sid for s in streams[:4]], 2)
    val = bank.centres([s.sid for s in streams[4:]], 5)
    cfg = {"steps": 30, "batch_size": 64, "lr": 3e-3, "weight_decay": 0.01, "warmup_steps": 5, "label_smoothing": 0.05,
           "aux_weight": 0.3, "eval_every": 10, "augment": {**AUG, "rotation_deg": 0, "drop_prob": 0.0}, "fs": 20.0,
           "encoder_opts": {"filters": 16, "lstm_units": 8, "channels": "acc_gyr", "fs": 20.0}}
    model, info = train_supervised("cnn_bilstm", bank, {"all": (tr, 1.0)}, {}, val, 5, cfg, 0, torch.device("cpu"),
                                   log=lambda m: None, ckpt_path=tmp_path / "ck.pt")
    assert info["best_step"] in (10, 20, 30) and len(info["history"]) == 3
    assert np.isclose(sum(info["prior"]), 1.0)
    p = predict(model, bank, val, torch.device("cpu"))
    assert p.shape == (len(val), 5) and np.allclose(p.sum(1), 1, atol=1e-5)
    model2, info2 = train_supervised("cnn_bilstm", bank, {"all": (tr, 1.0)}, {}, val, 5, cfg, 0, torch.device("cpu"),
                                     log=lambda m: None)
    assert np.allclose(predict(model2, bank, val, torch.device("cpu")), p, atol=1e-6)
