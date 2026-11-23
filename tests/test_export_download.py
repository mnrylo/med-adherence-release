"""ONNX export of a trained classifier and the dataset download script."""
import hashlib
import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from core.models import MultiHeadClassifier, build_encoder
from thesis import registry as REG

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_datasets", ROOT / "scripts" / "download_datasets.py")
DL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DL)


@pytest.mark.parametrize("name", ["cnn", "cnn_bilstm"])
def test_onnx_export_matches_torch(tmp_path, name):
    pytest.importorskip("onnxruntime")
    opts = {"filters": 16, "lstm_units": 8, "dropout": 0.1, "kernel_size": 5, "channels": "acc_gyr", "fs": 100.0}
    m = MultiHeadClassifier(build_encoder(name, 200, **opts), 5, {"wisdm": 18}, main_head="linear")
    torch.save({"state": m.state_dict(), "encoder": name, "encoder_opts": opts, "aux": {"wisdm": 18}, "n_classes": 5,
                "labels": ["G1", "G2", "G3", "G4", "G5"], "window": 200, "fs": 100.0}, tmp_path / "model.pt")
    info = REG.export_onnx(tmp_path / "model.pt", tmp_path / "model.onnx")
    assert info["max_abs_diff_vs_torch"] < 1e-5
    assert info["input"] == "imu: float32 (batch, 6, 200)"


def test_dataset_specs_match_paths_conf():
    assert DL.DATASETS["ingesture"]["target"] == DL.PATHS["ingesture"]
    assert DL.DATASETS["wisdm"]["target"] == DL.PATHS["wisdm"]
    for spec_ in DL.DATASETS.values():
        assert len(spec_["sha256"]) == 64 and spec_["urls"]


def test_sha256_and_inner_archive(tmp_path):
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("wisdm-dataset/readme.txt", "x")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.write(inner, "folder/wisdm-dataset.zip")
    DL.extract_inner(outer, "wisdm-dataset.zip", tmp_path / "got.zip")
    assert DL.sha256(tmp_path / "got.zip") == hashlib.sha256(inner.read_bytes()).hexdigest()


def test_unknown_dataset_rejected():
    with pytest.raises(SystemExit):
        DL.main(["nope"])
