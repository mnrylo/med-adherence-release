"""Configuration files and configuration keys."""
import json

import pytest

from core import paths
from thesis import common as C
from thesis import sweep as SW


def test_conf_files_are_valid_json():
    for f in paths.CONF.glob("*.json"):
        json.loads(f.read_text())


def test_plan_has_only_e1_e3():
    assert sorted(C.EXPERIMENTS) == ["E1", "E2", "E3"]
    assert set(C.EXPERIMENT_DATA) == {"E1", "E2"}


def test_paths_point_inside_repository():
    for p in (paths.GESTURES, paths.LAB_SESSIONS, paths.INGESTURE, paths.WISDM):
        assert str(p).startswith(str(paths.ROOT))


def test_outputs_redirected_by_environment():
    assert "medadh_tests_" in str(paths.OUTPUTS)
    assert C.RUNS.is_relative_to(paths.OUTPUTS)


@pytest.mark.parametrize("key", SW.stage_a()[::17] + ["cnn__fs100_w075_ov75_acc_raw__hp03",
                                                      "cnn_bilstm__fs100_w200_ov75_accgyr_hp03c__hp05"])
def test_config_key_round_trip(key):
    assert C.config_key(C.parse_key(key)) == key


def test_stage_sizes():
    assert len(SW.stage_a()) == 240
    assert len(C.hp_points()) == 64
    assert len(SW.stage_b(SW.stage_a()[:5])) == 15
    c = SW.stage_c(["cnn__fs100_w100_ov50_acc_raw__hpdef", "cnn_bilstm__fs100_w100_ov50_acc_raw__hpdef"])
    assert len(c) == 32 + 64


def test_training_config_applies_factors():
    c = C.parse_key("cnn_lstm__fs50_w200_ov50_acc_hp03__hp07")
    cfg = C.training_config("E2", c)
    assert cfg["data"]["fs"] == 50.0 and cfg["data"]["window_s"] == 2.0 and cfg["data"]["overlap"] == 0.5
    assert cfg["data"]["lowpass_hz"] == 20.0
    assert cfg["data"]["post_filter"] == {"type": "highpass_acc", "cutoff": 0.3}
    assert cfg["train"]["mix"] == C.EXPERIMENT_DATA["E2"]["mix"]
    hp = C.hp_points()[7]
    assert cfg["train"]["lr"] == hp["lr"] and cfg["variant"]["encoder_opts"]["filters"] == hp["conv_filters"]
    assert cfg["variant"]["encoder"] == "cnn_lstm" and cfg["variant"]["encoder_opts"]["channels"] == "acc"
    assert len(cfg["data"]["val_worlds"]) == 2
    e1 = C.training_config("E1", c)
    assert "ingesture" not in e1["train"]["mix"] and e1["world"]["context_s"] is None


def test_hp_default():
    assert C.hp_values("hpdef") == C.DEFAULT_HP
