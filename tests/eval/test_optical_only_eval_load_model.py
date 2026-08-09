import sys
from pathlib import Path

import torch

OPTICAL_ONLY_DIR = Path(__file__).resolve().parents[2] / "optical_only"
MODEL_DIR = OPTICAL_ONLY_DIR.parent / "Model"
if str(OPTICAL_ONLY_DIR) not in sys.path:
    sys.path.insert(0, str(OPTICAL_ONLY_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from model import OpticalKNClassifier
from scripts.eval.evaluate import load_model


def _write_ckpt(path, universal_aux_enable):
    model = OpticalKNClassifier(
        ref_time_dim=16,
        enc_dim=16,
        optical_curve_dim=16,
        optical_curve_hidden_dim=16,
        num_heads=2,
        k_dim=8,
        head_hidden_dim=16,
        universal_aux_enable=universal_aux_enable,
        proj_dim=64,
        adv_hidden_dim=16,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": {
                "universal_train_enable": True,
                "universal_stage2_mode": "single",
                "universal_stage3_mode": "single",
                "ref_dim": 16,
                "enc_dim": 16,
                "optical_curve_dim": 16,
                "optical_curve_hidden_dim": 16,
                "num_heads": 2,
                "k_dim": 8,
                "head_hidden_dim": 16,
                "arch_version": "optical_only_albef_v11_curve_random_init",
            },
        },
        path,
    )


def test_load_model_without_aux_heads_when_universal_flag_true(tmp_path):
    ckpt = tmp_path / "single.pth"
    _write_ckpt(ckpt, universal_aux_enable=False)
    model, ckpt_args = load_model(str(ckpt), torch.device("cpu"), {})
    assert model.universal_aux_enable is False
    assert ckpt_args["universal_train_enable"] is True


def test_load_model_with_aux_heads_still_loads(tmp_path):
    ckpt = tmp_path / "aux.pth"
    _write_ckpt(ckpt, universal_aux_enable=True)
    model, _ = load_model(str(ckpt), torch.device("cpu"), {})
    assert model.universal_aux_enable is True
