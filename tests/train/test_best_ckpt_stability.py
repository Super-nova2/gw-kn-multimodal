import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
sys.path.insert(0, str(MODEL_DIR))

from scripts.train.train import (  # noqa: E402
    compute_best_ckpt_selection_start_epoch,
    compute_best_ckpt_stage_epochs,
    is_best_ckpt_selection_eligible,
    load_merged_json_config,
    validate_best_ckpt_selection_schedule,
)


DEFAULT_CONFIG = MODEL_DIR / "args" / "defaults" / "MAGIKS_BNS_NSBH_default.json"


def make_args(**overrides):
    values = {
        "epochs": 100,
        "best_ckpt_metric": "g2o_mrr",
        "best_ckpt_start_epoch": None,
        "lr_scheduler": "cosine",
        "warmup_epochs": 0,
        "itc_weight": 1.0,
        "itc_decay_start_epoch": 0,
        "itc_decay_epochs": 0,
        "itc_decay_ratio": 0.0,
        "cls_weight": 0.0,
        "cls_start_epoch": 0,
        "cls_ramp_epochs": 0,
        "gallery_loss_weight": 0.0,
        "retrieval_start_epoch": 0,
        "gallery_loss_ramp_epochs": 0,
        "gallery_hard_neg_enable": False,
        "gallery_hard_neg_weight": 0.5,
        "gallery_hard_neg_start_after_retrieval_epochs": 2,
        "gallery_hard_neg_ramp_epochs": 2,
        "compute_cls_metrics": True,
        "compute_fusion_gallery_metrics": True,
        "fusion_gallery_metrics_start_epoch": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class BestCheckpointStabilityTests(unittest.TestCase):
    def test_ablation_configs_start_after_all_enabled_phases(self):
        cases = {
            "full": 14,
            "no_cls_loss": 14,
            "no_gallery_loss": 9,
            "no_itc_loss": 14,
            "no_cross_atten": 14,
            "no_hard_mining": 14,
            "no_fusion": 4,
        }
        for config_name, expected_epoch in cases.items():
            with self.subTest(config_name=config_name):
                config_path = (
                    MODEL_DIR / "args" / f"MAGIKS_BNS_NSBH_{config_name}.json"
                )
                merged = load_merged_json_config(
                    default_json_config=str(DEFAULT_CONFIG),
                    json_config=str(config_path),
                )
                args = SimpleNamespace(**merged)

                self.assertEqual(
                    validate_best_ckpt_selection_schedule(args), expected_epoch
                )
                self.assertFalse(
                    is_best_ckpt_selection_eligible(args, expected_epoch - 1)
                )
                self.assertTrue(
                    is_best_ckpt_selection_eligible(args, expected_epoch)
                )

    def test_disabled_future_phases_do_not_delay_selection(self):
        args = make_args(
            warmup_epochs=5,
            cls_weight=0.0,
            cls_start_epoch=999,
            gallery_loss_weight=0.0,
            retrieval_start_epoch=999,
            gallery_hard_neg_enable=True,
            gallery_hard_neg_start_after_retrieval_epochs=999,
        )

        self.assertEqual(compute_best_ckpt_stage_epochs(args), {"lr_warmup": 4})
        self.assertEqual(compute_best_ckpt_selection_start_epoch(args), 4)

    def test_manual_start_epoch_can_only_delay_automatic_start(self):
        automatic = make_args(warmup_epochs=5, best_ckpt_start_epoch=2)
        delayed = make_args(warmup_epochs=5, best_ckpt_start_epoch=8)

        self.assertEqual(compute_best_ckpt_selection_start_epoch(automatic), 4)
        self.assertEqual(compute_best_ckpt_selection_start_epoch(delayed), 8)

    def test_itc_decay_is_included_when_enabled(self):
        args = make_args(
            warmup_epochs=5,
            itc_decay_start_epoch=8,
            itc_decay_epochs=3,
            itc_decay_ratio=0.5,
        )

        self.assertEqual(compute_best_ckpt_stage_epochs(args)["itc_decay"], 10)
        self.assertEqual(compute_best_ckpt_selection_start_epoch(args), 10)

    def test_unreachable_phase_is_rejected(self):
        args = make_args(
            epochs=10,
            cls_weight=1.0,
            cls_start_epoch=8,
            cls_ramp_epochs=3,
        )

        with self.assertRaisesRegex(ValueError, "do not complete"):
            validate_best_ckpt_selection_schedule(args)

    def test_selected_metric_must_be_enabled(self):
        args = make_args(
            best_ckpt_metric="fusion_gallery_mrr",
            compute_fusion_gallery_metrics=False,
        )

        with self.assertRaisesRegex(ValueError, "compute_fusion_gallery_metrics=true"):
            validate_best_ckpt_selection_schedule(args)


if __name__ == "__main__":
    unittest.main()
