import importlib.util
import math
from pathlib import Path
import unittest


def load_hpo_module():
    module_path = Path(__file__).resolve().parents[2] / "Model" / "scripts" / "hpo" / "hpo_optuna.py"
    spec = importlib.util.spec_from_file_location("hpo_optuna", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HpoObjectiveTests(unittest.TestCase):
    def test_fusion_gallery_priority_objective_uses_saved_metrics(self):
        hpo = load_hpo_module()
        results = {
            "val_fusion_gallery_mrr": 0.80,
            "val_fusion_gallery_recall_at_1": 0.75,
            "val_auprc": 0.90,
            "val_auroc": 0.95,
        }

        score = hpo.compute_objective_score(results, hpo.OBJECTIVE_PRESETS["fusion_gallery_priority"])

        expected = (0.70 * 0.80) + (0.15 * 0.75) + (0.10 * 0.90) + (0.05 * 0.95)
        self.assertTrue(math.isclose(score, expected, rel_tol=0.0, abs_tol=1e-12))
        summary = hpo.format_metric_summary(
            results, hpo.OBJECTIVE_PRESETS["fusion_gallery_priority"].keys()
        )
        self.assertIn("FG_MRR=0.800000", summary)
        self.assertIn("FG_R@1=0.750000", summary)

    def test_fusion_gallery_priority_objective_nan_when_required_metric_missing(self):
        hpo = load_hpo_module()
        results = {
            "best_ckpt_metric": "fusion_gallery_mrr",
            "best_ckpt_score": 0.80,
            "val_auprc": 0.90,
            "val_auroc": 0.95,
        }

        score = hpo.compute_objective_score(results, hpo.OBJECTIVE_PRESETS["fusion_gallery_priority"])

        self.assertTrue(math.isnan(score))


class HpoV6ConfigTests(unittest.TestCase):
    def test_current_training_keys_are_allowed_by_hpo(self):
        hpo = load_hpo_module()
        required = {
            "neg_gw_pair_ratio",
            "staged_training_enable",
            "stage_alignment_epochs",
            "stage_head_epochs",
            "stage_joint_itc_start_weight",
            "stage_joint_itc_end_weight",
            "encoder_lr_ratio",
            "itc_extra_negative_enable",
            "neg_gw_guardrail_enable",
            "neg_gw_guardrail_recall",
            "cls_aligned_pos_weight",
            "cls_neg_gw_weight",
            "cls_mismatched_weight",
            "cls_external_neg_weight",
        }
        self.assertTrue(required <= hpo.MAGIKS_ARG_KEYS)

    def test_batch_size_derivation_keeps_event_coverage(self):
        hpo = load_hpo_module()

        cfg_512 = {"batch_size": 512, "samples_per_gw": 4, "neg_gw_pair_ratio": 0.2}
        hpo.apply_batch_size_step_derivation(cfg_512)
        self.assertEqual(cfg_512["steps_per_epoch"], 201)
        self.assertEqual(cfg_512["val_steps_per_epoch"], 32)

        cfg_1024 = {"batch_size": 1024, "samples_per_gw": 4, "neg_gw_pair_ratio": 0.2}
        hpo.apply_batch_size_step_derivation(cfg_1024)
        self.assertEqual(cfg_1024["steps_per_epoch"], 100)
        self.assertEqual(cfg_1024["val_steps_per_epoch"], 16)

        cfg_2048 = {"batch_size": 2048, "samples_per_gw": 4, "neg_gw_pair_ratio": 0.2}
        hpo.apply_batch_size_step_derivation(cfg_2048)
        self.assertEqual(cfg_2048["steps_per_epoch"], 50)
        self.assertEqual(cfg_2048["val_steps_per_epoch"], 8)

    def test_stage_constraints_reject_overflow(self):
        hpo = load_hpo_module()
        with self.assertRaisesRegex(ValueError, "staged training stages do not fit"):
            hpo._apply_three_stage_constraints({
                "staged_training_enable": True,
                "stage_alignment_epochs": 20,
                "stage_head_epochs": 4,
                "epochs": 24,
            })

    def test_guardrail_threshold_constraint(self):
        hpo = load_hpo_module()
        with self.assertRaisesRegex(ValueError, "neg_gw_guardrail_recall"):
            hpo._apply_three_stage_constraints({"neg_gw_guardrail_recall": 1.5})

    def test_min_lr_must_be_less_than_lr(self):
        hpo = load_hpo_module()
        with self.assertRaisesRegex(ValueError, "min_lr"):
            hpo._apply_three_stage_constraints({"lr": 2e-4, "min_lr": 2e-4})

    def test_neg_gw_min_recall_is_supported_min_metric(self):
        hpo = load_hpo_module()
        weights = hpo._validate_metric_weights(
            {"val_neg_gw_min_recall": 0.85}, "objective_min_metrics"
        )
        self.assertEqual(weights["val_neg_gw_min_recall"], 0.85)

        passed, reason = hpo.check_objective_min_metrics(
            {"val_neg_gw_min_recall": 0.86},
            {"val_neg_gw_min_recall": 0.85},
        )
        self.assertTrue(passed)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
