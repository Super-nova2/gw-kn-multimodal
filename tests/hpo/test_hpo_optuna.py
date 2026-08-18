import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def load_hpo_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "Model"
        / "scripts"
        / "hpo"
        / "hpo_optuna.py"
    )
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

        score = hpo.compute_objective_score(
            results, hpo.OBJECTIVE_PRESETS["fusion_gallery_priority"]
        )

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

        score = hpo.compute_objective_score(
            results, hpo.OBJECTIVE_PRESETS["fusion_gallery_priority"]
        )

        self.assertTrue(math.isnan(score))


class HpoV6ConfigTests(unittest.TestCase):
    def test_current_training_keys_are_allowed_by_hpo(self):
        hpo = load_hpo_module()
        required = {
            "neg_gw_pair_ratio",
            "staged_training_enable",
            "stage_itc_epochs",
            "stage_cls_ramp_epochs",
            "stage_retrieval_ramp_epochs",
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
        with self.assertRaisesRegex(ValueError, "leaves no joint epoch"):
            hpo._apply_curriculum_constraints(
                {
                    "staged_training_enable": True,
                    "itc_weight": 1.0,
                    "cls_weight": 1.0,
                    "gallery_loss_weight": 1.0,
                    "stage_itc_epochs": 20,
                    "stage_cls_ramp_epochs": 4,
                    "stage_retrieval_ramp_epochs": 4,
                    "epochs": 24,
                }
            )

    def test_guardrail_threshold_constraint(self):
        hpo = load_hpo_module()
        with self.assertRaisesRegex(ValueError, "neg_gw_guardrail_recall"):
            hpo._apply_curriculum_constraints(
                {"itc_weight": 1.0, "epochs": 100, "neg_gw_guardrail_recall": 1.5}
            )

    def test_min_lr_must_be_less_than_lr(self):
        hpo = load_hpo_module()
        with self.assertRaisesRegex(ValueError, "min_lr"):
            hpo._apply_curriculum_constraints(
                {"itc_weight": 1.0, "epochs": 100, "lr": 2e-4, "min_lr": 2e-4}
            )

    def test_joint_itc_end_weight_is_derived_from_start_weight(self):
        hpo = load_hpo_module()
        config = {"stage_joint_itc_start_weight": 0.3}

        hpo._apply_stage_joint_itc_schedule(config, 0.5)

        self.assertAlmostEqual(config["stage_joint_itc_end_weight"], 0.15)

    def test_joint_itc_end_ratio_rejects_explicit_end_weight(self):
        hpo = load_hpo_module()
        config = {
            "stage_joint_itc_end_ratio": 0.5,
            "tunable_params": ["stage_joint_itc_start_weight"],
            "fixed_overrides": {"stage_joint_itc_end_weight": 0.25},
        }

        with self.assertRaisesRegex(ValueError, "cannot be fixed"):
            hpo._validate_stage_joint_itc_schedule_config(config)

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

    def test_weighted_retrieval_objective_prioritizes_large_galleries(self):
        hpo = load_hpo_module()
        source_macro = {}
        weights = {100: 0.10, 500: 0.15, 1000: 0.20, 2000: 0.25, 5000: 0.30}
        for size in weights:
            source_macro[f"gallery_{size}_mrr"] = size / 5000.0
            source_macro[f"gallery_{size}_recall_at_1"] = 0.5
        result = {"val_hard_gallery": {"source_macro": source_macro}}
        expected = sum(
            w * (0.8 * (size / 5000.0) + 0.2 * 0.5) for size, w in weights.items()
        )
        self.assertAlmostEqual(
            hpo.compute_retrieval_weighted_score(result, weights), expected
        )

    def test_relative_constraints_and_baseline_promotion(self):
        hpo = load_hpo_module()
        baseline = {"val_neg_gw_min_recall": 0.90, "val_auprc": 0.80}
        candidate = {"val_neg_gw_min_recall": 0.86, "val_auprc": 0.796}
        values = hpo.compute_relative_constraint_values(
            candidate, baseline, min_neg_recall=0.85, max_auprc_drop=0.005
        )
        self.assertTrue(hpo.constraints_are_feasible(values))
        records = [
            {"trial_number": 0, "score": 0.1, "constraints": [0.1, 0.0]},
            {"trial_number": 1, "score": 0.9, "constraints": [-0.1, -0.1]},
            {"trial_number": 2, "score": 1.0, "constraints": [0.1, -0.1]},
        ]
        promoted = hpo.select_promotions(records, 2, 0)
        self.assertEqual([item["trial_number"] for item in promoted], [0, 1])

    def test_restore_first_rung_records_reuses_complete_study(self):
        hpo = load_hpo_module()
        weights = {100: 1.0}

        def result(score, auprc):
            return {
                "val_hard_gallery": {
                    "source_macro": {
                        "gallery_100_mrr": score,
                        "gallery_100_recall_at_1": score,
                    }
                },
                "val_neg_gw_min_recall": 0.9,
                "val_auprc": auprc,
            }

        with tempfile.TemporaryDirectory() as output_dir:
            trials = []
            for number, (score, auprc) in enumerate(((0.5, 0.9), (0.6, 0.898))):
                config_dir = Path(output_dir) / "configs"
                result_dir = Path(output_dir) / "results" / f"trial_{number}"
                config_dir.mkdir(parents=True, exist_ok=True)
                result_dir.mkdir(parents=True, exist_ok=True)
                (config_dir / f"trial_{number}_epoch_30.json").write_text(
                    json.dumps({"ckpt_path": f"trial_{number}"})
                )
                (result_dir / "results_epoch_30.json").write_text(
                    json.dumps(result(score, auprc))
                )
                trials.append(
                    SimpleNamespace(
                        number=number,
                        state=hpo.optuna.trial.TrialState.COMPLETE,
                        value=score,
                        params={"lr": number + 1},
                    )
                )
            study = SimpleNamespace(trials=trials)
            config = {
                "n_trials": 2,
                "output_dir": output_dir,
                "data_path": "/new-jobfs/train.h5",
                "num_workers": 8,
                "constraints": {
                    "min_neg_gw_recall": 0.85,
                    "max_auprc_drop": 0.005,
                },
            }

            records = hpo._restore_first_rung_records(study, config, 30, weights)

        self.assertEqual([record["trial_number"] for record in records], [0, 1])
        self.assertEqual(records[1]["rungs"]["30"]["val_auprc"], 0.898)
        self.assertEqual(records[1]["config"]["data_path"], "/new-jobfs/train.h5")
        self.assertEqual(records[1]["config"]["num_workers"], 8)

    def test_labeled_training_auto_resumes_incomplete_checkpoint(self):
        hpo = load_hpo_module()
        expected = {"complete": True}
        with tempfile.TemporaryDirectory() as output_dir:
            checkpoint_dir = Path(output_dir) / "checkpoint"
            last_checkpoint = checkpoint_dir / "ALBEF" / "albef_last.pth"
            last_checkpoint.parent.mkdir(parents=True)
            last_checkpoint.write_bytes(b"checkpoint")
            config = {"ckpt_path": str(checkpoint_dir), "resume": None}
            record = {"trial_number": 4}
            hpo_config = {"output_dir": output_dir}

            with mock.patch.object(
                hpo, "run_trial_subprocess", return_value=expected
            ) as runner:
                actual = hpo._run_labeled_training(
                    config, record, hpo_config, "confirmation_seed_123"
                )

            used_config = runner.call_args.args[0]
            result_path = hpo._labeled_result_path(
                output_dir, 4, "confirmation_seed_123"
            )
            result_exists = Path(result_path).exists()

        self.assertEqual(actual, expected)
        self.assertEqual(used_config["resume"], str(last_checkpoint))
        self.assertTrue(result_exists)

    def test_confirmation_gates_can_select_robust_candidate(self):
        hpo = load_hpo_module()
        weights = {100: 0.1, 500: 0.15, 1000: 0.2, 2000: 0.25, 5000: 0.3}

        def result(value):
            metrics = {}
            for size in weights:
                metrics[f"gallery_{size}_mrr"] = value
                metrics[f"gallery_{size}_recall_at_1"] = value
            return {
                "val_hard_gallery": {
                    "source_macro": dict(metrics),
                    "by_source": {"bns": dict(metrics), "nsbh": dict(metrics)},
                },
                "val_auprc": 0.9,
                "val_neg_gw_min_recall": 0.9,
                "neg_gw_strata": {"negative": {"count": 100, "recall": 0.9}},
            }

        runs = []
        for seed in (42, 123, 456):
            runs.append({"trial_number": 0, "seed": seed, "results": result(0.50)})
            runs.append({"trial_number": 1, "seed": seed, "results": result(0.51)})
        decision = hpo.summarize_confirmation_runs(runs, weights)
        self.assertEqual(decision["recommended_trial_number"], 1)
        self.assertFalse(decision["retain_full_baseline"])


if __name__ == "__main__":
    unittest.main()
