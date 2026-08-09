import argparse
import os
import sys
import unittest

import torch


MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Model"))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from scripts.train import train  # noqa: E402


def _args(**overrides):
    values = {
        "best_ckpt_metric": "fusion_gallery_mrr",
        "cls_weight": 0.0,
        "cls_start_epoch": 10,
        "cls_ramp_epochs": 3,
        "gallery_loss_weight": 1.0,
        "gallery_loss_ramp_epochs": 0,
        "retrieval_start_epoch": 10,
        "epochs": 100,
        "compute_cls_metrics": True,
        "compute_fusion_gallery_metrics": True,
        "fusion_gallery_metrics_start_epoch": 0,
        "gallery_hard_neg_enable": False,
        "gallery_hard_neg_topk": 32,
        "gallery_hard_neg_weight": 0.5,
        "gallery_hard_neg_start_after_retrieval_epochs": 2,
        "gallery_hard_neg_ramp_epochs": 2,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class BestCheckpointEligibilityTests(unittest.TestCase):
    def test_cls_metric_is_eligible_when_cls_active(self):
        args = _args(
            best_ckpt_metric="cls_composite_auprc_auroc",
            cls_weight=1.0,
            gallery_loss_weight=0.0,
        )

        self.assertTrue(train.is_best_ckpt_selection_eligible(args, epoch=14))

    def test_cls_metric_waits_for_cls_start(self):
        args = _args(
            best_ckpt_metric="cls_composite_auprc_auroc",
            cls_weight=1.0,
            gallery_loss_weight=0.0,
            cls_start_epoch=10,
        )

        self.assertFalse(train.is_best_ckpt_selection_eligible(args, epoch=9))

    def test_g2o_metric_is_eligible_without_cls_or_gallery_loss(self):
        args = _args(
            best_ckpt_metric="g2o_mrr",
            cls_weight=0.0,
            gallery_loss_weight=0.0,
            retrieval_start_epoch=999,
        )

        self.assertTrue(train.is_best_ckpt_selection_eligible(args, epoch=0))


class MetricComputationSwitchTests(unittest.TestCase):
    def test_fusion_gallery_metrics_can_be_enabled_without_gallery_loss(self):
        args = _args(gallery_loss_weight=0.0)

        self.assertTrue(train.should_compute_fusion_gallery_metrics(args, epoch=0))

    def test_cls_metrics_can_be_enabled_without_cls_loss(self):
        args = _args(cls_weight=0.0)

        self.assertTrue(train.should_compute_cls_metrics(args))


class GalleryLossWeightTests(unittest.TestCase):
    def test_gallery_loss_weight_zero_before_retrieval_start(self):
        args = _args(
            gallery_loss_weight=0.05,
            retrieval_start_epoch=10,
            gallery_loss_ramp_epochs=5,
        )

        self.assertEqual(train.compute_gallery_loss_weight(args, epoch=9), 0.0)

    def test_gallery_loss_weight_ramps_linearly_after_retrieval_start(self):
        args = _args(
            gallery_loss_weight=0.05,
            retrieval_start_epoch=10,
            gallery_loss_ramp_epochs=5,
        )

        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=10), 0.01)
        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=11), 0.02)
        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=14), 0.05)
        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=20), 0.05)

    def test_gallery_loss_weight_without_ramp_goes_straight_to_target(self):
        args = _args(
            gallery_loss_weight=0.05,
            retrieval_start_epoch=10,
            gallery_loss_ramp_epochs=0,
        )

        self.assertEqual(train.compute_gallery_loss_weight(args, epoch=9), 0.0)
        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=10), 0.05)

    def test_v11_gallery_loss_ramps_from_epoch_16_to_20_1based(self):
        args = _args(
            gallery_loss_weight=1.0,
            retrieval_start_epoch=15,
            gallery_loss_ramp_epochs=5,
        )

        self.assertEqual(train.compute_gallery_loss_weight(args, epoch=14), 0.0)
        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=15), 0.2)
        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=16), 0.4)
        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=17), 0.6)
        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=18), 0.8)
        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=19), 1.0)
        self.assertAlmostEqual(train.compute_gallery_loss_weight(args, epoch=20), 1.0)


class GalleryHardBestCkptGatingTests(unittest.TestCase):
    def test_fusion_gallery_eligible_when_hard_disabled(self):
        """Best ckpt eligible normally when gallery hard mining is off."""
        args = _args(
            best_ckpt_metric="fusion_gallery_mrr",
            gallery_hard_neg_enable=False,
        )
        self.assertTrue(
            train.is_best_ckpt_selection_eligible(args, epoch=11)
        )

    def test_fusion_gallery_pending_during_hard_ramp(self):
        """Best ckpt NOT eligible while retrieval hard mining is still ramping."""
        args = _args(
            best_ckpt_metric="fusion_gallery_mrr",
            gallery_hard_neg_enable=True,
            retrieval_start_epoch=10,
            gallery_hard_neg_start_after_retrieval_epochs=2,
            gallery_hard_neg_ramp_epochs=2,
            gallery_hard_neg_weight=0.5,
        )
        # hard_start = 12, ramp to 13, so epoch 12 is partial (weight=0.25)
        self.assertFalse(
            train.is_best_ckpt_selection_eligible(args, epoch=12)
        )

    def test_fusion_gallery_eligible_after_hard_ramp(self):
        """Best ckpt eligible once retrieval hard mining reaches full weight."""
        args = _args(
            best_ckpt_metric="fusion_gallery_mrr",
            gallery_hard_neg_enable=True,
            retrieval_start_epoch=10,
            gallery_hard_neg_start_after_retrieval_epochs=2,
            gallery_hard_neg_ramp_epochs=2,
            gallery_hard_neg_weight=0.5,
        )
        # hard_start=12, ramp 2 epochs, full at epoch 13
        self.assertTrue(
            train.is_best_ckpt_selection_eligible(args, epoch=13)
        )

    def test_fusion_gallery_eligible_when_hard_ramp_cannot_reach_full_weight(self):
        """An unreachable hard ramp must not silently relax checkpoint gating."""
        args = _args(
            best_ckpt_metric="fusion_gallery_mrr",
            gallery_hard_neg_enable=True,
            retrieval_start_epoch=10,
            gallery_hard_neg_start_after_retrieval_epochs=2,
            gallery_hard_neg_ramp_epochs=2,
            gallery_hard_neg_weight=0.5,
            epochs=13,
        )
        self.assertFalse(train.gallery_hard_neg_full_activation_reachable(args))
        self.assertFalse(
            train.is_best_ckpt_selection_eligible(args, epoch=12)
        )

    def test_alignment_only_eligible_immediately(self):
        """alignment_only (gallery_loss_weight=0) skips hard ramp gate."""
        args = _args(
            best_ckpt_metric="g2o_mrr",
            gallery_loss_weight=0.0,
            gallery_hard_neg_enable=True,
            retrieval_start_epoch=999,
        )
        self.assertTrue(
            train.is_best_ckpt_selection_eligible(args, epoch=5)
        )

    def test_describe_pending_mentions_retrieval_hard_ramp(self):
        args = _args(
            best_ckpt_metric="fusion_gallery_mrr",
            gallery_hard_neg_enable=True,
            retrieval_start_epoch=10,
            gallery_hard_neg_start_after_retrieval_epochs=2,
            gallery_hard_neg_ramp_epochs=2,
            gallery_hard_neg_weight=0.5,
        )
        msg = train.describe_best_ckpt_selection_pending(args, epoch=12)
        self.assertIn("fusion_gallery_hard_negative", msg)

    def test_fusion_gallery_pending_during_gallery_loss_ramp(self):
        args = _args(
            best_ckpt_metric="fusion_gallery_mrr",
            gallery_loss_weight=1.0,
            retrieval_start_epoch=15,
            gallery_loss_ramp_epochs=5,
            gallery_hard_neg_enable=False,
        )

        self.assertFalse(train.is_best_ckpt_selection_eligible(args, epoch=18))
        self.assertTrue(train.is_best_ckpt_selection_eligible(args, epoch=19))

    def test_describe_pending_mentions_gallery_loss_ramp(self):
        args = _args(
            best_ckpt_metric="fusion_gallery_mrr",
            gallery_loss_weight=1.0,
            retrieval_start_epoch=15,
            gallery_loss_ramp_epochs=5,
            gallery_hard_neg_enable=False,
        )

        msg = train.describe_best_ckpt_selection_pending(args, epoch=18)

        self.assertIn("fusion_gallery", msg)


class GalleryHardConfigTests(unittest.TestCase):
    def test_alignment_only_hard_mining_inactive(self):
        """alignment_only has gallery_loss_weight=0, so hard mining never
        activates even if the default config enables it."""
        args = _args(
            gallery_loss_weight=0.0,
            gallery_hard_neg_enable=True,
            retrieval_start_epoch=10,
            gallery_hard_neg_start_after_retrieval_epochs=2,
            gallery_hard_neg_ramp_epochs=2,
            gallery_hard_neg_weight=0.5,
        )
        self.assertEqual(train.compute_gallery_hard_neg_weight(args, epoch=20), 0.0)

    def test_hard_neg_full_activation_reachable_when_training_exceeds_full_epoch(self):
        args = _args(
            gallery_hard_neg_enable=True,
            gallery_loss_weight=1.0,
            retrieval_start_epoch=10,
            gallery_hard_neg_start_after_retrieval_epochs=2,
            gallery_hard_neg_ramp_epochs=2,
            epochs=15,
        )
        self.assertTrue(train.gallery_hard_neg_full_activation_reachable(args))

    def test_hard_neg_full_activation_unreachable_when_epochs_insufficient(self):
        args = _args(
            gallery_hard_neg_enable=True,
            gallery_loss_weight=1.0,
            retrieval_start_epoch=10,
            gallery_hard_neg_start_after_retrieval_epochs=2,
            gallery_hard_neg_ramp_epochs=2,
            epochs=13,
        )
        self.assertFalse(train.gallery_hard_neg_full_activation_reachable(args))

    def test_hard_neg_full_activation_unreachable_when_disabled(self):
        args = _args(
            gallery_hard_neg_enable=False,
            gallery_loss_weight=1.0,
            epochs=100,
        )
        self.assertFalse(train.gallery_hard_neg_full_activation_reachable(args))

    def test_hard_neg_full_activation_unreachable_when_gallery_loss_disabled(self):
        args = _args(
            gallery_hard_neg_enable=True,
            gallery_loss_weight=0.0,
            epochs=100,
        )
        self.assertFalse(train.gallery_hard_neg_full_activation_reachable(args))


class ClassificationLossBucketTests(unittest.TestCase):
    def test_four_bucket_weights_are_normalized(self):
        args = _args(
            cls_aligned_pos_weight=0.5,
            cls_neg_gw_weight=0.2,
            cls_mismatched_weight=0.15,
            cls_external_neg_weight=0.15,
        )
        losses = [
            torch.tensor(1.0),
            torch.tensor(2.0),
            torch.tensor(3.0),
            torch.tensor(4.0),
        ]
        out = train.compute_weighted_cls_loss(*losses, args)
        expected = (0.5 * 1.0 + 0.2 * 2.0 + 0.15 * 3.0 + 0.15 * 4.0)
        self.assertAlmostEqual(out.item(), expected)

    def test_missing_buckets_are_renormalized(self):
        args = _args(
            cls_aligned_pos_weight=0.5,
            cls_neg_gw_weight=0.2,
            cls_mismatched_weight=0.15,
            cls_external_neg_weight=0.15,
        )
        out = train.compute_weighted_cls_loss(
            torch.tensor(1.0),
            None,
            torch.tensor(3.0),
            None,
            args,
        )
        expected = (0.5 * 1.0 + 0.15 * 3.0) / 0.65
        self.assertAlmostEqual(out.item(), expected, places=6)

    def test_legacy_weight_keys_are_used_as_fallback(self):
        args = _args(
            cls_pos_weight=1.0,
            cls_neg_weight=2.0,
            cls_extra_neg_weight=3.0,
        )
        out = train.compute_weighted_cls_loss(
            torch.tensor(1.0),
            torch.tensor(2.0),
            torch.tensor(3.0),
            torch.tensor(4.0),
            args,
        )
        expected = (1.0 * 1.0 + 2.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0) / 8.0
        self.assertAlmostEqual(out.item(), expected)


class NegGwGuardrailTests(unittest.TestCase):
    def _metrics(self, recalls, counts):
        strata = {
            key: {"recall": float(recall), "count": int(count)}
            for key, recall, count in zip(train.NEG_GW_STRATA_KEYS, recalls, counts)
        }
        return {"neg_gw_strata": strata}

    def test_guardrail_passes_when_all_strata_meet_threshold(self):
        args = _args(neg_gw_guardrail_enable=True, neg_gw_guardrail_recall=0.9)
        metrics = self._metrics([0.95, 0.92, 0.98, 0.91], [10, 12, 8, 9])
        info = train.resolve_neg_gw_guardrail(args, metrics)
        self.assertTrue(info["met"])
        self.assertAlmostEqual(info["min_recall"], 0.91)

    def test_guardrail_fails_when_one_stratum_is_below_threshold(self):
        args = _args(neg_gw_guardrail_enable=True, neg_gw_guardrail_recall=0.9)
        metrics = self._metrics([0.95, 0.80, 0.98, 0.91], [10, 12, 8, 9])
        info = train.resolve_neg_gw_guardrail(args, metrics)
        self.assertFalse(info["met"])
        self.assertIn("min recall", info["reason"])

    def test_guardrail_fails_when_a_stratum_is_missing(self):
        args = _args(neg_gw_guardrail_enable=True, neg_gw_guardrail_recall=0.9)
        metrics = self._metrics([0.95, 0.95, 0.95], [10, 12, 8])
        info = train.resolve_neg_gw_guardrail(args, metrics)
        self.assertFalse(info["met"])
        self.assertIn("missing", info["reason"])

    def test_disabled_guardrail_is_always_met(self):
        args = _args(neg_gw_guardrail_enable=False, neg_gw_guardrail_recall=0.9)
        info = train.resolve_neg_gw_guardrail(args, {})
        self.assertTrue(info["met"])


class CurrentScheduleConfigTests(unittest.TestCase):
    REMOVED_TRAINING_KEYS = {
        "test_data_path",
        "test_steps",
        "enable_ood_monitoring",
        "hard_neg_start_epoch",
        "hard_neg_ramp_epochs",
        "semi_hard",
        "semi_hard_margin",
        "hardneg_fallback_mode",
        "hardneg_memory_bank_enable",
        "hardneg_memory_bank_size",
        "hardneg_memory_topk",
        "hardneg_memory_warmup_steps",
        "hardneg_memory_interval",
        "hardneg_memory_max_rows",
        "mask_itc",
        "ablation_description",
        "skip_epoch_checkpoints",
    }

    RUN_CONFIGS = (
        "MAGIKS_BNS_NSBH_full.json",
        "MAGIKS_BNS_NSBH_hard_mining.json",
        "MAGIKS_BNS_NSBH_no_retrieval_loss.json",
        "MAGIKS_BNS_NSBH_no_itc_loss.json",
        "MAGIKS_BNS_NSBH_no_cls_loss.json",
        "MAGIKS_BNS_NSBH_no_cross_atten.json",
        "MAGIKS_BNS_NSBH_no_gallery_loss.json",
        "MAGIKS_BNS_NSBH_no_fusion.json",
    )

    def _load_json(self, *parts):
        import json

        path = os.path.join(MODEL_DIR, *parts)
        with open(path) as f:
            return json.load(f)

    def _merged_config(self, run_config):
        root = os.path.join(MODEL_DIR, "args")
        return train.load_merged_json_config(
            default_json_config=os.path.join(
                root, "defaults", "MAGIKS_BNS_NSBH_default.json"
            ),
            json_config=os.path.join(root, run_config),
        )

    def test_default_uses_requested_three_stage_schedule(self):
        cfg = self._merged_config("MAGIKS_BNS_NSBH_full.json")

        self.assertTrue(cfg["staged_training_enable"])
        self.assertEqual(cfg["stage_alignment_epochs"], 8)
        self.assertEqual(cfg["stage_head_epochs"], 4)
        self.assertEqual(cfg["stage_joint_itc_start_weight"], 0.5)
        self.assertEqual(cfg["stage_joint_itc_end_weight"], 0.25)
        self.assertEqual(cfg["encoder_lr_ratio"], 0.1)
        self.assertTrue(cfg["neg_gw_guardrail_enable"])
        self.assertEqual(cfg["neg_gw_guardrail_recall"], 0.85)
        self.assertEqual(cfg["cls_aligned_pos_weight"], 0.5)
        self.assertEqual(cfg["cls_neg_gw_weight"], 0.2)
        self.assertEqual(cfg["cls_mismatched_weight"], 0.15)
        self.assertEqual(cfg["cls_external_neg_weight"], 0.15)
        self.assertEqual(cfg["gallery_loss_weight"], 1.0)
        self.assertEqual(cfg["retrieval_start_epoch"], 10)
        self.assertEqual(cfg["gallery_loss_ramp_epochs"], 5)

    def test_hard_mining_variant_keeps_gallery_hard_mining_disabled(self):
        cfg = self._merged_config("MAGIKS_BNS_NSBH_hard_mining.json")

        self.assertEqual(cfg["retrieval_start_epoch"], 10)
        self.assertEqual(cfg["gallery_loss_weight"], 1.0)
        self.assertFalse(cfg["gallery_hard_neg_enable"])

    def test_hard_disabled_ablations_keep_gallery_hard_mining_off(self):
        expected = {
            "MAGIKS_BNS_NSBH_hard_mining.json": (10, 1.0),
            "MAGIKS_BNS_NSBH_no_gallery_loss.json": (0, 0.0),
            "MAGIKS_BNS_NSBH_no_fusion.json": (10, 1.0),
        }

        for run_config, (retrieval_start, gallery_weight) in expected.items():
            with self.subTest(run_config=run_config):
                cfg = self._merged_config(run_config)
                self.assertEqual(cfg["retrieval_start_epoch"], retrieval_start)
                self.assertEqual(cfg["gallery_loss_weight"], gallery_weight)
                self.assertFalse(cfg["gallery_hard_neg_enable"])

    def test_current_training_configs_do_not_contain_removed_keys(self):
        configs = [
            ("defaults", "MAGIKS_BNS_NSBH_default.json"),
            *((name,) for name in self.RUN_CONFIGS),
        ]
        for parts in configs:
            with self.subTest(config="/".join(parts)):
                cfg = self._load_json("args", *parts)
                self.assertFalse(self.REMOVED_TRAINING_KEYS & set(cfg))

    def test_current_training_configs_keep_retained_feature_switches(self):
        cfg = self._merged_config("MAGIKS_BNS_NSBH_full.json")

        self.assertIn("use_similarity_as_cls_input", cfg)
        self.assertIn("use_cred_level_feature", cfg)
        self.assertIn("hardneg_time_window_days", cfg)
        self.assertIn("hardneg_min_candidates", cfg)

    def test_current_training_config_keys_are_supported_or_shell_only(self):
        import ast

        train_path = os.path.join(MODEL_DIR, "scripts", "train", "train.py")
        with open(train_path) as f:
            tree = ast.parse(f.read())
        parser_keys = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            opts = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            dest = None
            for kw in node.keywords:
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    dest = kw.value.value
            if dest is None and opts:
                option_names = [opt for opt in opts if opt.startswith("-")]
                dest = (option_names[-1] if option_names else opts[0]).lstrip("-").replace("-", "_")
            if dest:
                parser_keys.add(dest)

        allowed = parser_keys | {"stage_to_jobfs"}
        configs = [
            ("defaults", "MAGIKS_BNS_NSBH_default.json"),
            *((name,) for name in self.RUN_CONFIGS),
        ]
        for parts in configs:
            with self.subTest(config="/".join(parts)):
                cfg = self._load_json("args", *parts)
                self.assertFalse(set(cfg) - allowed)


if __name__ == "__main__":
    unittest.main()
