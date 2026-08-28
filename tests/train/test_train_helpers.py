import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Model")
)
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


def _curriculum_args(**overrides):
    values = {
        "staged_training_enable": True,
        "itc_weight": 1.0,
        "cls_weight": 1.0,
        "gallery_loss_weight": 1.0,
        "stage_itc_epochs": 8,
        "stage_cls_ramp_epochs": 4,
        "stage_retrieval_ramp_epochs": 4,
        "stage_joint_itc_start_weight": 0.5,
        "stage_joint_itc_end_weight": 0.25,
        "epochs": 100,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class SequentialCurriculumTests(unittest.TestCase):
    def test_atomic_checkpoint_save_preserves_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            with open(path, "wb") as handle:
                handle.write(b"previous")

            with mock.patch.object(torch, "save", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    train.save_training_checkpoint_atomic(
                        {"epoch": 1}, path, max_attempts=1
                    )

            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"previous")
            self.assertEqual(os.listdir(directory), ["checkpoint.pth"])

    def test_atomic_checkpoint_save_retries_transient_failure(self):
        original_save = torch.save
        attempts = 0

        def flaky_save(value, path):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary filesystem failure")
            original_save(value, path)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            with (
                mock.patch.object(torch, "save", side_effect=flaky_save),
                mock.patch.object(train.time, "sleep") as sleep,
            ):
                train.save_training_checkpoint_atomic(
                    {"epoch": 50}, path, max_attempts=2
                )
            loaded = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(attempts, 2)
        sleep.assert_called_once_with(1)
        self.assertEqual(loaded["epoch"], 50)

    def test_last_checkpoint_failure_is_nonfatal(self):
        with mock.patch.object(
            train,
            "save_training_checkpoint_atomic",
            side_effect=RuntimeError("filesystem unavailable"),
        ):
            saved = train.save_last_checkpoint_resilient({}, "checkpoint.pth")

        self.assertFalse(saved)

    def test_restore_rng_state_passes_cpu_tensors_to_torch_generators(self):
        state = train.capture_rng_state()
        state["cuda"] = [state["torch"].clone()]

        with (
            mock.patch.object(torch, "set_rng_state") as set_cpu_state,
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "set_rng_state_all") as set_cuda_states,
        ):
            train.restore_rng_state(state)

        self.assertEqual(set_cpu_state.call_args.args[0].device.type, "cpu")
        restored_cuda_states = set_cuda_states.call_args.args[0]
        self.assertTrue(restored_cuda_states)
        self.assertTrue(all(item.device.type == "cpu" for item in restored_cuda_states))

    def test_training_checkpoint_loads_numpy_rng_state_with_restricted_loader(self):
        checkpoint = {"rng_state": {"numpy": np.random.get_state()}, "epoch": 29}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            torch.save(checkpoint, path)

            loaded = train.load_training_checkpoint(path, map_location="cpu")

        self.assertEqual(loaded["epoch"], 29)
        self.assertEqual(loaded["rng_state"]["numpy"][0], "MT19937")

    def test_rung_uses_full_training_schedule_horizon(self):
        rung = _curriculum_args(epochs=30, training_schedule_total_epochs=100)
        full = _curriculum_args(epochs=100, training_schedule_total_epochs=100)
        for epoch in (0, 8, 16, 29):
            self.assertEqual(
                train.resolve_sequential_curriculum(rung, epoch),
                train.resolve_sequential_curriculum(full, epoch),
            )

    def test_cosine_lr_prefix_is_identical_across_rungs(self):
        common = dict(
            lr=5e-4,
            min_lr=2e-5,
            warmup_epochs=5,
            training_schedule_total_epochs=100,
        )
        rung = argparse.Namespace(epochs=30, **common)
        full = argparse.Namespace(epochs=100, **common)
        for step in (0, 499, 500, 1500, 2999):
            self.assertAlmostEqual(
                train._common_lr_scale(rung, step, 100),
                train._common_lr_scale(full, step, 100),
            )

    def test_validation_interval_always_includes_rung_end(self):
        self.assertFalse(train.should_run_validation(28, 30, 5))
        self.assertTrue(train.should_run_validation(29, 30, 5))

    def test_scheduler_resume_matches_uninterrupted_run(self):
        args = argparse.Namespace(
            epochs=30,
            training_schedule_total_epochs=100,
            lr_scheduler="cosine",
            lr=5e-4,
            min_lr=2e-5,
            warmup_epochs=5,
            staged_training_enable=False,
        )
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=args.lr)
        scheduler = train.build_lr_scheduler(optimizer, args, 10, 0)
        for _ in range(300):
            optimizer.step()
            scheduler.step()
        expected_lr = optimizer.param_groups[0]["lr"]
        state = optimizer.state_dict()

        resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
        resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=args.lr)
        resumed_optimizer.load_state_dict(state)
        train.build_lr_scheduler(resumed_optimizer, args, 10, 300)
        self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], expected_lr)

    def test_full_boundary_weights(self):
        args = _curriculum_args()
        expected = {
            0: ("itc", 1.0, 0.0, 0.0),
            7: ("itc", 1.0, 0.0, 0.0),
            8: ("cls_intro", 0.0, 0.25, 0.0),
            11: ("cls_intro", 0.0, 1.0, 0.0),
            12: ("retrieval_intro", 0.0, 1.0, 0.25),
            15: ("retrieval_intro", 0.0, 1.0, 1.0),
            16: ("joint", 0.5, 1.0, 1.0),
            99: ("joint", 0.25, 1.0, 1.0),
        }
        for epoch, (phase, itc, cls, retrieval) in expected.items():
            with self.subTest(epoch=epoch):
                state = train.resolve_sequential_curriculum(args, epoch)
                self.assertEqual(state["phase"], phase)
                self.assertAlmostEqual(state["weights"]["itc"], itc)
                self.assertAlmostEqual(state["weights"]["cls"], cls)
                self.assertAlmostEqual(state["weights"]["retrieval"], retrieval)

    def test_disabled_losses_skip_their_phases(self):
        cases = {
            "no_retrieval": (
                _curriculum_args(gallery_loss_weight=0.0),
                ["itc", "cls_intro", "joint"],
                12,
            ),
            "no_cls": (
                _curriculum_args(cls_weight=0.0),
                ["itc", "retrieval_intro", "joint"],
                12,
            ),
            "no_itc": (
                _curriculum_args(itc_weight=0.0),
                ["cls_intro", "retrieval_intro", "joint"],
                8,
            ),
            "no_fusion": (
                _curriculum_args(cls_weight=0.0),
                ["itc", "retrieval_intro", "joint"],
                12,
            ),
            "no_cross": (
                _curriculum_args(),
                ["itc", "cls_intro", "retrieval_intro", "joint"],
                16,
            ),
        }
        for name, (args, phases, joint_start) in cases.items():
            with self.subTest(name=name):
                resolved = train.build_sequential_curriculum(args)
                self.assertEqual([phase["name"] for phase in resolved], phases)
                self.assertEqual(resolved[-1]["start"], joint_start)

    def test_no_itc_is_end_to_end_and_itc_stays_zero(self):
        args = _curriculum_args(itc_weight=0.0)
        for epoch in range(args.epochs):
            self.assertEqual(train.compute_itc_weight(args, epoch), 0.0)
        state = train.resolve_sequential_curriculum(args, 0)
        self.assertEqual(state["phase"], "cls_intro")
        self.assertEqual(state["train_roles"], {"encoder", "head"})

    def test_model_freezing_matches_phase_roles(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 2)
                self.fusion = torch.nn.Linear(2, 1)
                self.log_temp = torch.nn.Parameter(torch.tensor(0.0))

        model = TinyModel()
        args = _curriculum_args()
        train.configure_model_for_stage(model, args, "itc")
        self.assertTrue(model.encoder.weight.requires_grad)
        self.assertFalse(model.fusion.weight.requires_grad)
        self.assertTrue(model.log_temp.requires_grad)
        train.configure_model_for_stage(model, args, "cls_intro")
        self.assertFalse(model.encoder.weight.requires_grad)
        self.assertTrue(model.fusion.weight.requires_grad)
        self.assertFalse(model.log_temp.requires_grad)


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


    def test_mixed_gallery_metric_is_available_when_validation_is_enabled(self):
        args = _args(
            best_ckpt_metric="mixed_gallery_macro_retrieval_score",
            validation_gallery_enable=True,
        )

        self.assertEqual(
            train.compute_best_ckpt_metric_ready_epoch(args),
            0,
        )


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
        self.assertTrue(train.is_best_ckpt_selection_eligible(args, epoch=11))

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
        self.assertFalse(train.is_best_ckpt_selection_eligible(args, epoch=12))

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
        self.assertTrue(train.is_best_ckpt_selection_eligible(args, epoch=13))

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
        self.assertFalse(train.is_best_ckpt_selection_eligible(args, epoch=12))

    def test_alignment_only_eligible_immediately(self):
        """alignment_only (gallery_loss_weight=0) skips hard ramp gate."""
        args = _args(
            best_ckpt_metric="g2o_mrr",
            gallery_loss_weight=0.0,
            gallery_hard_neg_enable=True,
            retrieval_start_epoch=999,
        )
        self.assertTrue(train.is_best_ckpt_selection_eligible(args, epoch=5))

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
            staged_training_enable=True,
            itc_weight=1.0,
            cls_weight=1.0,
            gallery_hard_neg_enable=True,
            gallery_loss_weight=1.0,
            stage_itc_epochs=8,
            stage_cls_ramp_epochs=2,
            stage_retrieval_ramp_epochs=1,
            gallery_hard_neg_start_after_retrieval_epochs=2,
            gallery_hard_neg_ramp_epochs=2,
            epochs=15,
        )
        self.assertTrue(train.gallery_hard_neg_full_activation_reachable(args))

    def test_hard_neg_full_activation_unreachable_when_epochs_insufficient(self):
        args = _args(
            staged_training_enable=True,
            itc_weight=1.0,
            cls_weight=1.0,
            gallery_hard_neg_enable=True,
            gallery_loss_weight=1.0,
            stage_itc_epochs=8,
            stage_cls_ramp_epochs=2,
            stage_retrieval_ramp_epochs=1,
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
        expected = 0.5 * 1.0 + 0.2 * 2.0 + 0.15 * 3.0 + 0.15 * 4.0
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
        "MAGIKS_BNS_NSBH_fiducial_params.json",
        "MAGIKS_BNS_NSBH_no_hard_mining.json",
        "MAGIKS_BNS_NSBH_no_retrieval_loss.json",
        "MAGIKS_BNS_NSBH_no_itc_loss.json",
        "MAGIKS_BNS_NSBH_no_cls_loss.json",
        "MAGIKS_BNS_NSBH_no_cross_atten.json",
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

    def test_default_uses_hpo_v7_mixed_gallery_schedule(self):
        cfg = self._merged_config("MAGIKS_BNS_NSBH_full.json")

        self.assertAlmostEqual(cfg["lr"], 0.0005782703604715653)
        self.assertAlmostEqual(cfg["weight_decay"], 0.016857292961516286)
        self.assertEqual(cfg["gallery_loss_weight"], 0.75)
        self.assertEqual(cfg["itc_weight"], 1.5)
        self.assertEqual(cfg["cls_weight"], 0.75)
        self.assertEqual(cfg["fusion_physical_weight"], 1.5)
        self.assertTrue(cfg["staged_training_enable"])
        self.assertEqual(cfg["stage_itc_epochs"], 10)
        self.assertEqual(cfg["stage_cls_ramp_epochs"], 4)
        self.assertEqual(cfg["stage_retrieval_ramp_epochs"], 4)
        self.assertEqual(cfg["stage_joint_itc_start_weight"], 0.5)
        self.assertEqual(cfg["stage_joint_itc_end_weight"], 0.25)
        self.assertEqual(cfg["encoder_lr_ratio"], 0.1)
        self.assertTrue(cfg["neg_gw_guardrail_enable"])
        self.assertEqual(cfg["neg_gw_guardrail_recall"], 0.85)
        self.assertEqual(cfg["cls_aligned_pos_weight"], 0.5)
        self.assertEqual(cfg["cls_neg_gw_weight"], 0.2)
        self.assertEqual(cfg["cls_mismatched_weight"], 0.15)
        self.assertEqual(cfg["cls_external_neg_weight"], 0.15)
        self.assertEqual(cfg["gallery_candidate_mode"], "mixed_kn_nonkn")
        self.assertEqual(cfg["validation_gallery_mode"], "mixed_kn_nonkn")
        self.assertEqual(cfg["validation_gallery_condition"], "training_aligned")
        self.assertEqual(
            cfg["best_ckpt_metric"], "mixed_gallery_macro_retrieval_score"
        )

    def test_fiducial_params_use_archived_typical_values(self):
        cfg = self._merged_config("MAGIKS_BNS_NSBH_fiducial_params.json")

        self.assertEqual(cfg["lr"], 0.0005)
        self.assertEqual(cfg["weight_decay"], 0.02)
        self.assertEqual(cfg["gallery_loss_weight"], 1.0)
        self.assertEqual(cfg["itc_weight"], 1.0)
        self.assertEqual(cfg["cls_weight"], 1.0)
        self.assertEqual(cfg["fusion_physical_weight"], 2.0)
        self.assertEqual(cfg["stage_itc_epochs"], 8)
        self.assertEqual(cfg["stage_joint_itc_start_weight"], 0.5)
        self.assertEqual(cfg["early_stop_min_delta"], 0.01)
        self.assertEqual(cfg["max_gallery_queries"], 64)
        self.assertEqual(cfg["gallery_distractor_time_mode"], "synthetic_after_gw")
        self.assertFalse(cfg["gallery_hard_neg_enable"])
        self.assertEqual(cfg["gallery_hard_neg_start_after_retrieval_epochs"], 10)
        self.assertEqual(cfg["gallery_hard_neg_ramp_epochs"], 0)
        self.assertEqual(cfg["gallery_candidate_mode"], "mixed_kn_nonkn")
        self.assertEqual(cfg["validation_gallery_condition"], "training_aligned")

    def test_current_training_configs_use_mixed_gallery(self):
        for run_config in self.RUN_CONFIGS:
            with self.subTest(run_config=run_config):
                cfg = self._merged_config(run_config)
                self.assertEqual(cfg["gallery_candidate_mode"], "mixed_kn_nonkn")
                self.assertEqual(cfg["validation_gallery_mode"], "mixed_kn_nonkn")
                self.assertEqual(
                    cfg["validation_gallery_condition"], "training_aligned"
                )

    def test_no_hard_mining_variants_keep_gallery_hard_mining_off(self):
        expected = {
            "MAGIKS_BNS_NSBH_fiducial_params.json": 1.0,
            "MAGIKS_BNS_NSBH_no_hard_mining.json": 0.75,
            "MAGIKS_BNS_NSBH_no_retrieval_loss.json": 0.0,
            "MAGIKS_BNS_NSBH_no_fusion.json": 0.75,
        }

        for run_config, gallery_weight in expected.items():
            with self.subTest(run_config=run_config):
                cfg = self._merged_config(run_config)
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
            if (
                not isinstance(node.func, ast.Attribute)
                or node.func.attr != "add_argument"
            ):
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
                dest = (
                    (option_names[-1] if option_names else opts[0])
                    .lstrip("-")
                    .replace("-", "_")
                )
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
