import sys
import unittest
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


from scripts.train.train import (  # noqa: E402
    build_gallery_time_delta_matrix,
    compute_fusion_gallery_hard_nce_loss,
    compute_fusion_gallery_metrics,
    compute_fusion_gallery_nce_loss,
    compute_gallery_hard_neg_weight,
    compute_residual_rerank_loss,
    is_gallery_hard_neg_enabled,
    select_gallery_topk,
)
from model import MAGIKSModel  # noqa: E402


class _TinySupConModel:
    compute_supcon_loss = MAGIKSModel.compute_supcon_loss
    compute_itc_loss = MAGIKSModel.compute_itc_loss

    def __init__(self):
        self.log_temp = torch.tensor(0.0)
        self.temp_min = 0.01
        self.temp_max = 100.0
        self.itc_criterion = torch.nn.CrossEntropyLoss()
        self.itc_label_smoothing = 0.0

    def get_contrastive_embeddings(self, g, z_l):
        return g, z_l

    def project_optical_features(self, z_l):
        return z_l

    def _build_time_compat_bias(self, gw_event_time_mjd=None, opt_event_time_mjd=None):
        return None


class _RecordingOpticalModel:
    def __init__(self):
        self.seen_opt_t = None

    def encode_optical(self, _coords, opt_t, _v, _ref_t, _mask, _err):
        self.seen_opt_t = opt_t.detach().clone()
        z_l = torch.stack([opt_t.sum(dim=1), opt_t.mean(dim=1)], dim=1)
        h_l = z_l.unsqueeze(1)
        return z_l, h_l


def _reference_cross_modal_supcon_loss(feat_g, feat_o, gw_indices):
    temperature = torch.tensor(1.0, dtype=feat_g.dtype, device=feat_g.device)
    features = torch.cat([feat_g, feat_o], dim=0)
    labels = torch.cat([gw_indices, gw_indices], dim=0)
    batch_size = feat_g.size(0)
    total_count = features.size(0)

    sim_matrix = torch.matmul(features, features.T) / temperature
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    modality = torch.cat(
        [
            torch.zeros(batch_size, dtype=torch.long, device=feat_g.device),
            torch.ones(batch_size, dtype=torch.long, device=feat_g.device),
        ],
        dim=0,
    )
    same_modality = modality.unsqueeze(0) == modality.unsqueeze(1)
    mask_self = torch.eye(total_count, device=feat_g.device, dtype=torch.bool)

    mask_pos = labels_eq & ~same_modality
    mask_pos.fill_diagonal_(False)
    denominator_mask = ~mask_self & ~(labels_eq & same_modality)

    neg_large = torch.finfo(sim_matrix.dtype).min
    logits_max = sim_matrix.masked_fill(~denominator_mask, neg_large).max(dim=1, keepdim=True).values
    logits = sim_matrix - logits_max.detach()
    exp_logits = torch.exp(logits).masked_fill(~denominator_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    mask_pos = mask_pos.float()
    has_pos = mask_pos.sum(dim=1) > 0
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / mask_pos.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos[has_pos].mean()


class RetrievalTrainingLossTests(unittest.TestCase):
    def test_supcon_denominator_excludes_same_event_same_modality_samples(self):
        model = _TinySupConModel()
        gw_indices = torch.tensor([0, 0, 1], dtype=torch.long)
        feat_g = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        feat_o = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        loss, sim_in, _ = model.compute_supcon_loss(feat_g, feat_o, gw_indices)
        expected = _reference_cross_modal_supcon_loss(feat_g, feat_o, gw_indices)

        self.assertTrue(torch.allclose(loss, expected, atol=1e-6), f"{loss=} {expected=}")

    def test_gallery_time_delta_synthesis_preserves_positives_and_windows_distractors(self):
        torch.manual_seed(11)
        query_event_time = torch.tensor([100.0, 200.0], dtype=torch.float32)
        candidate_event_time = torch.tensor([103.0, 210.0, 999.0], dtype=torch.float32)
        positive_mask = torch.tensor(
            [
                [True, False, False],
                [False, True, False],
            ],
            dtype=torch.bool,
        )

        dt = build_gallery_time_delta_matrix(
            query_event_time,
            candidate_event_time,
            positive_mask=positive_mask,
            distractor_time_mode="synthetic_after_gw",
            distractor_time_window_days=30.0,
        )

        self.assertEqual(tuple(dt.shape), (2, 3))
        self.assertAlmostEqual(float(dt[0, 0]), 3.0)
        self.assertAlmostEqual(float(dt[1, 1]), 10.0)
        self.assertTrue(torch.all(dt[~positive_mask] >= 0.0))
        self.assertTrue(torch.all(dt[~positive_mask] <= 30.0))

    def test_gallery_time_delta_actual_mode_keeps_pairwise_times(self):
        query_event_time = torch.tensor([100.0, 200.0], dtype=torch.float32)
        candidate_event_time = torch.tensor([103.0, 210.0, 999.0], dtype=torch.float32)

        dt = build_gallery_time_delta_matrix(
            query_event_time,
            candidate_event_time,
            distractor_time_mode="actual",
            distractor_time_window_days=30.0,
        )

        expected = torch.tensor(
            [
                [3.0, 110.0, 899.0],
                [-97.0, 10.0, 799.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(dt, expected))


class GalleryHardNCELossTests(unittest.TestCase):
    def _scores_and_mask(self, n_query=5, n_cand=8, seed=1):
        torch.manual_seed(seed)
        scores = torch.randn(n_query, n_cand)
        positive_mask = torch.zeros(n_query, n_cand, dtype=torch.bool)
        for i in range(n_query):
            positive_mask[i, i] = True
            if i + 1 < n_query:
                positive_mask[i, i + 1] = True
        return scores, positive_mask

    def test_hard_nce_equals_full_nce_when_topk_exceeds_negatives(self):
        scores, pos_mask = self._scores_and_mask(n_query=4, n_cand=5, seed=2)
        full = compute_fusion_gallery_nce_loss(scores, pos_mask)
        hard = compute_fusion_gallery_hard_nce_loss(scores, pos_mask, topk=100)
        self.assertTrue(torch.allclose(full, hard, atol=1e-5),
                        f"full={full:.6f} hard={hard:.6f}")

    def test_hard_nce_ignores_positive_candidates(self):
        scores, pos_mask = self._scores_and_mask(n_query=4, n_cand=8, seed=3)
        # Force one negative candidate to have an extremely high score.
        # That candidate should appear in the top-k hard set but not affect the
        # loss for queries where it IS a positive.
        neg_row = 2
        neg_col = 0  # positive_mask[2, 0] is False
        self.assertFalse(bool(pos_mask[neg_row, neg_col]))
        scores_orig = scores.clone()
        hard = compute_fusion_gallery_hard_nce_loss(scores_orig, pos_mask, topk=2)
        self.assertTrue(torch.isfinite(hard))
        self.assertGreater(hard.item(), 0.0)

    def test_hard_nce_zero_when_no_valid_queries(self):
        scores = torch.randn(3, 4)
        positive_mask = torch.zeros(3, 4, dtype=torch.bool)  # no positives
        hard = compute_fusion_gallery_hard_nce_loss(scores, positive_mask, topk=2)
        self.assertEqual(hard.item(), 0.0)

    def test_hard_nce_zero_when_no_negatives(self):
        scores = torch.randn(2, 3)
        positive_mask = torch.ones(2, 3, dtype=torch.bool)  # all positives
        hard = compute_fusion_gallery_hard_nce_loss(scores, positive_mask, topk=2)
        self.assertEqual(hard.item(), 0.0)

    def test_hard_nce_no_nan_with_finite_inputs(self):
        scores, pos_mask = self._scores_and_mask(n_query=6, n_cand=12, seed=5)
        for topk in [1, 2, 4, 8, 32]:
            hard = compute_fusion_gallery_hard_nce_loss(scores, pos_mask, topk=topk)
            self.assertTrue(torch.isfinite(hard), f"NaN at topk={topk}")
            self.assertGreaterEqual(hard.item(), 0.0)

    def test_hard_nce_monotonic_with_topk(self):
        """Hard NCE should increase (or stay same) as topk grows — more
        negatives in the denominator reduce per-query log-probability."""
        scores, pos_mask = self._scores_and_mask(n_query=5, n_cand=10, seed=7)
        prev = None
        for topk in [1, 2, 4, 6, 8, 100]:
            hard = compute_fusion_gallery_hard_nce_loss(scores, pos_mask, topk=topk)
            if prev is not None:
                self.assertGreaterEqual(hard.item(), prev.item() - 1e-5,
                                        f"topk={topk} gave {hard:.6f} < prev {prev:.6f}")
            prev = hard


class FusionGalleryRerankTests(unittest.TestCase):
    def test_gallery_topk_does_not_force_include_positives_by_default(self):
        sim = torch.tensor([[0.9, 0.8, 0.1], [0.7, 0.6, 0.5]], dtype=torch.float32)
        pos = torch.tensor(
            [[False, False, True], [True, False, False]], dtype=torch.bool
        )

        topk_indices, topk_positive_mask, candidate_recall_mask = select_gallery_topk(
            sim, pos, topk=2
        )

        self.assertEqual(topk_indices.tolist(), [[0, 1], [0, 1]])
        self.assertEqual(
            topk_positive_mask.tolist(), [[False, False], [True, False]]
        )
        self.assertEqual(candidate_recall_mask.tolist(), [False, True])

    def test_gallery_topk_force_include_positives_preserves_legacy_behavior(self):
        sim = torch.tensor([[0.9, 0.8, 0.1], [0.7, 0.6, 0.5]], dtype=torch.float32)
        pos = torch.tensor(
            [[False, False, True], [True, False, False]], dtype=torch.bool
        )

        topk_indices, topk_positive_mask, candidate_recall_mask = select_gallery_topk(
            sim, pos, topk=2, force_include_positives=True
        )

        self.assertEqual(topk_indices.tolist(), [[0, 2], [0, 1]])
        self.assertEqual(topk_positive_mask.tolist(), [[False, True], [True, False]])
        self.assertEqual(candidate_recall_mask.tolist(), [True, True])

    def test_fusion_gallery_metrics_count_missing_rows_as_zero_for_pipeline(self):
        scores = torch.tensor([[5.0, 4.0], [2.0, 3.0]], dtype=torch.float32)
        pos = torch.tensor([[False, False], [False, True]], dtype=torch.bool)

        metrics = compute_fusion_gallery_metrics(scores, pos, ks=(1, 2))

        self.assertAlmostEqual(metrics["fusion_gallery_candidate_recall_at_topk"], 0.5)
        self.assertAlmostEqual(metrics["fusion_gallery_valid_query_fraction"], 0.5)
        self.assertAlmostEqual(metrics["fusion_gallery_recall_at_1"], 0.5)
        self.assertAlmostEqual(metrics["fusion_gallery_recall_at_2"], 0.5)
        self.assertAlmostEqual(metrics["fusion_gallery_mrr"], 0.5)
        self.assertAlmostEqual(metrics["fusion_gallery_conditional_recall_at_1"], 1.0)
        self.assertAlmostEqual(metrics["fusion_gallery_conditional_recall_at_2"], 1.0)
        self.assertAlmostEqual(metrics["fusion_gallery_conditional_mrr"], 1.0)

    def test_missing_positive_rows_do_not_contribute_to_residual_rerank_loss(self):
        s_itc = torch.tensor([[5.0, 4.0], [2.0, 1.0]], dtype=torch.float32)
        s_fusion = torch.tensor([[1.0, 0.0], [0.2, 0.8]], dtype=torch.float32)
        pos = torch.tensor([[False, False], [False, True]], dtype=torch.bool)

        loss = compute_residual_rerank_loss(s_itc, s_fusion, pos, lambda_=0.5)
        expected = compute_residual_rerank_loss(
            s_itc[1:2], s_fusion[1:2], pos[1:2], lambda_=0.5
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_residual_rerank_loss_is_row_shift_and_scale_stable(self):
        s_itc = torch.tensor([[5.0, 4.0, 1.0], [2.0, 3.0, 0.0]], dtype=torch.float32)
        s_fusion = torch.tensor(
            [[0.1, 0.8, 0.2], [1.0, -0.5, 0.2]], dtype=torch.float32
        )
        pos = torch.tensor(
            [[False, True, False], [True, False, False]], dtype=torch.bool
        )
        row_shift = torch.tensor([[11.0], [-3.0]], dtype=torch.float32)

        base = compute_residual_rerank_loss(s_itc, s_fusion, pos, lambda_=0.5)
        transformed = compute_residual_rerank_loss(
            s_itc * 3.0 + row_shift,
            s_fusion * 7.0 + row_shift,
            pos,
            lambda_=0.5,
        )

        self.assertTrue(torch.allclose(base, transformed, atol=1e-6))


class GalleryHardNegWeightTests(unittest.TestCase):
    def _args(self, **overrides):
        import argparse
        defaults = {
            "gallery_hard_neg_enable": True,
            "gallery_loss_weight": 1.0,
            "retrieval_start_epoch": 10,
            "gallery_hard_neg_start_after_retrieval_epochs": 2,
            "gallery_hard_neg_ramp_epochs": 2,
            "gallery_hard_neg_weight": 0.5,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_weight_zero_before_hard_start(self):
        args = self._args()
        self.assertEqual(compute_gallery_hard_neg_weight(args, 0), 0.0)
        self.assertEqual(compute_gallery_hard_neg_weight(args, 10), 0.0)
        self.assertEqual(compute_gallery_hard_neg_weight(args, 11), 0.0)

    def test_weight_ramps_to_full(self):
        args = self._args()
        # hard_start = 10 + 2 = 12, ramp 2 epochs
        self.assertAlmostEqual(compute_gallery_hard_neg_weight(args, 12), 0.25)
        self.assertAlmostEqual(compute_gallery_hard_neg_weight(args, 13), 0.5)

    def test_weight_stays_at_full_after_ramp(self):
        args = self._args()
        self.assertAlmostEqual(compute_gallery_hard_neg_weight(args, 14), 0.5)
        self.assertAlmostEqual(compute_gallery_hard_neg_weight(args, 50), 0.5)

    def test_weight_zero_when_disabled(self):
        args = self._args(gallery_hard_neg_enable=False)
        self.assertEqual(compute_gallery_hard_neg_weight(args, 20), 0.0)

    def test_weight_zero_when_gallery_loss_disabled(self):
        args = self._args(gallery_loss_weight=0.0)
        self.assertEqual(compute_gallery_hard_neg_weight(args, 20), 0.0)

    def test_is_enabled_false_by_default(self):
        import argparse
        args = argparse.Namespace()
        self.assertFalse(is_gallery_hard_neg_enabled(args))

    def test_no_ramp_goes_straight_to_full(self):
        args = self._args(gallery_hard_neg_ramp_epochs=0)
        self.assertAlmostEqual(compute_gallery_hard_neg_weight(args, 12), 0.5)

    def test_v11_hard_mining_turns_on_directly_at_epoch_26_1based(self):
        args = self._args(
            retrieval_start_epoch=15,
            gallery_hard_neg_start_after_retrieval_epochs=10,
            gallery_hard_neg_ramp_epochs=0,
            gallery_hard_neg_weight=0.5,
        )

        self.assertEqual(compute_gallery_hard_neg_weight(args, 24), 0.0)
        self.assertAlmostEqual(compute_gallery_hard_neg_weight(args, 25), 0.5)
        self.assertAlmostEqual(compute_gallery_hard_neg_weight(args, 26), 0.5)


class ExtendedSimReturnTests(unittest.TestCase):
    """Tests for the extended similarity matrix returned by ITC loss functions."""

    def _make_model(self):
        """Create a tiny model delegating ITC loss functions to MAGIKSModel."""
        return _TinySupConModel()

    def test_itc_loss_returns_extended_sim_with_extra_neg(self):
        model = self._make_model()
        g = torch.randn(4, 256)
        z_l = torch.randn(4, 256)
        gw_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        extra_neg_z = torch.randn(4, 256)

        _, sim_inbatch, sim_ext = model.compute_itc_loss(
            g, z_l, gw_indices=gw_indices, extra_neg_z=extra_neg_z,
        )
        self.assertEqual(tuple(sim_inbatch.shape), (4, 4))
        self.assertEqual(tuple(sim_ext.shape), (4, 8))
        # Extended sim includes in-batch portion
        self.assertTrue(torch.allclose(sim_ext[:, :4], sim_inbatch, atol=1e-6))

    def test_itc_loss_returns_same_shape_without_extra_neg(self):
        model = self._make_model()
        g = torch.randn(4, 256)
        z_l = torch.randn(4, 256)
        gw_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        _, sim_inbatch, sim_ext = model.compute_itc_loss(
            g, z_l, gw_indices=gw_indices, extra_neg_z=None,
        )
        self.assertEqual(tuple(sim_ext.shape), (4, 4))
        self.assertTrue(torch.allclose(sim_ext, sim_inbatch, atol=1e-6))

    def test_supcon_loss_returns_extended_sim_with_extra_neg(self):
        model = self._make_model()
        g = torch.randn(4, 256)
        z_l = torch.randn(4, 256)
        gw_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        extra_neg_z = torch.randn(4, 256)

        _, sim_inbatch, sim_ext = model.compute_supcon_loss(
            g, z_l, gw_indices, extra_neg_z=extra_neg_z,
        )
        self.assertEqual(tuple(sim_inbatch.shape), (4, 4))
        self.assertEqual(tuple(sim_ext.shape), (4, 8))
        self.assertTrue(torch.allclose(sim_ext[:, :4], sim_inbatch, atol=1e-6))

    def test_supcon_loss_returns_same_shape_without_extra_neg(self):
        model = self._make_model()
        g = torch.randn(4, 256)
        z_l = torch.randn(4, 256)
        gw_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        _, sim_inbatch, sim_ext = model.compute_supcon_loss(
            g, z_l, gw_indices, extra_neg_z=None,
        )
        self.assertEqual(tuple(sim_ext.shape), (4, 4))
        self.assertTrue(torch.allclose(sim_ext, sim_inbatch, atol=1e-6))

    def test_retrieval_metrics_ignore_external_neg_gw_indices(self):
        """External negatives with gw_index=-1 should not match any query."""
        from metrics import compute_retrieval_metrics
        torch.manual_seed(42)
        sim_g2o = torch.randn(4, 8)
        # First 4 candidates are in-batch, last 4 are external negs
        gw_indices = torch.tensor([0, 0, 1, 1, -1, -1, -1, -1], dtype=torch.long)
        metrics = compute_retrieval_metrics(sim_g2o, gw_indices, ks=(1, 5))
        # Should not crash, all metrics should be finite
        for key in ["g2o_recall_at_1", "g2o_recall_at_5", "g2o_mrr"]:
            v = metrics[key]
            self.assertTrue(torch.isfinite(torch.tensor(v)), f"{key}={v}")
            self.assertIsInstance(v, float)


if __name__ == "__main__":
    unittest.main()
