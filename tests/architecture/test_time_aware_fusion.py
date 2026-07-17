import sys
import unittest
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model import CrossAttentionFusion  # noqa: E402
from scripts.train.train import (  # noqa: E402
    compute_fusion_gallery_metrics,
    compute_fusion_gallery_nce_loss,
)
from retrieval_gallery import extract_gallery_negative_abs_dt_days  # noqa: E402


class TimeAwareFusionTests(unittest.TestCase):
    def _make_inputs(self):
        torch.manual_seed(7)
        return {
            "g_feat": torch.randn(2, 4),
            "h_l": torch.randn(2, 3, 4),
            "H_gw": torch.randn(2, 5, 4),
            "g_param": torch.randn(2, 4),
            "coord_feat": torch.randn(2, 4),
            "dt_days": torch.tensor([0.0, 30.0]),
        }

    def test_time_delta_feature_adds_one_classifier_input(self):
        base = CrossAttentionFusion(
            4,
            4,
            coord_dim=4,
            attn_dim=4,
            hidden_dim=8,
            dual=True,
            fusion_mode="physical_dual_hgw",
            use_time_delta_cls_feature=False,
        )
        timed = CrossAttentionFusion(
            4,
            4,
            coord_dim=4,
            attn_dim=4,
            hidden_dim=8,
            dual=True,
            fusion_mode="physical_dual_hgw",
            use_time_delta_cls_feature=True,
            time_delta_cls_scale_days=30.0,
            time_delta_cls_clip=10.0,
        )

        self.assertEqual(base.classifier[0].in_features, 8)
        self.assertEqual(timed.classifier[0].in_features, 9)

        logits, combined, _ = timed(**self._make_inputs())
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.allclose(combined[:, -1], torch.tensor([0.0, 1.0])))

    def test_physical_and_spatial_weights_scale_combined_features(self):
        base = CrossAttentionFusion(
            4,
            4,
            coord_dim=4,
            attn_dim=4,
            hidden_dim=8,
            dual=True,
            fusion_mode="physical_dual_hgw",
            fusion_physical_weight=1.0,
            fusion_spatial_weight=1.0,
            dropout=0.0,
        ).eval()
        weighted = CrossAttentionFusion(
            4,
            4,
            coord_dim=4,
            attn_dim=4,
            hidden_dim=8,
            dual=True,
            fusion_mode="physical_dual_hgw",
            fusion_physical_weight=2.0,
            fusion_spatial_weight=0.25,
            dropout=0.0,
        ).eval()
        weighted.load_state_dict(base.state_dict())

        inputs = self._make_inputs()
        _, base_combined, _ = base(**inputs)
        _, weighted_combined, _ = weighted(**inputs)

        self.assertTrue(torch.allclose(weighted_combined[:, :4], base_combined[:, :4] * 2.0, atol=1e-6))
        self.assertTrue(torch.allclose(weighted_combined[:, 4:8], base_combined[:, 4:8] * 0.25, atol=1e-6))
        self.assertEqual(tuple(weighted_combined.shape), tuple(base_combined.shape))

    def test_gallery_nce_loss_rewards_positive_rank(self):
        positive_mask = torch.tensor(
            [
                [True, False, False],
                [False, True, False],
            ]
        )
        good_scores = torch.tensor([[4.0, 0.0, -1.0], [0.0, 4.0, -1.0]])
        bad_scores = torch.tensor([[0.0, 4.0, -1.0], [4.0, 0.0, -1.0]])

        good_loss = compute_fusion_gallery_nce_loss(good_scores, positive_mask)
        bad_loss = compute_fusion_gallery_nce_loss(bad_scores, positive_mask)
        metrics = compute_fusion_gallery_metrics(good_scores, positive_mask, ks=(1, 2))

        self.assertLess(float(good_loss), float(bad_loss))
        self.assertAlmostEqual(metrics["fusion_gallery_recall_at_1"], 1.0)
        self.assertAlmostEqual(metrics["fusion_gallery_recall_at_2"], 1.0)
        self.assertAlmostEqual(metrics["fusion_gallery_mrr"], 1.0)

    def test_gallery_dt_helper_prefers_recorded_abs_dt_then_synthetic_time(self):
        recorded = extract_gallery_negative_abs_dt_days(
            {"negative_abs_dt_days": np.asarray([3.0, 9.0], dtype=np.float32)},
            gw_event_time_mjd=100.0,
            n_negative=2,
        )
        synthetic = extract_gallery_negative_abs_dt_days(
            {"negative_synthetic_zero_time_mjd_cls_base": np.asarray([95.0, 107.0], dtype=np.float64)},
            gw_event_time_mjd=100.0,
            n_negative=2,
        )

        np.testing.assert_allclose(recorded, np.asarray([3.0, 9.0], dtype=np.float32))
        np.testing.assert_allclose(synthetic, np.asarray([5.0, 7.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
