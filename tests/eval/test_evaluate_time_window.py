import sys
import unittest
from pathlib import Path

import numpy as np
import torch


MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
sys.path.insert(0, str(MODEL_DIR))

from scripts.eval import evaluate as evaluate_module  # noqa: E402


class TestClassificationTimeWindow(unittest.TestCase):
    def test_windowed_optical_negative_sampling_prefers_actual_candidates(self):
        rng = np.random.default_rng(7)
        anchor_times = torch.tensor([100.0, 200.0])
        candidate_times = torch.tensor([95.0, 101.0, 150.0, 220.0, 260.0])

        selected, dt_days, actual_mask, next_cursor = evaluate_module._sample_time_windowed_indices_after_anchor(
            anchor_times,
            candidate_times,
            window_days=30.0,
            rng=rng,
            fallback_start=0,
        )

        self.assertEqual(selected.tolist(), [1, 3])
        self.assertTrue(torch.allclose(dt_days.cpu(), torch.tensor([1.0, 20.0])))
        self.assertEqual(actual_mask.tolist(), [True, True])
        self.assertEqual(next_cursor, 0)

    def test_windowed_sampling_falls_back_with_synthetic_dt_when_no_actual_candidate(self):
        rng = np.random.default_rng(11)
        anchor_times = torch.tensor([100.0, 200.0])
        candidate_times = torch.tensor([10.0, 90.0, 300.0])

        selected, dt_days, actual_mask, next_cursor = evaluate_module._sample_time_windowed_indices_after_anchor(
            anchor_times,
            candidate_times,
            window_days=30.0,
            rng=rng,
            fallback_start=1,
        )

        self.assertEqual(selected.tolist(), [1, 2])
        self.assertEqual(actual_mask.tolist(), [False, False])
        self.assertEqual(next_cursor, 3)
        self.assertTrue(bool(torch.all(dt_days >= 0.0)))
        self.assertTrue(bool(torch.all(dt_days <= 30.0)))

    def test_windowed_gw_negative_sampling_uses_preceding_gw_times(self):
        rng = np.random.default_rng(13)
        optical_times = torch.tensor([130.0, 210.0])
        neg_gw_times = np.array([100.0, 205.0, 300.0], dtype=np.float32)
        neg_gw_indices = np.array([5, 6, 7], dtype=np.int64)

        selected, dt_days, actual_mask = evaluate_module._sample_time_windowed_indices_before_event(
            optical_times,
            neg_gw_times,
            neg_gw_indices,
            window_days=30.0,
            rng=rng,
        )

        self.assertEqual(selected.tolist(), [5, 6])
        self.assertTrue(torch.allclose(dt_days.cpu(), torch.tensor([30.0, 5.0])))
        self.assertEqual(actual_mask.tolist(), [True, True])

    def test_signed_time_window_hard_negative_prefers_in_window_candidate(self):
        rng = np.random.default_rng(17)
        sim_g2o = torch.tensor(
            [
                [0.9, 0.2, 0.8],
                [0.3, 0.9, 0.4],
                [0.2, 0.7, 0.9],
            ],
            dtype=torch.float32,
        )
        gw_indices = torch.tensor([10, 11, 12])
        anchor_times = torch.tensor([100.0, 100.0, 100.0])
        candidate_times = torch.tensor([100.0, 105.0, 140.0])

        selected, dt_days, actual_mask = evaluate_module._sample_signed_time_window_hard_negatives(
            sim_g2o,
            gw_indices,
            anchor_times,
            candidate_times,
            window_days=30.0,
            rng=rng,
            semi_hard=True,
            semi_hard_margin=0.2,
            fallback_mode="inbatch_semihard",
        )

        self.assertEqual(selected.tolist(), [1, 0, 1])
        self.assertTrue(torch.allclose(dt_days.cpu(), torch.tensor([5.0, 0.0, 5.0])))
        self.assertEqual(actual_mask.tolist(), [True, True, True])

    def test_apply_cls_time_window_replaces_out_of_window_dt_and_counts_it(self):
        counts = evaluate_module._init_cls_time_window_counts()
        dt = torch.tensor([-5.0, 10.0, 35.0])
        rng = np.random.default_rng(19)

        out = evaluate_module._apply_cls_time_window_to_dt(
            dt,
            pair_type="positive",
            window_days=30.0,
            rng=rng,
            counts=counts,
        )

        self.assertTrue(bool(torch.all(out >= 0.0)))
        self.assertTrue(bool(torch.all(out <= 30.0)))
        self.assertTrue(torch.equal(out[1:2], torch.tensor([10.0])))
        self.assertEqual(counts["positive"]["actual_count"], 1)
        self.assertEqual(counts["positive"]["synthetic_count"], 2)
        self.assertEqual(counts["positive"]["out_of_window_count"], 2)

    def test_apply_cls_time_window_can_be_disabled(self):
        counts = evaluate_module._init_cls_time_window_counts()
        dt = torch.tensor([-5.0, 10.0, 35.0])
        rng = np.random.default_rng(19)

        out = evaluate_module._apply_cls_time_window_to_dt(
            dt,
            pair_type="positive",
            window_days=0.0,
            rng=rng,
            counts=counts,
        )

        self.assertTrue(torch.equal(out, dt))
        self.assertEqual(counts["positive"]["actual_count"], 0)
        self.assertEqual(counts["positive"]["synthetic_count"], 0)


if __name__ == "__main__":
    unittest.main()
