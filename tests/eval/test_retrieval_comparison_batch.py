import unittest
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
sys.path.insert(0, str(MODEL_DIR))

from scripts.eval.eval_retrieval_comparison import (
    build_batch_positive_bank_and_galleries,
    normalize_shared_config,
)


class RetrievalComparisonBatchPositiveTests(unittest.TestCase):
    def test_normalize_shared_config_preserves_positive_selection(self):
        cfg = normalize_shared_config(
            {
                "test_data_path": "/tmp/test_retrieval_batch.h5",
                "positive_selection": "batch",
            },
            Path("/tmp/retrieval_batch_config.json"),
        )

        self.assertEqual(cfg["positive_selection"], "batch")

    def test_batch_positive_bank_keeps_all_cached_rows(self):
        batch0 = self._batch(
            opt_t=torch.tensor([[0.0], [1.0], [2.0]]),
            opt_v=torch.tensor([[10.0], [11.0], [12.0]]),
            opt_mask=torch.ones(3, 1),
            opt_err=torch.full((3, 1), 0.1),
            opt_coords=torch.tensor([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]]),
            gw_indices=torch.tensor([7, 7, 9]),
        )
        batch1 = self._batch(
            opt_t=torch.tensor([[3.0], [4.0]]),
            opt_v=torch.tensor([[13.0], [14.0]]),
            opt_mask=torch.ones(2, 1),
            opt_err=torch.full((2, 1), 0.1),
            opt_coords=torch.tensor([[0.2, 0.2], [1.1, 1.1]]),
            gw_indices=torch.tensor([7, 9]),
        )
        candidate_sequences = {
            (0, 7): {"candidate_indices": np.arange(20), "credible_levels": np.zeros(20), "abs_dt_days": np.zeros(20)},
            (0, 9): {"candidate_indices": np.arange(20), "credible_levels": np.zeros(20), "abs_dt_days": np.zeros(20)},
        }

        bank, galleries, unique_gw = build_batch_positive_bank_and_galleries(
            [batch0, batch1],
            candidate_sequences,
            gallery_sizes=[10],
            n_trials=1,
            include_undersized=True,
        )

        self.assertEqual(bank["opt_t_raw"].shape[0], 5)
        self.assertEqual(unique_gw, [7, 9])
        self.assertEqual(bank["source_optical_indices"].tolist(), [0, 1, 2, 3, 4])

        for (_gallery_size, _trial, gw_id), spec in galleries.items():
            pos_idx = int(spec["positive_index"])
            self.assertEqual(int(bank["gw_indices"][pos_idx].item()), int(gw_id))

    @staticmethod
    def _batch(opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices):
        size = int(gw_indices.shape[0])
        gw_s = torch.zeros(size, 1)
        gw_m = torch.zeros(size, 1, 1)
        return (gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices)


if __name__ == "__main__":
    unittest.main()
