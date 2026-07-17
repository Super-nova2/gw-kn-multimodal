import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
sys.path.insert(0, str(MODEL_DIR))

from scripts.eval import evaluate as evaluate_module  # noqa: E402


class TestLogitMarginExtraction(unittest.TestCase):
    def test_logit_margin_uses_class_one_minus_class_zero(self):
        logits = torch.tensor(
            [
                [1.0, 3.5],
                [2.0, -1.0],
            ],
            dtype=torch.float32,
        )

        margins = evaluate_module._logit_margin_numpy(logits)

        self.assertTrue(np.allclose(margins, np.array([2.5, -3.0], dtype=np.float32)))

    def test_logit_margin_drops_non_finite_values(self):
        logits = torch.tensor(
            [
                [1.0, 2.0],
                [0.0, float("nan")],
                [0.0, float("inf")],
            ],
            dtype=torch.float32,
        )

        margins = evaluate_module._logit_margin_numpy(logits)

        self.assertTrue(np.allclose(margins, np.array([1.0], dtype=np.float32)))

    def test_logit_distribution_plot_does_not_draw_decision_margin_line(self):
        triplet_logits = {
            "logits_positive": torch.tensor([[0.0, 2.0], [0.0, 1.0]], dtype=torch.float32),
            "logits_optical_neg": torch.tensor([[2.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            "logits_gw_neg": torch.tensor([[3.0, 0.0], [2.0, 0.0]], dtype=torch.float32),
            "logits_hard_neg": torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("matplotlib.axes.Axes.axvline", side_effect=AssertionError("decision line drawn")):
                evaluate_module.generate_logits_distribution_plot(triplet_logits, tmpdir)


if __name__ == "__main__":
    unittest.main()
