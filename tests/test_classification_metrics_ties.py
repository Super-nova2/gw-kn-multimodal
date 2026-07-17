import sys
import unittest
from pathlib import Path

import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
sys.path.insert(0, str(MODEL_DIR))

from metrics import compute_classification_metrics  # noqa: E402


class ClassificationMetricTieTests(unittest.TestCase):
    def test_all_tied_scores_are_chance_level_for_auroc_and_prevalence_for_auprc(self):
        probs = torch.zeros(4)
        labels = torch.tensor([1, 1, 0, 0])

        metrics = compute_classification_metrics(probs, labels)

        self.assertAlmostEqual(metrics["auroc"], 0.5, places=6)
        self.assertAlmostEqual(metrics["auprc"], 0.5, places=6)

    def test_tied_top_group_uses_grouped_threshold_for_average_precision(self):
        probs = torch.tensor([0.9, 0.9, 0.5, 0.1])
        labels = torch.tensor([1, 0, 1, 0])

        metrics = compute_classification_metrics(probs, labels)

        self.assertAlmostEqual(metrics["auroc"], 0.625, places=6)
        self.assertAlmostEqual(metrics["auprc"], 7.0 / 12.0, places=6)

    def test_non_finite_scores_are_dropped_before_metrics(self):
        probs = torch.tensor([0.9, float("nan"), 0.1, float("inf")])
        labels = torch.tensor([1, 1, 0, 0])

        metrics = compute_classification_metrics(probs, labels)

        self.assertAlmostEqual(metrics["auroc"], 1.0, places=6)
        self.assertAlmostEqual(metrics["auprc"], 1.0, places=6)
        self.assertEqual(metrics["n_dropped_nonfinite"], 2)


if __name__ == "__main__":
    unittest.main()
