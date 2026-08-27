from __future__ import annotations

import importlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
SCRIPT_DIR = MODEL_DIR / "scripts" / "eval"
for path in (MODEL_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

retrieval_gallery = importlib.import_module("retrieval_gallery")
common_eval = importlib.import_module("eval_retrieval_comparison")
analysis = importlib.import_module("analyze_gw170817a_retrieval")


class EvalGW170817ARetrievalTests(unittest.TestCase):
    def test_config_is_fixed_event_exhaustive_three_model_task(self) -> None:
        path = MODEL_DIR / "args" / "eval" / "retrieval_gw170817a_lsst.json.example"
        config = json.loads(path.read_text())
        self.assertEqual(config["positive_selection"], "exhaustive")
        self.assertEqual(config["gallery_repeats_per_positive"], 3)
        self.assertEqual(
            [model["name"] for model in config["models"]],
            ["Skymap-only", "Optical-only baseline", "Full"],
        )
        self.assertFalse(config["redshift_analysis_enable"])
        self.assertFalse(config["compute_classification_metrics"])
        self.assertTrue(config["save_outcomes"])

    def test_normalization_accepts_exhaustive_and_custom_result_name(self) -> None:
        config = common_eval.normalize_shared_config(
            {
                "test_data_path": "/tmp/test.h5",
                "positive_selection": "exhaustive",
                "gallery_repeats_per_positive": 3,
                "result_filename": "gw170817a_retrieval.json",
            },
            Path("/tmp/config.json"),
        )
        self.assertEqual(config["positive_selection"], "exhaustive")
        self.assertEqual(config["gallery_repeats_per_positive"], 3)
        self.assertEqual(config["result_filename"], "gw170817a_retrieval.json")

    def test_exhaustive_gallery_queries_every_positive_three_times(self) -> None:
        positive_map = {0: np.asarray([5, 2]), 1: np.asarray([9, 7])}
        candidate_sequences = {
            (trial, gw): {
                "candidate_indices": np.arange(20),
                "credible_levels": np.linspace(0, 1, 20),
                "abs_dt_days": np.arange(20),
            }
            for trial in range(6)
            for gw in (0, 1)
        }
        galleries, unique_gw, n_trials = (
            retrieval_gallery.build_exhaustive_gallery_specs(
                gw_positive_indices=positive_map,
                candidate_sequences=candidate_sequences,
                gallery_sizes=[10],
                repeats_per_positive=3,
                include_undersized=False,
            )
        )
        self.assertEqual(unique_gw, [0, 1])
        self.assertEqual(n_trials, 6)
        self.assertEqual(len(galleries), 12)
        for gw, expected in ((0, [2, 5]), (1, [7, 9])):
            specs = [galleries[(10, trial, gw)] for trial in range(6)]
            self.assertEqual(
                [spec["positive_index"] for spec in specs],
                [expected[0]] * 3 + [expected[1]] * 3,
            )
            self.assertEqual([spec["repeat"] for spec in specs], [0, 1, 2, 0, 1, 2])

    def test_exhaustive_gallery_rejects_unbalanced_panel(self) -> None:
        with self.assertRaisesRegex(ValueError, "balanced crossed panel"):
            retrieval_gallery.build_exhaustive_gallery_specs(
                gw_positive_indices={0: [1, 2], 1: [3]},
                candidate_sequences={},
                gallery_sizes=[10],
                repeats_per_positive=3,
                include_undersized=True,
            )

    def test_two_way_bootstrap_is_reproducible_and_resamples_both_axes(self) -> None:
        values = np.arange(12, dtype=float).reshape(3, 4)
        first, draws = analysis.two_way_bootstrap(values, n_bootstrap=100, seed=17)
        second, _ = analysis.two_way_bootstrap(
            values, n_bootstrap=100, seed=999, draws=draws
        )
        np.testing.assert_allclose(first, second)
        self.assertGreater(float(np.std(first)), 0.0)
        self.assertEqual(draws[0].shape, (100, 3))
        self.assertEqual(draws[1].shape, (100, 4))


if __name__ == "__main__":
    unittest.main()
