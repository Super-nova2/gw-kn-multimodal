from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
SCRIPT_DIR = MODEL_DIR / "scripts" / "eval"
for path in (MODEL_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

retrieval_gallery = importlib.import_module("retrieval_gallery")
comparison = importlib.import_module("scripts.eval.eval_retrieval_comparison")


class UniformRandomTieMetricTests(unittest.TestCase):
    def test_large_top_tie_has_exact_expected_recall_and_mrr(self) -> None:
        contributions = retrieval_gallery.uniform_random_tie_metric_contributions(0, 99)

        self.assertAlmostEqual(contributions["recall_at_1"], 0.01)
        self.assertAlmostEqual(contributions["recall_at_5"], 0.05)
        self.assertAlmostEqual(contributions["recall_at_10"], 0.10)
        self.assertAlmostEqual(
            contributions["mrr"],
            sum(1.0 / rank for rank in range(1, 101)) / 100.0,
        )

    def test_no_tie_matches_deterministic_rank(self) -> None:
        contributions = retrieval_gallery.uniform_random_tie_metric_contributions(4, 0)

        self.assertEqual(contributions["recall_at_1"], 0.0)
        self.assertEqual(contributions["recall_at_5"], 1.0)
        self.assertEqual(contributions["recall_at_10"], 1.0)
        self.assertAlmostEqual(contributions["mrr"], 0.2)

    def test_aggregate_uses_expected_contributions_for_overall_and_source(self) -> None:
        expected = retrieval_gallery.uniform_random_tie_metric_contributions(0, 9)
        outcomes = {
            (10, 0, 7): {
                "rank": 4.5,
                "actual_gallery_size": 10,
                "metric_contributions": expected,
            }
        }

        metrics, by_source, _coverage = retrieval_gallery.aggregate_gallery_outcomes(
            outcomes=outcomes,
            gallery_sizes=[10],
            n_trials=1,
            unique_gw=[7],
            gw_source_map={7: "bns"},
        )

        self.assertAlmostEqual(metrics["gallery_10_recall_at_1"], 0.1)
        self.assertAlmostEqual(metrics["gallery_10_recall_at_5"], 0.5)
        self.assertAlmostEqual(metrics["gallery_10_mrr"], expected["mrr"])
        self.assertAlmostEqual(by_source["bns"]["gallery_10_recall_at_1"], 0.1)

    def test_common_redshift_aggregator_uses_expected_contributions(self) -> None:
        expected = retrieval_gallery.uniform_random_tie_metric_contributions(2, 7)
        outcomes = {
            (10, 0, 3): {
                "rank": 5.5,
                "actual_gallery_size": 10,
                "metric_contributions": expected,
            }
        }
        metadata = {3: {"redshift": 0.08, "redshift_bin": 1}}

        main_rows = comparison._aggregate_redshift_metrics(
            outcomes=outcomes,
            gallery_sizes=[10],
            n_trials=1,
            unique_gw=[3],
            redshift_metadata=metadata,
            bin_edges=[0.0, 0.1],
            bin_labels=["low"],
            method_name="Fink RF",
        )


        for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"):
            self.assertAlmostEqual(main_rows[0][key], expected[key])


    def test_fink_scorer_records_tie_counts_and_expected_metrics(self) -> None:
        galleries = {
            (5, 0, 3): {
                "positive_index": 0,
                "negative_indices": [10, 11, 12, 13],
                "requested_gallery_size": 5,
                "actual_gallery_size": 5,
                "coverage_met": True,
                "is_undersized": False,
            }
        }
        score_results = [
            ({0: 0.5}, {0: True}, {"n_rows": 1}),
            (
                {10: 0.7, 11: 0.5, 12: 0.5, 13: 0.2},
                {10: True, 11: True, 12: True, 13: True},
                {"n_rows": 4},
            ),
        ]

        with mock.patch.object(
            comparison,
            "score_fink_rf_candidate_bank",
            side_effect=score_results,
        ):
            outcomes, diagnostics = comparison.score_all_galleries_fink_rf(
                artifact={},
                positive_bank={},
                negative_bank={},
                galleries=galleries,
                positive_attrs={},
                negative_attrs={},
            )

        outcome = outcomes[(5, 0, 3)]
        self.assertEqual(outcome["n_strictly_better"], 1)
        self.assertEqual(outcome["n_tied_negatives"], 2)
        self.assertEqual(outcome["tie_block_size"], 3)
        self.assertEqual(outcome["tie_policy"], "uniform_random_expected")
        self.assertAlmostEqual(outcome["metric_contributions"]["recall_at_1"], 0.0)
        self.assertAlmostEqual(outcome["metric_contributions"]["recall_at_5"], 1.0)
        self.assertAlmostEqual(
            outcome["metric_contributions"]["mrr"],
            (1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0) / 3.0,
        )
        self.assertEqual(diagnostics["n_tied_valid_galleries"], 1)
        self.assertEqual(diagnostics["tie_policy"], "uniform_random_expected")


class PartialRefreshMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = [
            {"name": "Full", "type": "multimodal"},
            {"name": "Fink RF", "type": "fink_rf"},
        ]
        self.base = {
            "models": {
                "Full": {"type": "multimodal", "retrieval": {"gallery_10_mrr": 0.9}},
                "Fink RF": {"type": "fink_rf", "retrieval": {"gallery_10_mrr": 0.1}},
            },
            "curve_rows": [
                {"method": "Full", "metric_value": 0.9},
                {"method": "Fink RF", "metric_value": 0.1},
            ],
            "redshift_rows": [
                {"method": "Full", "mrr": 0.9},
                {"method": "Fink RF", "mrr": 0.1},
            ],
        }

    def test_merge_replaces_only_target_model_and_rows(self) -> None:
        merged, retrieval, curves, redshift, names = comparison.merge_partial_model_results(
            base_output=self.base,
            fresh_model_results={
                "Fink RF": {"type": "fink_rf", "retrieval": {"gallery_10_mrr": 0.3}}
            },
            fresh_curve_rows=[{"method": "Fink RF", "metric_value": 0.3}],
            fresh_redshift_rows=[{"method": "Fink RF", "mrr": 0.3}],
            configured_model_specs=self.specs,
            refresh_model_type="fink_rf",
        )

        self.assertEqual(names, ["Full", "Fink RF"])
        self.assertEqual(merged["Full"], self.base["models"]["Full"])
        self.assertEqual(retrieval["Fink RF"]["gallery_10_mrr"], 0.3)
        self.assertEqual([row["metric_value"] for row in curves], [0.9, 0.3])
        self.assertEqual([row["mrr"] for row in redshift], [0.9, 0.3])

    def test_compatibility_rejects_different_gallery_protocol(self) -> None:
        base = {
            "config": {
                "test_data_path": "/old/test.h5",
                "neg_data_path": "/old/neg.h5",
                "neg_group": "ELASTICC/optical_data",
                "seed": 42,
                "gallery_sizes": [10, 100],
            },
            "models": self.base["models"],
        }
        cfg = {
            "test_data_path": "/new/test.h5",
            "neg_data_path": "/new/neg.h5",
            "neg_group": "ELASTICC/optical_data",
            "seed": 42,
            "gallery_sizes": [10, 500],
        }

        with self.assertRaisesRegex(ValueError, "gallery_sizes"):
            comparison.validate_partial_refresh_compatibility(
                base_output=base,
                cfg=cfg,
                configured_model_specs=self.specs,
                refresh_model_type="fink_rf",
            )


if __name__ == "__main__":
    unittest.main()
