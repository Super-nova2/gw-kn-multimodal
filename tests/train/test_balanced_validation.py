from pathlib import Path
import sys
import unittest

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.train.train import is_checkpoint_score_improved
from data_loader import split_gw_map
from scripts.hpo.hpo_optuna import compute_objective_score
from retrieval_gallery import (
    build_prefixed_gallery_specs,
    compute_source_macro_and_gap,
)
from validation_gallery import (
    compute_hard_gallery_selection_summary,
    partition_validation_gw_ids,
)


class BalancedValidationTests(unittest.TestCase):
    def test_tune_and_confirmation_query_pools_are_disjoint_and_deterministic(self):
        by_source = {"bns": list(range(20)), "nsbh": list(range(100, 120))}
        first = partition_validation_gw_ids(by_source, fraction=0.75, seed=314159)
        second = partition_validation_gw_ids(by_source, fraction=0.75, seed=314159)
        self.assertEqual(first, second)
        for source in by_source:
            self.assertTrue(
                set(first["tune"][source]).isdisjoint(first["confirmation"][source])
            )
            self.assertEqual(len(first["tune"][source]), 15)
            self.assertEqual(len(first["confirmation"][source]), 5)

    def test_selection_score_accepts_explicit_gallery_size_weights(self):
        source_macro = {
            "gallery_100_mrr": 1.0,
            "gallery_5000_mrr": 0.0,
            "gallery_100_recall_at_1": 1.0,
            "gallery_5000_recall_at_1": 0.0,
        }
        summary = compute_hard_gallery_selection_summary(
            source_macro,
            [100, 5000],
            mrr_weight=0.8,
            recall_at_1_weight=0.2,
            gallery_size_weights={100: 0.1, 5000: 0.9},
        )
        self.assertTrue(np.isclose(summary["selection_score"], 0.1))

    def test_stratified_split_preserves_sources_and_has_no_overlap(self):
        gw_map = {gw_id: [gw_id] for gw_id in range(30)}
        source_map = {gw_id: ("bns" if gw_id < 20 else "nsbh") for gw_id in gw_map}

        train_map, val_map = split_gw_map(
            gw_map,
            val_split=0.2,
            seed=42,
            source_type_map=source_map,
        )

        self.assertTrue(set(train_map).isdisjoint(val_map))
        self.assertEqual(set(train_map) | set(val_map), set(gw_map))
        self.assertEqual(sum(source_map[idx] == "bns" for idx in val_map), 4)
        self.assertEqual(sum(source_map[idx] == "nsbh" for idx in val_map), 2)

    def test_source_macro_is_equal_weight_not_sample_weighted(self):
        by_source = {
            "bns": {"gallery_100_mrr": 0.2, "gallery_100_recall_at_1": 0.1},
            "nsbh": {"gallery_100_mrr": 0.8, "gallery_100_recall_at_1": 0.7},
        }

        macro, gap = compute_source_macro_and_gap(by_source)

        self.assertEqual(macro["gallery_100_mrr"], 0.5)
        self.assertTrue(np.isclose(macro["gallery_100_recall_at_1"], 0.4))
        self.assertTrue(np.isclose(gap["gallery_100_mrr"], 0.6))

    def test_selection_score_averages_gallery_sizes_before_weighting(self):
        source_macro = {
            "gallery_100_mrr": 0.9,
            "gallery_500_mrr": 0.6,
            "gallery_1000_mrr": 0.3,
            "gallery_100_recall_at_1": 0.6,
            "gallery_500_recall_at_1": 0.3,
            "gallery_1000_recall_at_1": 0.0,
        }

        summary = compute_hard_gallery_selection_summary(
            source_macro,
            [100, 500, 1000],
            mrr_weight=0.8,
            recall_at_1_weight=0.2,
        )

        self.assertTrue(np.isclose(summary["macro_mrr"], 0.6))
        self.assertTrue(np.isclose(summary["macro_recall_at_1"], 0.3))
        self.assertTrue(np.isclose(summary["selection_score"], 0.54))

    def test_prefixed_galleries_are_deterministic_and_nested(self):
        positive_map = {10: np.asarray([7, 8], dtype=np.int64)}
        candidate_sequences = {
            (0, 10): {
                "candidate_indices": np.arange(20, dtype=np.int64),
                "credible_levels": np.linspace(0.0, 0.9, 20, dtype=np.float32),
                "abs_dt_days": np.arange(20, dtype=np.float32),
            }
        }

        first, _ = build_prefixed_gallery_specs(
            gw_positive_indices=positive_map,
            candidate_sequences=candidate_sequences,
            gallery_sizes=[5, 10],
            n_trials=1,
            seed=42,
            include_undersized=False,
        )
        second, _ = build_prefixed_gallery_specs(
            gw_positive_indices=positive_map,
            candidate_sequences=candidate_sequences,
            gallery_sizes=[5, 10],
            n_trials=1,
            seed=42,
            include_undersized=False,
        )

        small = first[(5, 0, 10)]
        large = first[(10, 0, 10)]
        self.assertEqual(small["positive_index"], large["positive_index"])
        self.assertTrue(
            np.array_equal(small["negative_indices"], large["negative_indices"][:4])
        )
        self.assertTrue(
            np.array_equal(
                first[(10, 0, 10)]["negative_indices"],
                second[(10, 0, 10)]["negative_indices"],
            )
        )

    def test_checkpoint_improvement_keeps_existing_min_delta_semantics(self):
        self.assertTrue(is_checkpoint_score_improved(0.80, None, 0.01))
        self.assertFalse(is_checkpoint_score_improved(0.805, 0.80, 0.01))
        self.assertFalse(is_checkpoint_score_improved(0.81, 0.80, 0.01))
        self.assertTrue(is_checkpoint_score_improved(0.811, 0.80, 0.01))

    def test_hpo_objective_reads_best_checkpoint_hard_gallery_score(self):
        results = {
            "val_hard_gallery_macro_retrieval_score": 0.62,
            "final_epoch": 23,
        }
        score = compute_objective_score(
            results,
            {"val_hard_gallery_macro_retrieval_score": 1.0},
        )
        self.assertEqual(score, 0.62)


if __name__ == "__main__":
    unittest.main()
