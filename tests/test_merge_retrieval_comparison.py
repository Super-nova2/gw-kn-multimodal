from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval.merge_retrieval_comparison import _assert_same_galleries  # noqa: E402


def _result_stub():
    return {
        "config": {
            "seed": 42,
            "gallery_sizes": [10, 100],
            "gallery_trials": 10,
            "gallery_candidate_mode": "synthetic_time_sky_hard",
            "gallery_candidate_time_window_days": 30.0,
            "gallery_candidate_credible_level_max": 0.9,
            "gallery_include_undersized": True,
            "positive_selection": "random",
            "n_neg_samples": 500000,
            "negative_sample_strategy": "block_random",
        },
        "gallery_positive_summary": {"n_unique_gw": 1239},
        "selected_positive_summary": {"n_selected_positives": 4718},
    }


class MergeRetrievalComparisonTests(unittest.TestCase):
    def test_gallery_guard_accepts_identical_realizations(self):
        base = _result_stub()
        _assert_same_galleries(base, copy.deepcopy(base))

    def test_gallery_guard_rejects_seed_or_realization_mismatch(self):
        base = _result_stub()
        changed_seed = copy.deepcopy(base)
        changed_seed["config"]["seed"] = 7
        with self.assertRaisesRegex(ValueError, "seed"):
            _assert_same_galleries(base, changed_seed)

        changed_selection = copy.deepcopy(base)
        changed_selection["selected_positive_summary"]["n_selected_positives"] += 1
        with self.assertRaisesRegex(ValueError, "selected_positive_summary"):
            _assert_same_galleries(base, changed_selection)


if __name__ == "__main__":
    unittest.main()
