import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
EVAL_DIR = MODEL_DIR / "scripts" / "eval"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from scripts.eval import merge_gw170817a_retrieval as merge  # noqa: E402

def _result(method: str) -> dict:
    config = {key: None for key in merge.GALLERY_CONFIG_KEYS}
    return {
        "config": config,
        "models": {method: {"type": "test"}},
        "curve_rows": [{"method": method}],
        "redshift_rows": [],
        "redshift_macro_rows": [],
        "table": {"rows": [{"method": method}]},
    }

def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")

class MergeGw170817aRetrievalTests(unittest.TestCase):
    def test_assert_same_galleries_rejects_mismatch(self) -> None:
        base = _result("base")
        supplement = _result("new")
        supplement["config"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "seed"):
            merge._assert_same_galleries(base, supplement)

    def test_merge_rejects_duplicate_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_path = tmp_path / "base.json"
            supplement_path = tmp_path / "supplement.json"
            _write(base_path, _result("same"))
            _write(supplement_path, _result("same"))
            with self.assertRaisesRegex(ValueError, "already exists"):
                merge.merge_results(base_path, supplement_path, tmp_path / "out")

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

    def test_merge_comparison_help_runs_outside_repo(self):
        script = MODEL_DIR / "scripts" / "eval" / "merge_retrieval_comparison.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=tmpdir,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == "__main__":
    unittest.main()
