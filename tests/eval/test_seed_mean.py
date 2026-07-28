from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval import eval_run_io, repeat_mean as seed_mean  # noqa: E402


SEED_VALUES = {42: 0.0, 123: 0.3, 456: 0.9}
SEED_GW_COUNTS = {42: 1, 123: 2, 456: 4}


def _make_runs(root: Path) -> None:
    configs = root / "configs"
    configs.mkdir(parents=True)
    for seed in seed_mean.EVAL_SEEDS:
        config_path = configs / f"seed_{seed}.json"
        config = {
            "models": [
                {"name": "Full", "type": "multimodal", "checkpoint": "/ckpt/full"},
                {"name": "Baseline", "type": "multimodal", "checkpoint": "/ckpt/base"},
            ],
            "test_data_path": "/data/test.h5",
            "neg_data_path": "/data/negative.h5",
            "gallery_trials": 10,
            "gallery_sizes": "10",
            "seed": seed,
        }
        eval_run_io.write_json_atomic(config_path, config)
        run = root / f"seed_{seed}"
        manifest = eval_run_io.prepare_output_directory(
            run,
            manifest={
                "experiment_id": "mean-test",
                "experiment_digest": "same-experiment",
                "code_digest": "same-code",
                "seed": seed,
                "input_config": str(config_path),
                "test_data_path": f"/stage/{seed}/test.h5",
                "neg_data_path": f"/stage/{seed}/negative.h5",
            },
        )
        writer = eval_run_io.AtomicGzipCsvWriter(
            run / "retrieval_outcomes.csv.gz", eval_run_io.RETRIEVAL_FIELDS
        )
        full_value = SEED_VALUES[seed]
        baseline_value = max(0.0, full_value - 0.2)
        rows = []
        for trial in range(10):
            for gw_id in range(1, SEED_GW_COUNTS[seed] + 1):
                for model, value in (
                    ("Full", full_value),
                    ("Baseline", baseline_value),
                ):
                    rows.append(
                        {
                            "seed": seed,
                            "trial": trial,
                            "gw_id": gw_id,
                            "source_type": "bns",
                            "redshift_bin": "0.05",
                            "redshift": 0.05,
                            "gallery_size": 10,
                            "actual_gallery_size": 10,
                            "model": model,
                            "rank": 0,
                            "recall_at_1": value,
                            "recall_at_5": value,
                            "recall_at_10": value,
                            "mrr": value,
                            "coverage": 1.0,
                        }
                    )
        writer.writerows(rows)
        writer.commit()
        eval_run_io.mark_run_success(
            run, manifest, ["retrieval_outcomes.csv.gz"]
        )


def _config(root: Path, output: Path) -> dict[str, object]:
    return {
        "eval_seeds": [42, 123, 456],
        "expected_gallery_trials": 10,
        "reference_model": "Full",
        "input_root": str(root),
        "output_dir": str(output),
    }


class SeedMeanTests(unittest.TestCase):
    def test_equal_seed_mean_ignores_different_raw_row_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            output = Path(tmp) / "mean3_summary"
            _make_runs(root)
            result = seed_mean.aggregate_seed_mean(_config(root, output))
            summary = next(
                row
                for row in result["mean_summary"]
                if row["task"] == "retrieval"
                and row["scope"] == "overall"
                and row["model"] == "Full"
                and row["gallery_size"] == 10
                and row["metric"] == "recall_at_1"
            )
            self.assertAlmostEqual(summary["mean"], (0.0 + 0.3 + 0.9) / 3)
            self.assertNotAlmostEqual(
                summary["mean"],
                (0.0 * 1 + 0.3 * 2 + 0.9 * 4) / (1 + 2 + 4),
            )
            delta = next(
                row
                for row in result["mean_deltas"]
                if row["scope"] == "overall"
                and row["comparison_model"] == "Baseline"
                and row["metric"] == "recall_at_1"
            )
            self.assertAlmostEqual(delta["mean_delta"], (0.0 + 0.2 + 0.2) / 3)
            per_seed = pd.read_csv(output / "per_seed_metrics.csv")
            counts = per_seed[
                (per_seed["scope"] == "overall")
                & (per_seed["model"] == "Full")
                & (per_seed["metric"] == "recall_at_1")
            ].set_index("seed")["n_rows"]
            self.assertEqual(counts.to_dict(), {42: 10, 123: 20, 456: 40})
            self.assertTrue((output / "mean_diagnostics.pdf").is_file())
            self.assertTrue((output / "mean_diagnostics.png").is_file())
            self.assertTrue((output / "_SUCCESS.json").is_file())
            success = json.loads(
                (output / "_SUCCESS.json").read_text(encoding="utf-8")
            )
            self.assertIn("mean_diagnostics.png", success["artifacts"])
            self.assertEqual(
                set(result),
                {"eval_seeds", "reference_model", "per_seed_metrics", "mean_summary", "mean_deltas"},
            )
            self.assertTrue(
                all(set(row) == set(seed_mean.PER_SEED_FIELDS) for row in result["per_seed_metrics"])
            )
            self.assertTrue(
                all(set(row) == set(seed_mean.MEAN_FIELDS) for row in result["mean_summary"])
            )
            self.assertTrue(
                all(set(row) == set(seed_mean.DELTA_FIELDS) for row in result["mean_deltas"])
            )

    def test_classification_threshold_is_fixed_at_point_five(self) -> None:
        metrics = seed_mean._classification_metrics(
            np.array([1, 0, 1, 0]), np.array([0.5, 0.49, 0.9, 0.1])
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_rejects_seed_count_missing_success_and_output_conflict(self) -> None:
        with self.assertRaises(ValueError):
            seed_mean._validate_config(
                {"eval_seeds": [42, 123], "expected_gallery_trials": 10}
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            _make_runs(root)
            (root / "seed_123" / "_SUCCESS.json").unlink()
            with self.assertRaises(FileNotFoundError):
                seed_mean.aggregate_seed_mean(
                    _config(root, Path(tmp) / "missing")
                )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            output = Path(tmp) / "exists"
            _make_runs(root)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                seed_mean.aggregate_seed_mean(_config(root, output))

    def test_rejects_manifest_drift_and_unpaired_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            _make_runs(root)
            run = root / "seed_456"
            manifest_path = run / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["code_digest"] = "different-code"
            manifest.pop("manifest_digest")
            manifest["manifest_digest"] = eval_run_io.stable_digest(manifest)
            eval_run_io.write_json_atomic(manifest_path, manifest)
            success_path = run / "_SUCCESS.json"
            success = json.loads(success_path.read_text(encoding="utf-8"))
            success["manifest_digest"] = manifest["manifest_digest"]
            eval_run_io.write_json_atomic(success_path, success)
            with self.assertRaises(ValueError):
                seed_mean.aggregate_seed_mean(
                    _config(root, Path(tmp) / "drift")
                )

        rows = []
        for trial in range(10):
            for model in ("Full", "Baseline"):
                if trial == 9 and model == "Baseline":
                    continue
                rows.append(
                    {
                        "seed": 42,
                        "trial": trial,
                        "gw_id": 1,
                        "gallery_size": 10,
                        "source_type": "bns",
                        "redshift_bin": "0.05",
                        "actual_gallery_size": 10,
                        "model": model,
                        "recall_at_1": 1,
                        "recall_at_5": 1,
                        "recall_at_10": 1,
                        "mrr": 1,
                        "coverage": 1,
                    }
                )
        with self.assertRaises(ValueError):
            seed_mean._validate_retrieval(
                pd.DataFrame(rows), seeds=[42], trials=10
            )

    def test_failure_does_not_write_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            output = Path(tmp) / "interrupted"
            _make_runs(root)
            with mock.patch.object(
                seed_mean,
                "_plot_diagnostics",
                side_effect=RuntimeError("simulated interruption"),
            ):
                with self.assertRaises(RuntimeError):
                    seed_mean.aggregate_seed_mean(_config(root, output))
            self.assertTrue(output.is_dir())
            self.assertFalse((output / "_SUCCESS.json").exists())


if __name__ == "__main__":
    unittest.main()
