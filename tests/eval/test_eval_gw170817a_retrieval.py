from __future__ import annotations

import importlib
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


eval_gw170817a = importlib.import_module("eval_gw170817a_retrieval")
evaluate_module = importlib.import_module("scripts.eval.evaluate")


class EvalGW170817ARetrievalTests(unittest.TestCase):
    def test_resolve_mtan_eval_config_parses_string_lupt_m5_mag(self) -> None:
        cfg = evaluate_module.resolve_mtan_eval_config(
            "/tmp/does_not_exist.h5",
            {"mtan_lupt_m5_mag": "23.9,25.0,24.7,24.0,23.3,22.1"},
        )

        self.assertEqual(cfg["mtan_lupt_m5_mag"], (23.9, 25.0, 24.7, 24.0, 23.3, 22.1))

    def test_build_model_specs_accepts_current_skymap_method(self) -> None:
        specs = eval_gw170817a.build_comparison_model_specs(
            {"models": [{"name": "Skymap-only", "type": "skymap"}]},
            Path("/tmp"),
        )

        self.assertEqual([spec["name"] for spec in specs], ["Skymap-only"])
        self.assertEqual(specs[0]["scoring"], "skymap")
        self.assertIsNone(specs[0]["resolved_checkpoint"])

        with self.assertRaises(ValueError):
            eval_gw170817a.build_comparison_model_specs({"models": []}, Path("/tmp"))

    def test_aggregate_redshift_metrics_groups_by_bin_and_gallery_size(self) -> None:
        outcomes = {
            (10, 0, 0): {"rank": 0, "actual_gallery_size": 10, "coverage_met": True},
            (10, 0, 1): {"rank": 4, "actual_gallery_size": 8, "coverage_met": False},
            (10, 0, 2): {"rank": 11, "actual_gallery_size": 10, "coverage_met": True},
            (10, 1, 0): {"rank": 1, "actual_gallery_size": 10, "coverage_met": True},
            (10, 1, 1): {"rank": 0, "actual_gallery_size": 10, "coverage_met": True},
            (10, 1, 2): {"rank": 3, "actual_gallery_size": 9, "coverage_met": False},
        }
        metadata = {
            0: {"redshift": 0.01, "redshift_bin": 0},
            1: {"redshift": 0.01, "redshift_bin": 0},
            2: {"redshift": 0.05, "redshift_bin": 2},
        }

        rows = eval_gw170817a.aggregate_redshift_metrics(
            outcomes=outcomes,
            gallery_sizes=[10],
            n_trials=2,
            unique_gw=[0, 1, 2],
            redshift_metadata=metadata,
            method_name="Skymap-only",
        )

        by_bin = {(row["redshift_bin"], row["gallery_size"]): row for row in rows}
        z0 = by_bin[(0, 10)]
        z2 = by_bin[(2, 10)]

        self.assertEqual(z0["n_queries"], 4)
        self.assertEqual(z0["recall_at_1"], 0.5)
        self.assertEqual(z0["recall_at_5"], 1.0)
        self.assertAlmostEqual(z0["coverage"], 0.95)
        self.assertAlmostEqual(z0["mrr"], np.mean([1.0, 0.2, 0.5, 1.0]))

        self.assertEqual(z2["n_queries"], 2)
        self.assertEqual(z2["recall_at_1"], 0.0)
        self.assertEqual(z2["recall_at_5"], 0.5)
        self.assertAlmostEqual(z2["coverage"], 0.95)


if __name__ == "__main__":
    unittest.main()
