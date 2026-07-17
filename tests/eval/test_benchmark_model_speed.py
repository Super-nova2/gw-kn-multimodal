import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "Model" / "scripts" / "eval" / "benchmark_model_speed.py"
spec = importlib.util.spec_from_file_location("benchmark_model_speed", SCRIPT_PATH)
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


class BenchmarkModelSpeedTests(unittest.TestCase):
    def test_normalize_config_defaults(self):
        cfg = bench.normalize_config(
            {
                "checkpoint": "/tmp/model.pth",
                "data_path": "/tmp/data.h5",
                "output_dir": "/tmp/out",
            }
        )
        self.assertEqual(cfg["batch_sizes"], [1, 8, 32, 128, 256, 512, 1024])
        self.assertEqual(cfg["precision_modes"], ["fp32", "amp"])
        self.assertEqual(cfg["warmup_iterations"], 20)
        self.assertEqual(cfg["measure_iterations"], 100)
        self.assertIsNone(cfg["model_config"])

    def test_normalize_config_parses_strings_and_deduplicates(self):
        cfg = bench.normalize_config(
            {
                "checkpoint": "/tmp/model.pth",
                "data_path": "/tmp/data.h5",
                "output_dir": "/tmp/out",
                "batch_sizes": "8,1,8,32",
                "precision_modes": "amp,fp32,amp",
                "warmup_iterations": 0,
                "measure_iterations": 3,
            }
        )
        self.assertEqual(cfg["batch_sizes"], [1, 8, 32])
        self.assertEqual(cfg["precision_modes"], ["amp", "fp32"])
        self.assertEqual(cfg["warmup_iterations"], 0)
        self.assertEqual(cfg["measure_iterations"], 3)

    def test_normalize_config_rejects_bad_precision(self):
        with self.assertRaisesRegex(ValueError, "Invalid precision mode"):
            bench.normalize_config(
                {
                    "checkpoint": "/tmp/model.pth",
                    "data_path": "/tmp/data.h5",
                    "output_dir": "/tmp/out",
                    "precision_modes": ["int8"],
                }
            )

    def test_summarize_latency_ms(self):
        stats = bench.summarize_latency_ms([1.0, 2.0, 3.0, 4.0], batch_size=8)
        self.assertAlmostEqual(stats["mean_ms"], 2.5)
        self.assertAlmostEqual(stats["median_ms"], 2.5)
        self.assertAlmostEqual(stats["p90_ms"], 3.7)
        self.assertAlmostEqual(stats["p95_ms"], 3.85)
        self.assertAlmostEqual(stats["samples_per_sec_mean"], 3200.0)
        self.assertAlmostEqual(stats["samples_per_sec_median"], 3200.0)

    def test_oom_result_has_csv_fields(self):
        row = bench.oom_result(
            checkpoint="ckpt.pth",
            batch_size=32,
            precision="amp",
            amp_dtype_name="bf16",
            phase="full_multitask",
            warmup_iterations=2,
            measure_iterations=5,
            error="CUDA out of memory",
        )
        self.assertEqual(row["status"], "oom")
        self.assertTrue(set(bench.CSV_FIELDS).issubset(row.keys()))
        self.assertEqual(row["batch_size"], 32)

    def test_json_safe_numpy_values(self):
        payload = bench._json_safe({"x": np.asarray([1, 2]), "y": np.float32(1.5)})
        self.assertEqual(payload["x"], [1, 2])
        self.assertAlmostEqual(payload["y"], 1.5)


if __name__ == "__main__":
    unittest.main()
