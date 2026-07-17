import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


from scripts.train.train import apply_json_config_defaults, load_merged_json_config  # noqa: E402


class ConfigLoadingTests(unittest.TestCase):
    def _write_json(self, directory, name, payload):
        path = Path(directory) / name
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def test_default_config_loads_before_experiment_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = self._write_json(
                tmpdir,
                "default.json",
                {
                    "batch_size": 128,
                    "ref_start": -0.1,
                    "stage_to_jobfs": True,
                    "ignored_null": None,
                },
            )
            experiment_path = self._write_json(
                tmpdir,
                "experiment.json",
                {
                    "batch_size": 1024,
                    "ckpt_path": "/tmp/experiment",
                },
            )

            merged = load_merged_json_config(
                default_json_config=str(default_path),
                json_config=str(experiment_path),
            )

        self.assertEqual(merged["batch_size"], 1024)
        self.assertEqual(merged["ref_start"], -0.1)
        self.assertEqual(merged["ckpt_path"], "/tmp/experiment")
        self.assertNotIn("stage_to_jobfs", merged)
        self.assertNotIn("ignored_null", merged)

    def test_cli_args_override_default_and_experiment_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = self._write_json(tmpdir, "default.json", {"batch_size": 128})
            experiment_path = self._write_json(tmpdir, "experiment.json", {"batch_size": 1024})

            parser = argparse.ArgumentParser()
            parser.add_argument("--batch_size", type=int, default=32)
            apply_json_config_defaults(
                parser,
                default_json_config=str(default_path),
                json_config=str(experiment_path),
            )
            args = parser.parse_args(["--batch_size", "2048"])

        self.assertEqual(args.batch_size, 2048)

    def test_train_shell_has_optional_default_config_argument(self):
        script = (MODEL_DIR / "scripts" / "train" / "train.sh").read_text()

        self.assertIn("MAGIKS_BNS_NSBH_default.json", script)
        self.assertIn("--default_json_config", script)
        self.assertIn("[default.json]", script)


if __name__ == "__main__":
    unittest.main()
