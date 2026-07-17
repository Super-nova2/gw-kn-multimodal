import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OPTICAL_ONLY_DIR = REPO_ROOT / "optical_only"

class OpticalOnlyLayoutTests(unittest.TestCase):
    def test_supported_entrypoints_use_the_new_script_layout(self):
        required = (
            "scripts/train/train.py",
            "scripts/train/train.sh",
            "scripts/eval/evaluate.py",
            "scripts/data/create_datasets.py",
            "scripts/data/submit_create_datasets.sh",
            "scripts/data/submit_create_negative_dataset.sh",
            "scripts/analysis/snana_first_detection_delay.py",
            "scripts/plot/plot_mag_vs_luptitude.py",
            "scripts/plot/plot_typical_luptitude_lightcurves.py",
        )
        missing = [
            path for path in required if not (OPTICAL_ONLY_DIR / path).is_file()
        ]
        self.assertEqual(missing, [])

    def test_removed_root_entrypoints_are_absent(self):
        removed = (
            "train_optical_only.py",
            "train_optical_only.sh",
            "test_evaluate_optical_only.py",
            "create_optical_only_datasets.py",
            "submit_create_optical_only_datasets.sh",
            "submit_creat_optical_negative_dataset.sh",
        )
        present = [path for path in removed if (OPTICAL_ONLY_DIR / path).exists()]
        self.assertEqual(present, [])

    def test_shell_entrypoints_have_valid_syntax(self):
        scripts = sorted((OPTICAL_ONLY_DIR / "scripts").rglob("*.sh"))
        self.assertGreater(len(scripts), 0)
        for script in scripts:
            with self.subTest(script=script.relative_to(REPO_ROOT)):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_launchers_use_milan_partition(self):
        launchers = (
            "scripts/data/submit_create_datasets.sh",
            "scripts/data/submit_create_negative_dataset.sh",
            "scripts/analysis/submit_snana_first_detection_delay.sh",
        )
        for relative_path in launchers:
            text = (OPTICAL_ONLY_DIR / relative_path).read_text(encoding="utf-8")
            with self.subTest(script=relative_path):
                self.assertIn("#SBATCH --partition=milan", text)
                self.assertNotIn("#SBATCH --partition=cpu", text)

    def test_dataset_self_submit_explicitly_exports_environment(self):
        script = OPTICAL_ONLY_DIR / "scripts" / "data" / "submit_create_datasets.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_sbatch = Path(tmpdir) / "sbatch"
            fake_sbatch.symlink_to("/bin/echo")
            env = os.environ.copy()
            env.update(
                {
                    "BASE_DIR": tmpdir,
                    "DATASET_MODE": "test",
                    "PATH": f"{tmpdir}:{env['PATH']}",
                }
            )
            env.pop("SLURM_JOB_ID", None)
            result = subprocess.run(
                ["bash", str(script)],
                cwd=tmpdir,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--export=ALL", result.stdout)

if __name__ == "__main__":
    unittest.main()
