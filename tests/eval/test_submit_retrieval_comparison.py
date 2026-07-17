from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "Model" / "scripts" / "eval" / "submit_retrieval_comparison.sh"


class SubmitRetrievalComparisonTests(unittest.TestCase):
    def test_submit_script_reads_new_gallery_config_and_prepares_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_data = tmp_path / "combined_dataset_test.h5"
            neg_data = tmp_path / "tutorial_negative.h5"
            output_dir = tmp_path / "eval_results"
            config_path = tmp_path / "retrieval_comparison.json"
            fake_bin = tmp_path / "bin"
            fake_python = fake_bin / "python"

            test_data.write_bytes(b"test")
            neg_data.write_bytes(b"neg")
            config_path.write_text(
                json.dumps(
                    {
                        "models": [{"name": "skymap-only", "type": "skymap"}],
                        "test_data_path": str(test_data),
                        "neg_data_path": str(neg_data),
                        "neg_group": "Tutorial/optical_data",
                        "output_dir": str(output_dir),
                        "gallery_sizes": "10,100,5000",
                        "gallery_candidate_mode": "time_sky_hard",
                        "gallery_candidate_time_window_days": 50.0,
                        "gallery_candidate_credible_level_max": 0.9,
                        "gallery_include_undersized": True,
                        "device": "cuda",
                        "amp_dtype": "bf16",
                    }
                ),
                encoding="utf-8",
            )

            fake_bin.mkdir(parents=True, exist_ok=True)
            fake_python.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    echo "FAKE_PYTHON $*"
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env.get('PATH', '')}",
                    "SLURM_JOB_ID": "12345",
                    "SLURM_SUBMIT_DIR": str(REPO_ROOT),
                    "SLURMD_NODENAME": "test-node",
                    "SLURM_JOB_PARTITION": "gpu",
                    "SLURM_CPUS_PER_TASK": "4",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "WORKSPACE_ROOT": str(tmp_path),
                }
            )

            result = subprocess.run(
                ["bash", str(SCRIPT_PATH), str(config_path)],
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_dir.exists(), msg=result.stdout)
            self.assertIn("Gallery sizes: 10,100,5000", result.stdout)
            self.assertIn("Gallery candidate mode: time_sky_hard", result.stdout)
            self.assertIn("Candidate time window: +/-50", result.stdout)
            self.assertIn("Candidate credible max: 0.9", result.stdout)
            self.assertIn("Include undersized galleries: true", result.stdout)
            self.assertIn(str(output_dir / "ablation_comparison.json"), result.stdout)
            self.assertIn(str(output_dir / "retrieval_curves.png"), result.stdout)
            self.assertIn(str(output_dir / "retrieval_coverage.png"), result.stdout)
            self.assertIn("FAKE_PYTHON -u", result.stdout)


if __name__ == "__main__":
    unittest.main()
