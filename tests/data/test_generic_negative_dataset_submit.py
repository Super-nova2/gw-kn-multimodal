import os
import subprocess
import unittest
from pathlib import Path

from optical_only.scripts.data.create_datasets import infer_transient_type


REPO_ROOT = Path(__file__).resolve().parents[2]
OPTICAL_ONLY_DIR = REPO_ROOT / "optical_only"
SUBMIT_SCRIPT = OPTICAL_ONLY_DIR / "scripts" / "data" / "submit_create_negative_dataset.sh"


class GenericNegativeDatasetSubmitTests(unittest.TestCase):
    def test_infer_transient_type_skips_kn_tokens_only(self):
        self.assertIsNone(infer_transient_type("LSST_KN_BNS"))
        self.assertEqual(infer_transient_type("unknown"), "unknown")

    def test_infer_transient_type_preserves_unknown_non_kn_folder_name(self):
        self.assertEqual(
            infer_transient_type("ELASTICC_TRAIN_CaRT"),
            "ELASTICC_TRAIN_CaRT",
        )

    def test_infer_transient_type_keeps_existing_known_categories(self):
        cases = {
            "ELASTICC_TRAIN_SNIa-SALT2": "SN",
            "ELASTICC_TRAIN_SLSN-I_no_host": "SN",
            "ELASTICC_TRAIN_PISN": "SN",
            "ELASTICC_TRAIN_TDE": "TDE",
            "ELASTICC_TRAIN_AGN": "AGN",
            "ELASTICC_TRAIN_uLens": "uLens",
            "ELASTICC_TRAIN_dwarf-nova": "dwarf-nova",
        }
        for folder_name, expected in cases.items():
            with self.subTest(folder_name=folder_name):
                self.assertEqual(infer_transient_type(folder_name), expected)

    def test_submit_script_syntax_and_help(self):
        syntax = subprocess.run(
            ["bash", "-n", str(SUBMIT_SCRIPT)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        help_result = subprocess.run(
            ["bash", str(SUBMIT_SCRIPT), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("NEG_SIM_ROOT", help_result.stdout)
        self.assertIn("OUTPUT_NEG_DIR", help_result.stdout)
        self.assertIn("OUTPUT_NEG_FILENAME", help_result.stdout)
        self.assertNotIn("NEG_DATASET_TAG", help_result.stdout)

    def test_submit_script_uses_manually_defined_output_path_and_group(self):
        env = os.environ.copy()
        env.update(
            {
                "BASE_DIR": "/tmp/codex_neg_submit_base",
                "SLURM_JOB_ID": "999",
                "SLURM_SUBMIT_DIR": str(REPO_ROOT),
                "NEG_SIM_ROOT": "/tmp",
                "OUTPUT_NEG_DIR": "/tmp/codex_neg_submit_out",
                "OUTPUT_NEG_FILENAME": "manual_negative.h5",
                "NEG_GROUP": "Manual/optical_data",
                "NEG_MATCH_POS_DENSITY": "false",
                "PYTHON_BIN": "true",
            }
        )
        result = subprocess.run(
            ["bash", str(SUBMIT_SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output_neg_h5 /tmp/codex_neg_submit_out/manual_negative.h5", result.stdout)
        self.assertIn("--neg_group Manual/optical_data", result.stdout)
        self.assertIn("--neg_sim_root /tmp", result.stdout)
        self.assertIn("--buffer_limit 3000", result.stdout)
        self.assertIn("--num_workers 4", result.stdout)
        self.assertNotIn("Dataset tag", result.stdout)


if __name__ == "__main__":
    unittest.main()
