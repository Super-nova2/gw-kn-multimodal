import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OPTICAL_ONLY_DIR = REPO_ROOT / "optical_only"
SUBMIT_SCRIPT = OPTICAL_ONLY_DIR / "submit_creat_optical_negative_dataset.sh"
OLD_SUBMIT_SCRIPT = OPTICAL_ONLY_DIR / "submit_create_elasticc_test_negative_h5.sh"

if str(OPTICAL_ONLY_DIR) not in sys.path:
    sys.path.insert(0, str(OPTICAL_ONLY_DIR))


class GenericNegativeDatasetSubmitTests(unittest.TestCase):
    def test_infer_transient_type_skips_kn_tokens_only(self):
        from create_optical_only_datasets import infer_transient_type

        self.assertIsNone(infer_transient_type("LSST_KN_BNS"))
        self.assertEqual(infer_transient_type("unknown"), "unknown")

    def test_infer_transient_type_preserves_unknown_non_kn_folder_name(self):
        from create_optical_only_datasets import infer_transient_type

        self.assertEqual(
            infer_transient_type("ELASTICC_TRAIN_CaRT"),
            "ELASTICC_TRAIN_CaRT",
        )

    def test_infer_transient_type_keeps_existing_known_categories(self):
        from create_optical_only_datasets import infer_transient_type

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

    def test_submit_script_has_explicit_core_parameter_interface(self):
        self.assertTrue(SUBMIT_SCRIPT.exists())
        self.assertFalse(OLD_SUBMIT_SCRIPT.exists())
        script = SUBMIT_SCRIPT.read_text()

        self.assertNotIn("INCLUDE_CLASSES", script)
        self.assertNotIn("STAGING_DIR", script)
        self.assertNotIn("slugify", script)
        self.assertNotIn("NEG_DATASET_TAG", script)
        self.assertNotIn("DEFAULT_NEG_SIM_ROOT", script)
        self.assertIn("require_var NEG_SIM_ROOT", script)
        self.assertIn("require_var OUTPUT_NEG_DIR", script)
        self.assertIn("require_var OUTPUT_NEG_FILENAME", script)
        self.assertIn("require_var NEG_GROUP", script)
        self.assertIn('--output_neg_h5 "${OUTPUT_NEG_H5}"', script)
        self.assertIn('--neg_group "${NEG_GROUP}"', script)
        self.assertIn('--neg_sim_root "${NEG_SIM_ROOT}"', script)
        self.assertIn('--buffer_limit "${BUFFER_LIMIT}"', script)
        self.assertIn('--num_workers "${NUM_WORKERS}"', script)

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
