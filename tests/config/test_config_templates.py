import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ARGS_DIR = REPO_ROOT / "Model" / "args"
OPTICAL_ARGS_DIR = REPO_ROOT / "optical_only" / "args"

class ConfigTemplateTests(unittest.TestCase):
    def test_all_examples_are_valid_portable_json(self):
        examples = sorted(MODEL_ARGS_DIR.rglob("*.json.example"))
        examples.extend(sorted(OPTICAL_ARGS_DIR.rglob("*.json.example")))
        self.assertGreater(len(examples), 0)
        for path in examples:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertNotIn("/fred/oz016/bgao_kn", text)
                self.assertNotIn("ML+GW+KN", text)
                self.assertIsInstance(json.loads(text), dict)

    def test_current_template_suites_exist(self):
        required = [
            MODEL_ARGS_DIR / "defaults" / "MAGIKS_BNS_NSBH_default.json.example",
            MODEL_ARGS_DIR / "MAGIKS_BNS_NSBH_full.json.example",
            MODEL_ARGS_DIR / "hpo" / "hpo_v5_fusion.json.example",
            MODEL_ARGS_DIR / "eval" / "retrieval_comparison.json.example",
            MODEL_ARGS_DIR / "eval" / "retrieval_gw170817a_lsst.json.example",
            OPTICAL_ARGS_DIR / "optical_only_kn_baseline.json.example",
            OPTICAL_ARGS_DIR / "optical_only_kn_v15_win1020.json.example",
            OPTICAL_ARGS_DIR / "optical_only_kn_v16.json.example",
        ]
        self.assertEqual([path for path in required if not path.is_file()], [])

    def test_current_templates_do_not_reference_removed_entrypoints(self):
        removed_paths = (
            "Model/ALBEF_train.py",
            "Model/ALBEF_train.sh",
            "Model/test_evaluate.py",
            "Model/script/",
            "Model/hpo/",
            "optical_only/train_optical_only.py",
            "optical_only/train_optical_only.sh",
            "optical_only/test_evaluate_optical_only.py",
            "optical_only/create_optical_only_datasets.py",
            "optical_only/submit_creat_optical_negative_dataset.sh",
        )
        current_templates = list(MODEL_ARGS_DIR.rglob("*.json.example"))
        current_templates.extend(OPTICAL_ARGS_DIR.glob("*.json.example"))
        for path in current_templates:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                for removed in removed_paths:
                    self.assertNotIn(removed, text)

    def test_optical_pretrained_templates_target_current_magiks_checkpoint(self):
        expected = (
            "<BASE_DIR>/data/model/checkpoints_bns_nsbh/"
            "bns_nsbh_full/ALBEF/albef_best.pth"
        )
        for name in (
            "optical_only_kn_v15_win1020.json.example",
            "optical_only_kn_v16.json.example",
        ):
            config = json.loads((OPTICAL_ARGS_DIR / name).read_text(encoding="utf-8"))
            with self.subTest(name=name):
                self.assertEqual(config["pretrained_albef_ckpt"], expected)
                self.assertEqual(config["optical_curve_dim"], 192)
                self.assertEqual(config["optical_curve_hidden_dim"], 256)

    def test_unsupported_v17_is_not_a_current_template(self):
        self.assertFalse(
            (OPTICAL_ARGS_DIR / "optical_only_kn_v17_dann.json.example").exists()
        )
        self.assertTrue(
            (
                OPTICAL_ARGS_DIR
                / "old"
                / "experimental"
                / "optical_only_kn_v17_dann.json"
            ).exists()
        )

if __name__ == "__main__":
    unittest.main()
