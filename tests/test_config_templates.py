import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARGS_DIR = REPO_ROOT / "Model" / "args"


class ConfigTemplateTests(unittest.TestCase):
    def test_all_examples_are_valid_portable_json(self):
        examples = sorted(ARGS_DIR.rglob("*.json.example"))
        self.assertGreater(len(examples), 0)
        for path in examples:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertNotIn("/fred/oz016/bgao_kn", text)
                self.assertNotIn("ML+GW+KN", text)
                self.assertIsInstance(json.loads(text), dict)

    def test_current_magiks_template_suite_exists(self):
        required = [
            ARGS_DIR / "defaults" / "MAGIKS_BNS_NSBH_default.json.example",
            ARGS_DIR / "MAGIKS_BNS_NSBH_full.json.example",
            ARGS_DIR / "hpo" / "hpo_v5_fusion.json.example",
            ARGS_DIR / "eval" / "retrieval_comparison.json.example",
            ARGS_DIR / "eval" / "retrieval_gw170817a_lsst.json.example",
        ]
        self.assertEqual([path for path in required if not path.is_file()], [])

    def test_current_templates_do_not_reference_removed_entrypoints(self):
        removed_paths = (
            "Model/ALBEF_train.py",
            "Model/ALBEF_train.sh",
            "Model/test_evaluate.py",
            "Model/script/",
            "Model/hpo/",
        )
        for path in ARGS_DIR.rglob("*.json.example"):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                for removed in removed_paths:
                    self.assertNotIn(removed, text)


if __name__ == "__main__":
    unittest.main()
