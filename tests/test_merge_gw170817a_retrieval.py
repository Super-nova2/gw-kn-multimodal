import json
import sys
import tempfile
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval import merge_gw170817a_retrieval as merge  # noqa: E402


def _result(method: str) -> dict:
    config = {key: None for key in merge.GALLERY_CONFIG_KEYS}
    return {
        "config": config,
        "models": {method: {"type": "test"}},
        "curve_rows": [{"method": method}],
        "redshift_rows": [],
        "redshift_macro_rows": [],
        "table": {"rows": [{"method": method}]},
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class MergeGw170817aRetrievalTests(unittest.TestCase):
    def test_assert_same_galleries_rejects_mismatch(self) -> None:
        base = _result("base")
        supplement = _result("new")
        supplement["config"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "seed"):
            merge._assert_same_galleries(base, supplement)

    def test_merge_rejects_duplicate_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_path = tmp_path / "base.json"
            supplement_path = tmp_path / "supplement.json"
            _write(base_path, _result("same"))
            _write(supplement_path, _result("same"))
            with self.assertRaisesRegex(ValueError, "already exists"):
                merge.merge_results(base_path, supplement_path, tmp_path / "out")


if __name__ == "__main__":
    unittest.main()
