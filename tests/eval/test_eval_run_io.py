from __future__ import annotations

import csv
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval import eval_run_io  # noqa: E402


class EvaluationRunIoTests(unittest.TestCase):
    def test_atomic_writer_is_invisible_until_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv.gz"
            writer = eval_run_io.AtomicGzipCsvWriter(path, ("a", "b"))
            writer.writerows([{"a": 1, "b": 2}])
            self.assertFalse(path.exists())
            writer.commit()
            self.assertTrue(path.exists())
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                self.assertEqual(
                    list(csv.DictReader(handle)), [{"a": "1", "b": "2"}]
                )

    def test_uncommitted_writer_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv.gz"
            writer = eval_run_io.AtomicGzipCsvWriter(path, ("value",))
            temporary = writer.tmp_path
            writer.writerows([{"value": 1}])
            writer.close()
            self.assertFalse(path.exists())
            self.assertFalse(temporary.exists())

    def test_output_directory_fails_closed_and_validates_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            manifest = eval_run_io.prepare_output_directory(
                output, manifest={"experiment_digest": "same", "seed": 42}
            )
            with self.assertRaises(FileExistsError):
                eval_run_io.prepare_output_directory(
                    output,
                    manifest={"experiment_digest": "same", "seed": 42},
                )
            resumed = eval_run_io.prepare_output_directory(
                output,
                manifest={"experiment_digest": "same", "seed": 42},
                resume=True,
            )
            self.assertEqual(
                resumed["manifest_digest"], manifest["manifest_digest"]
            )
            with self.assertRaises(ValueError):
                eval_run_io.prepare_output_directory(
                    output,
                    manifest={"experiment_digest": "different", "seed": 42},
                    resume=True,
                )
            result = output / "result.json"
            eval_run_io.write_json_atomic(result, {"ok": True})
            eval_run_io.mark_run_success(output, manifest, [result.name])
            with self.assertRaises(FileExistsError):
                eval_run_io.prepare_output_directory(
                    output,
                    manifest={"experiment_digest": "same", "seed": 42},
                    resume=True,
                )

    def test_classification_rows_require_and_preserve_gw_ids(self) -> None:
        logits = {
            "logits_positive": torch.tensor([[0.0, 2.0], [1.0, 0.0]]),
            "gw_id_positive": [7, 9],
            "source_positive": ["bns", "nsbh"],
        }
        rows = list(
            eval_run_io.classification_prediction_rows(
                seed=42, model="Full", triplet_logits=logits
            )
        )
        self.assertEqual([row["gw_id"] for row in rows], [7, 9])
        self.assertEqual([row["label"] for row in rows], [1, 1])
        bad = dict(logits)
        bad["gw_id_positive"] = [7]
        with self.assertRaises(ValueError):
            list(
                eval_run_io.classification_prediction_rows(
                    seed=42, model="Full", triplet_logits=bad
                )
            )


if __name__ == "__main__":
    unittest.main()
