from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
SCRIPT_DIR = MODEL_DIR / "scripts" / "eval"
for path in (MODEL_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


eval_retrieval_comparison = importlib.import_module("eval_retrieval_comparison")


class EvalRetrievalComparisonTests(unittest.TestCase):
    def test_load_test_positive_index_map_covers_all_gw_and_preserves_optical_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test.h5"
            with h5py.File(h5_path, "w") as f:
                opt = f.create_group("events/optical_data")
                opt.create_dataset("times", data=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32))
                opt.create_dataset("values", data=np.asarray([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32))
                opt.create_dataset("masks", data=np.asarray([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32))
                opt.create_dataset("errors", data=np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32))
                opt.create_dataset(
                    "coordinates",
                    data=np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
                )
                opt.create_dataset("parent_gw_idx", data=np.asarray([7, 9, 7, 12], dtype=np.int64))

            gw_positive_indices, all_test_gw_ids = eval_retrieval_comparison.load_test_positive_index_map(
                str(h5_path)
            )

        self.assertEqual(all_test_gw_ids, [7, 9, 12])
        self.assertEqual(gw_positive_indices[7].tolist(), [0, 2])
        self.assertEqual(gw_positive_indices[9].tolist(), [1])
        self.assertEqual(gw_positive_indices[12].tolist(), [3])

    def test_load_selected_positive_bank_compacts_requested_optical_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test.h5"
            with h5py.File(h5_path, "w") as f:
                opt = f.create_group("events/optical_data")
                opt.create_dataset("times", data=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32))
                opt.create_dataset("values", data=np.asarray([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32))
                opt.create_dataset("masks", data=np.asarray([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32))
                opt.create_dataset("errors", data=np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32))
                opt.create_dataset(
                    "coordinates",
                    data=np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
                )
                opt.create_dataset("parent_gw_idx", data=np.asarray([7, 9, 7, 12], dtype=np.int64))
                opt.create_dataset(
                    "zero_time_mjd_base",
                    data=np.asarray([60000.0, 60001.0, 60002.0, 60003.0], dtype=np.float64),
                )

            positive_bank, remap = eval_retrieval_comparison.load_selected_positive_bank(
                str(h5_path),
                np.asarray([3, 0, 2], dtype=np.int64),
            )

        self.assertEqual(remap, {0: 0, 2: 1, 3: 2})
        self.assertEqual(tuple(positive_bank["opt_t_raw"].shape), (3, 1))
        self.assertEqual(tuple(positive_bank["opt_coords"].shape), (3, 2))
        self.assertEqual(positive_bank["gw_indices"].tolist(), [7, 7, 12])
        self.assertEqual(positive_bank["source_optical_indices"].tolist(), [0, 2, 3])

    def test_temporary_simple_hard_negative_sampling_handles_missing_train_sampler(self) -> None:
        from scripts.train import train as train_module

        attr_name = "sample_inbatch_hard_negatives_with_time"
        missing = object()
        original = getattr(train_module, attr_name, missing)
        if original is not missing:
            delattr(train_module, attr_name)

        class DummyModel:
            pass

        try:
            self.assertFalse(hasattr(train_module, attr_name))
            with eval_retrieval_comparison.temporary_simple_hard_negative_sampling(DummyModel()):
                self.assertIs(
                    getattr(train_module, attr_name),
                    eval_retrieval_comparison.sample_simple_inbatch_hard_negatives_with_time,
                )
            self.assertFalse(hasattr(train_module, attr_name))
        finally:
            if original is not missing:
                setattr(train_module, attr_name, original)


if __name__ == "__main__":
    unittest.main()
