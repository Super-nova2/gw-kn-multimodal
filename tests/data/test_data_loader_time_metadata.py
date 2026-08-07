import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import MixedGWBatchedSampler, RelationalHDF5Dataset  # noqa: E402


def _write_dataset(path, *, include_first_detection):
    with h5py.File(path, "w") as f:
        opt = f.create_group("events/optical_data")
        opt.create_dataset("values", data=np.zeros((1, 2, 6), dtype=np.float32))
        opt.create_dataset("errors", data=np.ones((1, 2, 6), dtype=np.float32))
        opt.create_dataset("masks", data=np.ones((1, 2, 6), dtype=np.float32))
        opt.create_dataset("times", data=np.zeros((1, 2), dtype=np.float32))
        opt.create_dataset("coordinates", data=np.zeros((1, 2), dtype=np.float32))
        opt.create_dataset("parent_gw_idx", data=np.asarray([1], dtype=np.int64))
        if include_first_detection:
            opt.create_dataset("zero_time_mjd_base", data=np.asarray([60001.5], dtype=np.float32))
            opt.create_dataset("first_detection_mjd", data=np.asarray([60011.25], dtype=np.float32))

        gw = f.create_group("events/gw_data")
        gw.create_dataset("scalars", data=np.zeros((2, 7), dtype=np.float32))
        gw.create_dataset("skymaps", data=np.zeros((2, 7, 4), dtype=np.float32))
        gw.create_dataset("event_time_mjd", data=np.asarray([60000.0, 60001.5], dtype=np.float32))
        gw.create_dataset("has_kn", data=np.asarray([0, 1], dtype=np.int8))
        gw.create_dataset("neg_type", data=np.asarray([1, 0], dtype=np.int8))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        gw.create_dataset(
            "source_type",
            data=np.asarray(["bns", "bns"], dtype=object),
            dtype=string_dtype,
        )


class RelationalDatasetTimeMetadataTests(unittest.TestCase):
    def test_requires_optical_first_detection_even_when_gw_event_time_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "train.h5"
            _write_dataset(h5_path, include_first_detection=False)

            with self.assertRaisesRegex(KeyError, "first_detection_mjd"):
                RelationalHDF5Dataset(str(h5_path), return_zero_time_mjd=True)

    def _assert_first_detection_is_returned(self, *, cache_in_memory):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "train.h5"
            _write_dataset(h5_path, include_first_detection=True)

            dataset = RelationalHDF5Dataset(
                str(h5_path),
                cache_in_memory=cache_in_memory,
                return_zero_time_mjd=True,
            )

            item = dataset[0]
            self.assertAlmostEqual(float(item[9]), 60011.25, places=4)

    def test_lazy_dataset_returns_optical_first_detection(self):
        self._assert_first_detection_is_returned(cache_in_memory=False)

    def test_cached_dataset_returns_optical_first_detection(self):
        self._assert_first_detection_is_returned(cache_in_memory=True)

    def test_negative_gw_returns_actual_index_and_reanchors_optical_times(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "train.h5"
            _write_dataset(h5_path, include_first_detection=True)
            dataset = RelationalHDF5Dataset(
                str(h5_path), use_neg_gw=True, return_zero_time_mjd=True
            )

            item = dataset[(0, 0)]
            self.assertEqual(item[7], 0)
            self.assertAlmostEqual(float(item[8]), 60000.0, places=3)
            self.assertAlmostEqual(float(item[9]), 60009.75, places=3)
            self.assertTrue(item[10])


if __name__ == "__main__":
    unittest.main()


class MixedNegativeGWSamplerTests(unittest.TestCase):
    def test_balances_strata_and_pairs_optical_from_same_source(self):
        sampler = MixedGWBatchedSampler(
            gw_to_lc_map={
                10: np.asarray([100]), 11: np.asarray([101]),
                20: np.asarray([200]), 21: np.asarray([201]),
            },
            neg_gw_indices=np.arange(4),
            batch_size=8,
            steps_per_epoch=4,
            neg_gw_ratio=0.5,
            neg_gw_source_types=np.asarray(["bns", "bns", "nsbh", "nsbh"]),
            neg_gw_types=np.asarray([1, 2, 1, 2]),
            gw_source_type_map={10: "bns", 11: "bns", 20: "nsbh", 21: "nsbh"},
        )
        local_source = {0: "bns", 1: "bns", 2: "nsbh", 3: "nsbh"}
        counts = {idx: 0 for idx in range(4)}
        for batch in sampler:
            negatives = [(opt_idx, local_idx) for opt_idx, local_idx in batch if local_idx >= 0]
            self.assertEqual(len(negatives), 4)
            for opt_idx, local_idx in negatives:
                counts[local_idx] += 1
                expected_options = {100, 101} if local_source[local_idx] == "bns" else {200, 201}
                self.assertIn(opt_idx, expected_options)
        self.assertEqual(counts, {0: 4, 1: 4, 2: 4, 3: 4})
