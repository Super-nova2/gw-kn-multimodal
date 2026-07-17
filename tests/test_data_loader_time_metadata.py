import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import RelationalHDF5Dataset  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
