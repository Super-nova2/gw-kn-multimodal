import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.train import train as train_module  # noqa: E402


def _write_minimal_gw_h5(path, *, event_time=None, n_gw=3):
    with h5py.File(path, "w") as f:
        gw = f.create_group("events/gw_data")
        gw.create_dataset("scalars", data=np.zeros((n_gw, 7), dtype=np.float32))
        if event_time is not None:
            gw.create_dataset("event_time_mjd", data=np.asarray(event_time, dtype=np.float32))


class LoadGwEventTimeMjdTableTests(unittest.TestCase):
    def test_reads_matching_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "train.h5"
            _write_minimal_gw_h5(h5_path, event_time=[60000.0, 60001.5, 60003.25])

            table = train_module.load_gw_event_time_mjd_table(str(h5_path), torch.device("cpu"))

            self.assertEqual(table.dtype, torch.float32)
            self.assertEqual(table.device.type, "cpu")
            torch.testing.assert_close(table, torch.tensor([60000.0, 60001.5, 60003.25]))

    def test_returns_none_when_optional_field_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "train.h5"
            _write_minimal_gw_h5(h5_path, event_time=None)

            self.assertIsNone(
                train_module.load_gw_event_time_mjd_table(str(h5_path), torch.device("cpu"))
            )

    def test_returns_none_on_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "train.h5"
            _write_minimal_gw_h5(h5_path, event_time=[60000.0, 60001.5], n_gw=3)

            self.assertIsNone(
                train_module.load_gw_event_time_mjd_table(str(h5_path), torch.device("cpu"))
            )


if __name__ == "__main__":
    unittest.main()
