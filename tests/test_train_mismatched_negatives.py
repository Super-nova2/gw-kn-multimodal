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


class LoadGwSourceTypeTableTests(unittest.TestCase):
    def test_loads_string_labels_as_integer_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "train.h5"
            with h5py.File(h5_path, "w") as f:
                gw = f.create_group("events/gw_data")
                gw.create_dataset("scalars", data=np.zeros((4, 7), dtype=np.float32))
                string_dtype = h5py.string_dtype(encoding="utf-8")
                gw.create_dataset(
                    "source_type",
                    data=np.asarray(["bns", "nsbh", "bns", "nsbh"], dtype=object),
                    dtype=string_dtype,
                )

            table = train_module.load_gw_source_type_table(
                str(h5_path), torch.device("cpu")
            )

            self.assertEqual(table.dtype, torch.long)
            self.assertEqual(table[0].item(), table[2].item())
            self.assertEqual(table[1].item(), table[3].item())
            self.assertNotEqual(table[0].item(), table[1].item())


class SourceMatchedNegativeSamplingTests(unittest.TestCase):
    def test_prefers_different_gw_with_same_source_type(self):
        gw_indices = torch.tensor([10, 10, 20, 20, 30, 30, 40, 40])
        source_types = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        for _ in range(20):
            selected = train_module.sample_mismatched_negatives(
                batch_size=8,
                device=torch.device("cpu"),
                samples_per_gw=2,
                gw_indices=gw_indices,
                source_types=source_types,
            )

            self.assertTrue(torch.all(gw_indices[selected] != gw_indices))
            self.assertTrue(torch.all(source_types[selected] == source_types))

    def test_falls_back_when_no_other_gw_has_same_source_type(self):
        gw_indices = torch.tensor([10, 10, 20, 20])
        source_types = torch.tensor([0, 0, 1, 1])

        selected = train_module.sample_mismatched_negatives(
            batch_size=4,
            device=torch.device("cpu"),
            samples_per_gw=2,
            gw_indices=gw_indices,
            source_types=source_types,
        )

        self.assertTrue(torch.all(gw_indices[selected] != gw_indices))
        self.assertTrue(torch.all(source_types[selected] != source_types))

    def test_returns_none_when_batch_contains_only_one_gw(self):
        selected = train_module.sample_mismatched_negatives(
            batch_size=4,
            device=torch.device("cpu"),
            samples_per_gw=4,
            gw_indices=torch.tensor([10, 10, 10, 10]),
            source_types=torch.tensor([0, 0, 0, 0]),
        )

        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
