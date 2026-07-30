import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval.evaluate import (
    load_negative_optical_samples,
    sample_negative_optical_source_indices,
)


class NegativeSampleLoaderTests(unittest.TestCase):
    @staticmethod
    def _make_test_h5(h5_path, n_rows=50):
        values = np.arange(n_rows * 2 * 3, dtype=np.float32).reshape(n_rows, 2, 3)
        times = np.arange(n_rows * 2, dtype=np.float32).reshape(n_rows, 2)
        masks = np.ones((n_rows, 2, 3), dtype=np.float32)
        errors = values + 0.5
        coordinates = np.stack(
            [np.arange(n_rows, dtype=np.float32), np.arange(n_rows, dtype=np.float32) + 100.0],
            axis=1,
        )
        zero_time = np.arange(n_rows, dtype=np.float64) + 60000.0
        zero_time_cls = zero_time + 10.0
        types_arr = np.asarray([f"T{i:02d}".encode("utf-8") for i in range(n_rows)])
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("ELASTICC/optical_data")
            grp.create_dataset("values", data=values, chunks=(10, 2, 3))
            grp.create_dataset("times", data=times, chunks=(10, 2))
            grp.create_dataset("masks", data=masks, chunks=(10, 2, 3))
            grp.create_dataset("errors", data=errors, chunks=(10, 2, 3))
            grp.create_dataset("coordinates", data=coordinates, chunks=(10, 2))
            grp.create_dataset("zero_time_mjd_base", data=zero_time, chunks=(10,))
            grp.create_dataset("zero_time_mjd_cls_base", data=zero_time_cls, chunks=(10,))
            grp.create_dataset("types", data=types_arr, chunks=(10,))
        return values, times, coordinates, zero_time, zero_time_cls, types_arr

    def test_block_random_reads_exact_sample_count_and_preserves_fields(self):
        with TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "negative.h5"
            values, times, coordinates, zero_time, zero_time_cls, types = self._make_test_h5(
                str(h5_path), n_rows=50
            )

            loaded = load_negative_optical_samples(
                str(h5_path),
                "ELASTICC/optical_data",
                n_samples=25,
                seed=7,
                negative_sample_strategy="block_random",
                negative_sample_block_rows=10,
                negative_sample_shuffle=False,
            )

            source_indices = loaded["source_indices"].numpy()
            self.assertEqual(source_indices.shape[0], 25)
            self.assertEqual(loaded["values"].shape[0], 25)
            self.assertEqual(loaded["times"].shape[0], 25)
            self.assertEqual(loaded["masks"].shape[0], 25)
            self.assertEqual(loaded["errors"].shape[0], 25)
            self.assertEqual(loaded["coordinates"].shape[0], 25)
            np.testing.assert_array_equal(loaded["values"].numpy(), values[source_indices])
            np.testing.assert_array_equal(loaded["times"].numpy(), times[source_indices])
            np.testing.assert_array_equal(loaded["coordinates"].numpy(), coordinates[source_indices])
            np.testing.assert_array_equal(loaded["zero_time_mjd_base"].numpy(), zero_time[source_indices])
            np.testing.assert_array_equal(
                loaded["zero_time_mjd_cls_base"].numpy(), zero_time_cls[source_indices]
            )
            self.assertEqual(loaded["types"], [types[i].decode("utf-8") for i in source_indices])

    def test_block_random_same_seed_returns_same_source_indices(self):
        with TemporaryDirectory() as tmpdir:
            h5_path = str(Path(tmpdir) / "negative.h5")
            self._make_test_h5(h5_path, n_rows=50)

            loaded_a = load_negative_optical_samples(
                h5_path,
                "ELASTICC/optical_data",
                n_samples=25,
                seed=7,
                negative_sample_strategy="block_random",
                negative_sample_block_rows=10,
                negative_sample_shuffle=False,
            )
            loaded_b = load_negative_optical_samples(
                h5_path,
                "ELASTICC/optical_data",
                n_samples=25,
                seed=7,
                negative_sample_strategy="block_random",
                negative_sample_block_rows=10,
                negative_sample_shuffle=False,
            )

            np.testing.assert_array_equal(
                loaded_a["source_indices"].numpy(),
                loaded_b["source_indices"].numpy(),
            )
            np.testing.assert_array_equal(loaded_a["values"].numpy(), loaded_b["values"].numpy())
            np.testing.assert_array_equal(loaded_a["times"].numpy(), loaded_b["times"].numpy())

    def test_block_random_different_seed_returns_different_source_indices(self):
        with TemporaryDirectory() as tmpdir:
            h5_path = str(Path(tmpdir) / "negative.h5")
            self._make_test_h5(h5_path, n_rows=200)

            loaded_a = load_negative_optical_samples(
                h5_path,
                "ELASTICC/optical_data",
                n_samples=30,
                seed=3,
                negative_sample_strategy="block_random",
                negative_sample_block_rows=10,
                negative_sample_shuffle=False,
            )
            loaded_b = load_negative_optical_samples(
                h5_path,
                "ELASTICC/optical_data",
                n_samples=30,
                seed=99,
                negative_sample_strategy="block_random",
                negative_sample_block_rows=10,
                negative_sample_shuffle=False,
            )

            self.assertFalse(
                np.array_equal(
                    loaded_a["source_indices"].numpy(),
                    loaded_b["source_indices"].numpy(),
                )
            )

    def test_block_random_falls_back_to_full_load_when_n_ge_total(self):
        with TemporaryDirectory() as tmpdir:
            h5_path = str(Path(tmpdir) / "negative.h5")
            values, times, coordinates, zero_time, zero_time_cls, types = self._make_test_h5(
                h5_path, n_rows=20
            )

            loaded = load_negative_optical_samples(
                h5_path,
                "ELASTICC/optical_data",
                n_samples=100,
                seed=0,
                negative_sample_strategy="block_random",
                negative_sample_block_rows=10,
                negative_sample_shuffle=False,
            )

            source_indices = loaded["source_indices"].numpy()
            self.assertEqual(source_indices.shape[0], 20)
            np.testing.assert_array_equal(source_indices, np.arange(20, dtype=np.int64))
            np.testing.assert_array_equal(loaded["values"].numpy(), values[:])
            np.testing.assert_array_equal(loaded["times"].numpy(), times[:])
            np.testing.assert_array_equal(loaded["coordinates"].numpy(), coordinates[:])
            np.testing.assert_array_equal(loaded["zero_time_mjd_base"].numpy(), zero_time[:])
            np.testing.assert_array_equal(loaded["zero_time_mjd_cls_base"].numpy(), zero_time_cls[:])
            self.assertEqual(loaded["types"], [types[i].decode("utf-8") for i in range(20)])

    def test_block_random_shuffle_preserves_data_integrity(self):
        with TemporaryDirectory() as tmpdir:
            h5_path = str(Path(tmpdir) / "negative.h5")
            values, times, coordinates, zero_time, zero_time_cls, types = self._make_test_h5(
                h5_path, n_rows=50
            )

            loaded = load_negative_optical_samples(
                h5_path,
                "ELASTICC/optical_data",
                n_samples=25,
                seed=7,
                negative_sample_strategy="block_random",
                negative_sample_block_rows=10,
                negative_sample_shuffle=True,
            )

            source_indices = loaded["source_indices"].numpy()
            self.assertEqual(source_indices.shape[0], 25)
            np.testing.assert_array_equal(loaded["values"].numpy(), values[source_indices])
            np.testing.assert_array_equal(loaded["times"].numpy(), times[source_indices])
            np.testing.assert_array_equal(loaded["coordinates"].numpy(), coordinates[source_indices])
            np.testing.assert_array_equal(loaded["zero_time_mjd_base"].numpy(), zero_time[source_indices])
            np.testing.assert_array_equal(
                loaded["zero_time_mjd_cls_base"].numpy(), zero_time_cls[source_indices]
            )
            self.assertEqual(loaded["types"], [types[i].decode("utf-8") for i in source_indices])

            lightweight = sample_negative_optical_source_indices(
                h5_path,
                "ELASTICC/optical_data",
                n_samples=25,
                seed=7,
                negative_sample_strategy="block_random",
                negative_sample_block_rows=10,
                negative_sample_shuffle=True,
            )
            np.testing.assert_array_equal(lightweight, source_indices)

    def test_strategy_aliases_produce_same_result(self):
        with TemporaryDirectory() as tmpdir:
            h5_path = str(Path(tmpdir) / "negative.h5")
            self._make_test_h5(h5_path, n_rows=50)

            loaded = load_negative_optical_samples(
                h5_path,
                "ELASTICC/optical_data",
                n_samples=20,
                seed=7,
                negative_sample_strategy="chunk",
                negative_sample_block_rows=10,
                negative_sample_shuffle=False,
            )

            self.assertEqual(loaded["source_indices"].shape[0], 20)

    def test_full_read_strategy_still_works(self):
        with TemporaryDirectory() as tmpdir:
            h5_path = str(Path(tmpdir) / "negative.h5")
            values, _times, _coordinates, _zero_time, _zero_time_cls, _types = self._make_test_h5(
                h5_path, n_rows=30
            )

            loaded = load_negative_optical_samples(
                h5_path,
                "ELASTICC/optical_data",
                n_samples=15,
                seed=7,
                negative_sample_strategy="full_read",
            )

            source_indices = loaded["source_indices"].numpy()
            self.assertEqual(source_indices.shape[0], 15)
            self.assertTrue(np.all(np.diff(source_indices) > 0))
            np.testing.assert_array_equal(loaded["values"].numpy(), values[source_indices])


if __name__ == "__main__":
    unittest.main()
