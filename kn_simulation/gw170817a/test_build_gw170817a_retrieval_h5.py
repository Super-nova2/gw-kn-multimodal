from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = REPO_ROOT / "kn_simulation" / "gw170817a"
for path in (REPO_ROOT / "Model", MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_gw170817a_retrieval_h5 as builder


def write_posterior(path: Path) -> None:
    dtype = np.dtype(
        [
            ("m1_detector_frame_Msun", "f8"),
            ("m2_detector_frame_Msun", "f8"),
            ("spin1", "f8"),
            ("spin2", "f8"),
            ("costilt1", "f8"),
            ("costilt2", "f8"),
            ("costheta_jn", "f8"),
        ]
    )
    data = np.zeros(4, dtype=dtype)
    data["m1_detector_frame_Msun"] = [1.0, 2.0, np.nan, 4.0]
    data["m2_detector_frame_Msun"] = [1.1, np.nan, 1.5, 1.7]
    data["spin1"] = [0.1, 0.2, np.nan, 0.4]
    data["spin2"] = [0.2, 0.3, 0.4, 0.5]
    data["costilt1"] = [1.0, -1.0, 1.0, 0.5]
    data["costilt2"] = [0.5, 0.5, -0.5, -0.5]
    data["costheta_jn"] = [-0.9, -0.3, np.nan, 0.6]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("posterior", data=data)


class Gw170817ARetrievalH5Test(unittest.TestCase):
    def test_simulation_id_comes_from_event_filename(self):
        self.assertEqual(
            builder.simulation_id_from_head_path(
                "/tmp/LSST_KN_GW170817A_123/LSST_KN_GW170817A_123_HEAD.FITS"
            ),
            123,
        )

    def test_true_coordinate_catalog_requires_exactly_one_truth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "123.csv"
            path.write_text(
                "simulation_id,libid,is_true_position\n123,8,False\n123,42,True\n"
            )
            self.assertEqual(builder.load_true_libid(tmpdir, 123), 42)
            path.write_text(
                "simulation_id,libid,is_true_position\n123,8,True\n123,42,True\n"
            )
            with self.assertRaisesRegex(ValueError, "exactly one true position"):
                builder.load_true_libid(tmpdir, 123)

    def test_posterior_scalar_reference_uses_aligned_spin_medians_and_raw_distance(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            posterior = Path(tmpdir) / "posterior.h5"
            write_posterior(posterior)
            raw_map = np.zeros((7, 19200), dtype=np.float32)
            with mock.patch.object(
                builder, "_load_real_skymap", return_value=(raw_map, 34.15, 9.34)
            ):
                reference = builder.load_posterior_scalar_reference(
                    posterior, "posterior", "bayestar.fits"
                )
        self.assertAlmostEqual(reference.mass1_detector, 2.0)
        self.assertAlmostEqual(reference.mass2_detector, 1.5)
        self.assertAlmostEqual(reference.spin1z, 0.1)
        self.assertAlmostEqual(reference.spin2z, -0.05)
        self.assertAlmostEqual(reference.costheta, -0.3)
        np.testing.assert_allclose(
            reference.scalar_vector(), [2.0, 1.5, 0.1, -0.05, -0.3, 0.03415, 0.00934]
        )

    def test_unbalanced_complete_panel_is_rejected(self):
        records = []
        with self.assertRaisesRegex(ValueError, "true coordinate"):
            builder.select_complete_coordinate_panel(
                records, target_per_scenario=2, expected_scenario_ids=[0, 1]
            )


if __name__ == "__main__":
    unittest.main()
