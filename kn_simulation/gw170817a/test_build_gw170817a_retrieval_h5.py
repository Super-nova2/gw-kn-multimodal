from pathlib import Path
import importlib.util
import sys
import tempfile
import types
import unittest
from unittest import mock

import h5py
import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


_MISSING = object()
_STUB_MODULE_NAMES = (
    "astropy",
    "astropy.io",
    "astropy.io.fits",
    "ligo",
    "ligo.skymap",
    "ligo.skymap.io",
    "ligo.skymap.io.fits",
)


def _install_skymap_import_stubs() -> dict[str, object]:
    original_modules = {name: sys.modules.get(name, _MISSING) for name in _STUB_MODULE_NAMES}

    fits_module = types.ModuleType("astropy.io.fits")
    astropy_module = types.ModuleType("astropy")
    astropy_io_module = types.ModuleType("astropy.io")
    astropy_io_module.fits = fits_module
    astropy_module.io = astropy_io_module

    ligo_fits_module = types.ModuleType("ligo.skymap.io.fits")
    ligo_fits_module.read_sky_map = lambda *args, **kwargs: None
    ligo_io_module = types.ModuleType("ligo.skymap.io")
    ligo_io_module.fits = ligo_fits_module
    ligo_skymap_module = types.ModuleType("ligo.skymap")
    ligo_skymap_module.io = ligo_io_module
    ligo_module = types.ModuleType("ligo")
    ligo_module.skymap = ligo_skymap_module

    sys.modules["astropy"] = astropy_module
    sys.modules["astropy.io"] = astropy_io_module
    sys.modules["astropy.io.fits"] = fits_module
    sys.modules["ligo"] = ligo_module
    sys.modules["ligo.skymap"] = ligo_skymap_module
    sys.modules["ligo.skymap.io"] = ligo_io_module
    sys.modules["ligo.skymap.io.fits"] = ligo_fits_module
    return original_modules


def _restore_modules(original_modules: dict[str, object]) -> None:
    for name, module in original_modules.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class _FakeDataLoaderSpec:
    class Loader:
        @staticmethod
        def exec_module(module):
            module.MAX_LC_LENGTH = 4
            module.NUM_BANDS = 3
            module.MERGE_WINDOW_HOURS = 2.0
            module.MERGE_MODE = "weighted_average"
            module.MERGE_FLUX_DOMAIN = "psfflux"
            module.parse_lupt_m5_mag = lambda value: np.asarray(
                [float(part) for part in str(value).split(",") if part],
                dtype=np.float64,
            )
            module.build_luptitude_params = lambda **kwargs: (
                1.0,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )

    loader = Loader()


def _import_builder_with_lightweight_data_loader():
    original_spec_from_file_location = importlib.util.spec_from_file_location
    original_module_from_spec = importlib.util.module_from_spec
    original_data_loader = sys.modules.get("data_loader", _MISSING)

    def fake_spec_from_file_location(name, location, *args, **kwargs):
        if name == "data_loader":
            return _FakeDataLoaderSpec()
        return original_spec_from_file_location(name, location, *args, **kwargs)

    def fake_module_from_spec(spec):
        if isinstance(spec, _FakeDataLoaderSpec):
            return types.ModuleType("data_loader")
        return original_module_from_spec(spec)

    importlib.util.spec_from_file_location = fake_spec_from_file_location
    importlib.util.module_from_spec = fake_module_from_spec
    try:
        import build_gw170817a_retrieval_h5 as builder_module
    finally:
        importlib.util.spec_from_file_location = original_spec_from_file_location
        importlib.util.module_from_spec = original_module_from_spec
        if original_data_loader is _MISSING:
            sys.modules.pop("data_loader", None)
        else:
            sys.modules["data_loader"] = original_data_loader
    return builder_module


_original_skymap_modules = _install_skymap_import_stubs()
try:
    builder = _import_builder_with_lightweight_data_loader()
finally:
    _restore_modules(_original_skymap_modules)


def _write_test_posterior(path: Path) -> None:
    dtype = np.dtype(
        [
            ("m1_detector_frame_Msun", np.float64),
            ("m2_detector_frame_Msun", np.float64),
            ("costheta_jn", np.float64),
        ]
    )
    data = np.empty(4, dtype=dtype)
    data["m1_detector_frame_Msun"] = [1.0, 2.0, np.nan, 4.0]
    data["m2_detector_frame_Msun"] = [1.1, np.nan, 1.5, 1.7]
    data["costheta_jn"] = [-0.9, -0.3, np.nan, 0.6]
    with h5py.File(path, "w") as f:
        f.create_dataset("posterior", data=data)


class Gw170817ARetrievalH5Test(unittest.TestCase):
    def test_load_posterior_scalar_reference_uses_finite_signed_medians(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            posterior_path = Path(tmpdir) / "posterior.h5"
            _write_test_posterior(posterior_path)

            reference = builder.load_posterior_scalar_reference(posterior_path, "posterior")

        self.assertAlmostEqual(reference.mass1_detector, 2.0)
        self.assertAlmostEqual(reference.mass2_detector, 1.5)
        self.assertAlmostEqual(reference.costheta, -0.3)
        self.assertAlmostEqual(reference.spin1z, 0.0)
        self.assertAlmostEqual(reference.spin2z, 0.0)
        self.assertEqual(reference.posterior_h5, str(posterior_path))
        self.assertEqual(reference.posterior_dataset, "posterior")
        np.testing.assert_allclose(
            reference.scalar_prefix(),
            np.asarray([2.0, 1.5, 0.0, 0.0, -0.3, 0.0, 0.0], dtype=np.float32),
        )

    def test_scaled_gw_inputs_uses_reference_scalar_and_redshift_distance(self):
        reference = builder.GwScalarReference(
            mass1_detector=2.0,
            mass2_detector=1.5,
            costheta=-0.3,
            posterior_h5="posterior.h5",
            posterior_dataset="posterior",
        )
        distances = {0.01: 100.0, 0.03: 300.0}
        base_skymap = np.ones((7, 3), dtype=np.float32)

        with mock.patch.object(
            builder,
            "luminosity_distance_mpc",
            side_effect=lambda z: distances[round(float(z), 2)],
        ):
            scalar_ref, skymap_ref = builder._scaled_gw_inputs(
                base_skymap=base_skymap,
                reference_distance_mpc=50.0,
                reference_distance_std_mpc=5.0,
                redshift=0.01,
                scalar_prefix=reference.scalar_prefix(),
            )
            scalar_z, _skymap_z = builder._scaled_gw_inputs(
                base_skymap=base_skymap,
                reference_distance_mpc=50.0,
                reference_distance_std_mpc=5.0,
                redshift=0.03,
                scalar_prefix=reference.scalar_prefix(),
            )

        np.testing.assert_allclose(scalar_ref[:5], [2.0, 1.5, 0.0, 0.0, -0.3])
        self.assertAlmostEqual(float(scalar_ref[5]), 0.1)
        self.assertAlmostEqual(float(scalar_ref[6]), 0.01)
        np.testing.assert_allclose(skymap_ref[5:], np.full((2, 3), 2.0, dtype=np.float32))

        mass_scale = 1.03 / 1.01
        self.assertAlmostEqual(float(scalar_z[0]), 2.0 * mass_scale, places=6)
        self.assertAlmostEqual(float(scalar_z[1]), 1.5 * mass_scale, places=6)
        self.assertAlmostEqual(float(scalar_z[2]), 0.0)
        self.assertAlmostEqual(float(scalar_z[3]), 0.0)
        self.assertAlmostEqual(float(scalar_z[4]), -0.3, places=6)
        self.assertAlmostEqual(float(scalar_z[5]), 0.3)
        self.assertAlmostEqual(float(scalar_z[6]), 0.03)

    def test_write_retrieval_h5_records_posterior_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            posterior_path = tmp_path / "posterior.h5"
            _write_test_posterior(posterior_path)
            reference = builder.load_posterior_scalar_reference(posterior_path, "posterior")
            scalar = np.asarray([2.0, 1.5, 0.0, 0.0, -0.3, 0.1, 0.01], dtype=np.float32)
            record = builder.FormattedEventRecord(
                sim_event_id=7,
                values=np.zeros((builder.MAX_LC_LENGTH, builder.NUM_BANDS), dtype=np.float32),
                errors=np.zeros((builder.MAX_LC_LENGTH, builder.NUM_BANDS), dtype=np.float32),
                masks=np.zeros((builder.MAX_LC_LENGTH, builder.NUM_BANDS), dtype=np.float32),
                times=np.zeros((builder.MAX_LC_LENGTH,), dtype=np.float32),
                coordinates=np.asarray([12.0, -35.0], dtype=np.float32),
                event_time_mjd=60000.0,
                first_detection_mjd=60001.0,
                redshift=0.01,
                redshift_bin=0,
                credible_level=0.42,
                scalar=scalar,
                skymap=np.zeros((7, 19200), dtype=np.float32),
            )
            output_path = tmp_path / "retrieval.h5"

            builder.write_retrieval_h5(
                output_path,
                [record],
                generated_per_redshift={0.01: 1},
                scalar_reference=reference,
            )

            with h5py.File(output_path, "r") as f:
                np.testing.assert_allclose(f["events/gw_data/scalars"][:], scalar.reshape(1, 7))
                self.assertEqual(f.attrs["gw_scalar_source"], "posterior_median")
                self.assertEqual(f.attrs["gw_scalar_posterior_h5"], str(posterior_path))
                self.assertEqual(f.attrs["gw_scalar_posterior_dataset"], "posterior")
                self.assertAlmostEqual(float(f.attrs["posterior_mass1_detector_median"]), 2.0)
                self.assertAlmostEqual(float(f.attrs["posterior_mass2_detector_median"]), 1.5)
                self.assertAlmostEqual(float(f.attrs["posterior_costheta_median"]), -0.3)
                self.assertEqual(f.attrs["spin_policy"], "fixed_zero")
                self.assertAlmostEqual(float(f.attrs["reference_detector_mass1"]), 2.0)
                self.assertAlmostEqual(float(f.attrs["reference_detector_mass2"]), 1.5)
                self.assertAlmostEqual(float(f.attrs["reference_redshift"]), 0.01)
                self.assertEqual(int(f.attrs["n_generated_z0.0100"]), 1)
                self.assertEqual(int(f.attrs["n_kept_z0.0100"]), 1)


if __name__ == "__main__":
    unittest.main()
