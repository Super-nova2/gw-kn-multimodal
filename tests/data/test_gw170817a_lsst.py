from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "kn_simulation" / "gw170817a"
for path in (REPO_ROOT / "Model", DATASET_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


generate_docs = importlib.import_module("generate_gw170817a_lsst_docs")
build_h5 = importlib.import_module("build_gw170817a_retrieval_h5")


class GW170817ALSSTTests(unittest.TestCase):
    def test_pipeline_uses_one_parent_with_many_fixed_redshift_coordinates(self) -> None:
        pipeline = (DATASET_DIR / "submit_gw170817a_lsst_pipeline.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('SAMPLES_PER_EVENT="${SAMPLES_PER_EVENT:-1}"', pipeline)
        self.assertIn("--coordinate_mode posterior_fixed_distance", pipeline)
        self.assertIn('TARGET_PER_REDSHIFT="${TARGET_PER_REDSHIFT:-200}"', pipeline)
        self.assertNotIn('if [[ "${SAMPLES_PER_EVENT}" != "1" ]]', pipeline)

    def test_real_bayestar_moc_has_training_pixel_count(self) -> None:
        bayestar_path = Path("/fred/oz016/bgao_kn/data/GW_real_events/GW_data/GW170817A/bayestar_no_virgo.fits")
        self.assertTrue(bayestar_path.exists(), msg=f"Missing fixture: {bayestar_path}")

        summary = generate_docs.summarize_moc_skymap(str(bayestar_path))

        self.assertEqual(summary["n_pixels"], 19200)
        self.assertTrue(summary["has_distance"])
        self.assertAlmostEqual(summary["probability_sum"], 1.0, places=6)

    def test_build_manifest_uses_all_redshift_bins_with_fixed_counts(self) -> None:
        manifest = generate_docs.build_manifest_from_probability_pixels(
            ra=np.asarray([10.0, 20.0], dtype=np.float64),
            dec=np.asarray([-1.0, 1.0], dtype=np.float64),
            probability=np.asarray([0.75, 0.25], dtype=np.float64),
            credible_level=np.asarray([0.2, 0.8], dtype=np.float64),
            redshifts=[0.01, 0.02, 0.05],
            n_per_redshift=4,
            seed=123,
        )

        self.assertEqual(len(manifest), 12)
        self.assertEqual(manifest["redshift"].round(4).value_counts().to_dict(), {0.01: 4, 0.02: 4, 0.05: 4})
        self.assertEqual(manifest["sim_event_id"].tolist(), list(range(12)))
        self.assertTrue(set(["ra", "dec", "skymap_credible_level", "skymap_pixel_index"]).issubset(manifest.columns))

    def test_prepared_catalog_has_one_gw_parent_per_redshift(self) -> None:
        pixels = {
            "ra": np.asarray([10.0, 20.0]),
            "dec": np.asarray([-1.0, 1.0]),
            "probability": np.asarray([0.75, 0.25]),
            "credible_level": np.asarray([0.2, 0.8]),
            "pixel_index": np.asarray([4, 9]),
        }
        posterior = {
            "mass1_detector_ref": 1.46,
            "mass2_detector_ref": 1.27,
            "viewing_costheta": 0.7,
        }
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "ligo.skymap.io.fits.read_sky_map", return_value=object()
        ), mock.patch.object(
            generate_docs, "load_probability_pixels_from_moc", return_value=pixels
        ), mock.patch.object(
            generate_docs, "load_posterior_medians", return_value=posterior
        ), mock.patch.object(
            generate_docs, "write_rescaled_skymap"
        ), mock.patch.object(
            generate_docs,
            "luminosity_distance_mpc",
            side_effect=lambda z: 1000.0 * z,
        ):
            rows = generate_docs.build_prepared_catalog(
                skymap_path="/tmp/fake.fits",
                posterior_h5="/tmp/fake.h5",
                redshifts=[0.01, 0.02, 0.16],
                n_per_redshift=[8, 12, 60],
                output_dir=tmpdir,
                skymap_dir=Path(tmpdir) / "skymaps",
            )

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["simulation_id"] for row in rows], [0, 1, 2])
        self.assertEqual(
            [row["optical_candidate_count"] for row in rows], [8, 12, 60]
        )
        self.assertEqual([row["redshift"] for row in rows], [0.01, 0.02, 0.16])

    def test_snana_input_uses_each_simlib_id_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "SIMGEN.INPUT"
            simlib_path = Path(tmpdir) / "test.SIMLIB"
            simlib_path.write_text("NLIBID: 2\n", encoding="utf-8")

            generate_docs.write_snana_input(
                input_path,
                simlib_path=simlib_path,
                genversion="TEST_GW170817A",
                n_lc=2,
            )

            text = input_path.read_text(encoding="utf-8")
            self.assertIn("SIMLIB_NREPEAT: 1", text)
            self.assertLess(text.index("SIMLIB_NREPEAT: 1"), text.index("NGENTOT_LC: 2"))
            self.assertNotIn("NGEN_SEASON", text)

    def test_snana_input_simgen_dump_count_matches_variable_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "SIMGEN.INPUT"
            simlib_path = Path(tmpdir) / "test.SIMLIB"
            simlib_path.write_text("NLIBID: 2\n", encoding="utf-8")

            generate_docs.write_snana_input(
                input_path,
                simlib_path=simlib_path,
                genversion="TEST_GW170817A",
                n_lc=2,
            )

            lines = input_path.read_text(encoding="utf-8").splitlines()
            dump_idx = next(i for i, line in enumerate(lines) if line.startswith("SIMGEN_DUMP:"))
            dump_count = int(lines[dump_idx].split()[1])
            dump_vars = []
            for line in lines[dump_idx + 1 :]:
                if not line.startswith("  "):
                    break
                dump_vars.extend(line.split())

            self.assertEqual(dump_count, 28)
            self.assertEqual(dump_count, len(dump_vars))

    def test_rescale_skymap_distance_channels_preserves_probability_channel(self) -> None:
        skymap = np.zeros((7, 3), dtype=np.float32)
        skymap[4] = np.asarray([50.0, 30.0, 20.0], dtype=np.float32)
        skymap[5] = np.asarray([0.030, 0.040, 0.050], dtype=np.float32)
        skymap[6] = np.asarray([0.005, 0.006, 0.007], dtype=np.float32)

        scaled = generate_docs.rescale_skymap_distance_channels(
            skymap,
            target_distance_mpc=120.0,
            reference_distance_mpc=40.0,
        )

        np.testing.assert_allclose(scaled[4], skymap[4])
        np.testing.assert_allclose(scaled[5], skymap[5] * 3.0)
        np.testing.assert_allclose(scaled[6], skymap[6] * 3.0)

    def test_scaled_gw_inputs_redshift_detector_frame_masses_relative_to_zref(self) -> None:
        base_skymap = np.zeros((7, 3), dtype=np.float32)
        base_skymap[5] = np.asarray([0.030, 0.040, 0.050], dtype=np.float32)
        base_skymap[6] = np.asarray([0.005, 0.006, 0.007], dtype=np.float32)

        scalar, _skymap = build_h5._scaled_gw_inputs(
            base_skymap=base_skymap,
            reference_distance_mpc=40.0,
            reference_distance_std_mpc=10.0,
            redshift=0.16,
            scalar_prefix=np.asarray([1.46, 1.27, 0.0, 0.0, 0.7, 0.0, 0.0], dtype=np.float32),
        )

        mass_scale = (1.0 + 0.16) / (1.0 + 0.01)
        self.assertAlmostEqual(float(scalar[0]), 1.46 * mass_scale, places=6)
        self.assertAlmostEqual(float(scalar[1]), 1.27 * mass_scale, places=6)

    def test_write_retrieval_h5_schema_includes_redshift_metadata(self) -> None:
        skymap = np.zeros((7, 19200), dtype=np.float32)
        skymap[4] = 100.0 / 19200.0
        records = [
            build_h5.FormattedEventRecord(
                sim_event_id=7,
                values=np.ones((200, 6), dtype=np.float32),
                errors=np.full((200, 6), 0.2, dtype=np.float32),
                masks=np.ones((200, 6), dtype=np.float32),
                times=np.linspace(-0.1, 0.2, 200, dtype=np.float32),
                coordinates=np.asarray([197.4, -23.3], dtype=np.float32),
                event_time_mjd=63000.0,
                first_detection_mjd=63001.5,
                redshift=0.05,
                redshift_bin=2,
                credible_level=0.42,
                scalar=np.asarray([1.4, 1.3, 0.0, 0.0, 0.7, 0.22, 0.02], dtype=np.float32),
                skymap=skymap,
                event_uid="gw170817a_7",
                simulation_id=7,
                sample_class="positive",
                mej_dynamic=0.016,
                mej_wind=0.024,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "gw170817a_test.h5"
            build_h5.write_retrieval_h5(
                str(h5_path),
                records,
                generated_per_redshift={0.05: 200},
                scalar_reference=build_h5.GwScalarReference(
                    mass1_detector=1.46,
                    mass2_detector=1.27,
                    costheta=0.7,
                    posterior_h5="/tmp/reference.h5",
                    posterior_dataset="posterior",
                ),
            )

            with h5py.File(h5_path, "r") as f:
                self.assertEqual(f["events/gw_data/scalars"].shape, (1, 7))
                self.assertEqual(f["events/gw_data/skymaps"].shape, (1, 7, 19200))
                self.assertEqual(f["events/gw_data/event_uid"][0], b"gw170817a_7")
                self.assertEqual(f["events/gw_data/simulation_id"][0], 7)
                self.assertEqual(f["events/gw_data/sample_class"][0], b"positive")
                self.assertAlmostEqual(float(f["events/gw_data/mej_dynamic"][0]), 0.016)
                self.assertAlmostEqual(float(f["events/gw_data/mej_wind"][0]), 0.024)
                self.assertEqual(f["events/gw_data/redshift"][0], 0.05)
                self.assertEqual(f["events/gw_data/redshift_bin"][0], 2)
                self.assertEqual(f["events/optical_data/parent_gw_idx"][0], 0)
                self.assertEqual(f.attrs["n_generated_z0.0500"], 200)
                self.assertEqual(f.attrs["n_kept_z0.0500"], 1)
                np.testing.assert_allclose(
                    f.attrs["lupt_m5_mag"],
                    np.asarray([23.9, 25.0, 24.7, 24.0, 23.3, 22.1], dtype=np.float64),
                )
                self.assertEqual(f.attrs["psfflux_zp"], 31.4)
                self.assertEqual(f.attrs["lupt_k"], 1.0)
                self.assertIn("lupt_b_njy", f.attrs)
                self.assertEqual(
                    f.attrs["first_detection_policy"],
                    "psfflux_snr5_then_photflag_then_head_mjd_detect_first",
                )
                self.assertEqual(f.attrs["first_detection_snr_domain"], "merged_psfflux")
                self.assertEqual(
                    f.attrs["scalar_column_names"],
                    "mass1_detector,mass2_detector,spin1z,spin2z,costheta,distmean_gpc,diststd_gpc",
                )
                self.assertEqual(
                    f.attrs["mass_scaling_policy"],
                    "detector_frame_scaled_relative_to_zref_0.01",
                )
                self.assertEqual(float(f.attrs["reference_detector_mass1"]), 1.46)
                self.assertEqual(float(f.attrs["reference_detector_mass2"]), 1.27)
                self.assertEqual(float(f.attrs["reference_redshift"]), 0.01)
                self.assertIn("1 positive KN", f.attrs["gallery_task_definition"])

    def test_write_retrieval_h5_maps_multiple_optical_curves_to_one_gw(self) -> None:
        skymap = np.zeros((7, 19200), dtype=np.float32)

        def make_record(sim_event_id: int, redshift_bin: int, redshift: float):
            return build_h5.FormattedEventRecord(
                sim_event_id=sim_event_id,
                values=np.ones((200, 6), dtype=np.float32),
                errors=np.ones((200, 6), dtype=np.float32),
                masks=np.ones((200, 6), dtype=np.float32),
                times=np.zeros(200, dtype=np.float32),
                coordinates=np.asarray([sim_event_id, redshift_bin], dtype=np.float32),
                event_time_mjd=63000.0,
                first_detection_mjd=63001.0,
                redshift=redshift,
                redshift_bin=redshift_bin,
                credible_level=0.5,
                scalar=np.zeros(7, dtype=np.float32),
                skymap=skymap,
                event_uid=f"gw170817a_{redshift_bin}",
                simulation_id=redshift_bin,
                sample_class="positive",
                mej_dynamic=0.016,
                mej_wind=0.024,
            )

        records = [
            make_record(0, 0, 0.01),
            make_record(0, 0, 0.01),
            make_record(1, 1, 0.02),
            make_record(1, 1, 0.02),
        ]
        reference = build_h5.GwScalarReference(
            mass1_detector=1.46,
            mass2_detector=1.27,
            costheta=0.7,
            posterior_h5="/tmp/reference.h5",
            posterior_dataset="posterior",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "one_to_many.h5"
            build_h5.write_retrieval_h5(path, records, scalar_reference=reference)
            with h5py.File(path, "r") as f:
                self.assertEqual(f["events/gw_data/scalars"].shape[0], 2)
                self.assertEqual(f["events/optical_data/values"].shape[0], 4)
                np.testing.assert_array_equal(
                    f["events/optical_data/parent_gw_idx"][:], [0, 0, 1, 1]
                )
                self.assertEqual(f.attrs["n_total_gw"], 2)
                self.assertEqual(f.attrs["n_total_optical"], 4)

    def test_balanced_selection_requires_and_keeps_exact_target(self) -> None:
        record = build_h5.FormattedEventRecord(
            sim_event_id=0,
            values=np.ones((200, 6), dtype=np.float32),
            errors=np.ones((200, 6), dtype=np.float32),
            masks=np.ones((200, 6), dtype=np.float32),
            times=np.zeros(200, dtype=np.float32),
            coordinates=np.zeros(2, dtype=np.float32),
            event_time_mjd=63000.0,
            first_detection_mjd=63001.0,
            redshift=0.01,
            redshift_bin=0,
            credible_level=0.5,
            scalar=np.zeros(7, dtype=np.float32),
            skymap=np.zeros((7, 19200), dtype=np.float32),
            event_uid="gw170817a_0",
            simulation_id=0,
            sample_class="positive",
            mej_dynamic=0.016,
            mej_wind=0.024,
        )
        selected = build_h5.select_balanced_records(
            [record] * 3,
            target_per_redshift=2,
            expected_redshifts_by_bin={0: 0.01},
        )
        self.assertEqual(len(selected), 2)
        with self.assertRaisesRegex(ValueError, "only 0 usable"):
            build_h5.select_balanced_records(
                [record] * 3,
                target_per_redshift=2,
                expected_redshifts_by_bin={0: 0.01, 1: 0.02},
            )


if __name__ == "__main__":
    unittest.main()
