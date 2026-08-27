from __future__ import annotations

import importlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "kn_simulation" / "gw170817a"
for path in (REPO_ROOT / "Model", DATASET_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

generate_docs = importlib.import_module("generate_gw170817a_lsst_docs")
build_h5 = importlib.import_module("build_gw170817a_retrieval_h5")


def make_record(scenario_id: int, coordinate_id: int, *, true: bool = False):
    scalar = np.asarray(
        [1.48, 1.28, 0.003, 0.001, -0.835, 0.034, 0.009], dtype=np.float32
    )
    return build_h5.FormattedEventRecord(
        sim_event_id=scenario_id,
        scenario_id=scenario_id,
        coordinate_id=coordinate_id,
        libid=coordinate_id + 1,
        values=np.ones((build_h5.MAX_LC_LENGTH, build_h5.NUM_BANDS), dtype=np.float32),
        errors=np.full(
            (build_h5.MAX_LC_LENGTH, build_h5.NUM_BANDS), 0.2, dtype=np.float32
        ),
        masks=np.ones((build_h5.MAX_LC_LENGTH, build_h5.NUM_BANDS), dtype=np.float32),
        times=np.linspace(-0.1, 0.2, build_h5.MAX_LC_LENGTH, dtype=np.float32),
        coordinates=np.asarray([197.45 + coordinate_id, -23.38], dtype=np.float32),
        event_time_mjd=62000.0 + 365.0 * scenario_id,
        sim_explosion_mjd=62000.0 + 365.0 * scenario_id,
        first_detection_mjd=62001.0 + 365.0 * scenario_id,
        scalar=scalar,
        skymap=np.zeros((7, 19200), dtype=np.float32),
        event_uid=f"gw170817a_scenario_{scenario_id:02d}",
        simulation_id=scenario_id,
        sample_class="positive",
        mej_dynamic=0.016,
        mej_wind=0.024,
        is_true_position=true,
        posterior_probability=np.nan if true else 0.01,
        skymap_credible_level=0.1 if true else 0.7,
        too_nobs=4,
        too_mode="Silver",
        n_observations=12,
    )


class GW170817ALSSTTests(unittest.TestCase):
    def test_pipeline_defines_fixed_event_scenario_panel(self) -> None:
        pipeline = (DATASET_DIR / "submit_gw170817a_lsst_pipeline.sh").read_text()
        self.assertIn('N_SCENARIOS="${N_SCENARIOS:-10}"', pipeline)
        self.assertIn('TARGET_PER_SCENARIO="${TARGET_PER_SCENARIO:-50}"', pipeline)
        self.assertIn("posterior_fixed_distance_with_truth", pipeline)
        self.assertIn("gw170817a_lsst_scenarios.h5", pipeline)
        self.assertNotIn("TARGET_PER_REDSHIFT", pipeline)

    def test_annual_scenarios_are_sidereal_shifts_inside_opsim(self) -> None:
        triggers = generate_docs.annual_scenario_trigger_mjds(
            real_trigger_mjd=100.0,
            n_scenarios=3,
            opsim_min_mjd=800.0,
            opsim_max_mjd=2000.0,
        )
        self.assertEqual(len(triggers), 3)
        np.testing.assert_allclose(np.diff(triggers), generate_docs.SIDEREAL_YEAR_DAYS)
        self.assertGreaterEqual(triggers[0], 800.0)
        self.assertLessEqual(triggers[-1] + 4.0, 2000.0)

    def test_opsim_bounds_open_database_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "opsim.db"
            with sqlite3.connect(db) as connection:
                connection.execute(
                    "CREATE TABLE observations (observationStartMJD REAL)"
                )
                connection.executemany(
                    "INSERT INTO observations VALUES (?)", [(10.0,), (20.0,)]
                )
            self.assertEqual(generate_docs.opsim_mjd_bounds(db), (10.0, 20.0))

    def test_catalog_keeps_physics_and_coordinate_seed_fixed(self) -> None:
        posterior = {
            "mass1_detector": 1.48,
            "mass2_detector": 1.28,
            "spin1z": 0.003,
            "spin2z": 0.001,
            "costheta": -0.835,
        }
        sky = {"distmean_mpc": 34.15, "diststd_mpc": 9.34}
        with (
            mock.patch.object(
                generate_docs, "load_posterior_medians", return_value=posterior
            ),
            mock.patch.object(generate_docs, "summarize_moc_skymap", return_value=sky),
            mock.patch.object(
                generate_docs, "opsim_mjd_bounds", return_value=(1000.0, 2500.0)
            ),
            mock.patch.object(
                generate_docs, "simulation_redshift_for_distance", return_value=0.00913
            ),
        ):
            rows = generate_docs.build_scenario_catalog(
                skymap_path="event.fits",
                posterior_h5="posterior.h5",
                opsim_db="opsim.db",
                n_scenarios=3,
                candidate_coordinates=80,
                real_trigger_mjd=100.0,
            )
        self.assertEqual([row["scenario_id"] for row in rows], [0, 1, 2])
        self.assertEqual(
            {row["coordinate_seed"] for row in rows}, {rows[0]["coordinate_seed"]}
        )
        self.assertEqual(len({row["snana_seed"] for row in rows}), 3)
        self.assertEqual({row["luminosity_distance"] for row in rows}, {40.7})
        self.assertEqual({row["mass1_detector"] for row in rows}, {1.48})
        self.assertEqual({row["skymap_distmean_mpc"] for row in rows}, {34.15})

    def test_complete_panel_keeps_true_coordinate_and_shared_alternatives(self) -> None:
        records = [
            make_record(scenario, coordinate, true=coordinate == 0)
            for scenario in range(3)
            for coordinate in range(6)
        ]
        selected = build_h5.select_complete_coordinate_panel(
            records, target_per_scenario=4, expected_scenario_ids=[0, 1, 2], seed=7
        )
        self.assertEqual(len(selected), 12)
        panels = [
            {r.coordinate_id for r in selected if r.scenario_id == scenario}
            for scenario in range(3)
        ]
        self.assertTrue(all(panel == panels[0] for panel in panels))
        self.assertIn(0, panels[0])

    def test_h5_has_scenario_schema_and_unmodified_gw_inputs(self) -> None:
        records = [
            make_record(scenario, coordinate, true=coordinate == 0)
            for scenario in range(2)
            for coordinate in range(2)
        ]
        manifest = pd.DataFrame(
            {
                "scenario_id": [0, 1],
                "trigger_mjd": [62000.0, 62365.0],
                "real_trigger_mjd": [generate_docs.REAL_TRIGGER_MJD] * 2,
                "luminosity_distance": [40.7] * 2,
                "host_redshift_observed": [0.009783] * 2,
                "redshift": [0.00913] * 2,
            }
        )
        reference = build_h5.GwScalarReference(
            1.48,
            1.28,
            0.003,
            0.001,
            -0.835,
            34.15,
            9.34,
            "posterior.h5",
            "posterior",
            "bayestar.fits",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "event.h5"
            build_h5.write_retrieval_h5(
                output,
                records,
                manifest=manifest,
                generated_per_scenario={0: 80, 1: 80},
                scalar_reference=reference,
            )
            with h5py.File(output, "r") as handle:
                self.assertEqual(
                    handle.attrs["dataset_mode"], "gw170817a_lsst_scenarios_v1"
                )
                self.assertEqual(
                    handle.attrs["gw_distance_policy"], "unmodified_bayestar_no_virgo"
                )
                self.assertEqual(handle.attrs["mass_scaling_policy"], "none")
                self.assertEqual(handle["events/gw_data/scalars"].shape, (2, 7))
                self.assertEqual(handle["events/optical_data/scenario_id"].shape, (4,))
                self.assertNotIn("redshift_bin", handle["events/gw_data"])
                np.testing.assert_array_equal(
                    handle["events/optical_data/parent_gw_idx"][:], [0, 0, 1, 1]
                )


if __name__ == "__main__":
    unittest.main()
