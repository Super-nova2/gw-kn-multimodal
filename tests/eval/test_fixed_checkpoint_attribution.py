from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval.aggregate_fixed_checkpoint_attribution import (  # noqa: E402
    FACTORIAL_DEFINITIONS,
    METRICS,
    _four_way_factorial_frame,
    _within_query_spearman,
    benjamini_hochberg,
    build_gallery_strategy_comparisons,
    hierarchical_paired_bootstrap,
    prepare_output_dir,
    validate_run_matrix,
)
from scripts.eval.run_fixed_checkpoint_attribution import (  # noqa: E402
    MATCHED_CONDITIONS,
    OPERATIONAL_CONDITIONS,
    RANDOM_MATCHED_CONDITIONS,
    prepare_suite,
)
from scripts.eval.evaluate import _build_gallery_query_cache  # noqa: E402
from scripts.eval.run_fixed_checkpoint_attribution import (  # noqa: E402
    _compact_operational_negative_galleries,
    _encode_streamed_negative_bank,
    _read_gw_tables,
    _read_selected_distance_skymaps,
)


class AttributionPreparationTests(unittest.TestCase):
    def _template(self, root: Path) -> Path:
        template = root / "template.json"
        template.write_text(
            json.dumps(
                {
                    "experiment_id": "unit",
                    "generated_config_root": str(root / "generated"),
                    "output_root": str(root / "outputs"),
                    "models": [
                        {
                            "name": "Full",
                            "type": "multimodal",
                            "checkpoint": "/tmp/checkpoint",
                            "config": "/tmp/model.json",
                        }
                    ],
                    "test_data_path": "/tmp/test.h5",
                    "neg_data_path": "/tmp/negative.h5",
                    "physical_catalogs": {
                        "bns": "/tmp/bns.csv",
                        "nsbh": "/tmp/nsbh.csv",
                    },
                }
            ),
            encoding="utf-8",
        )
        return template

    def test_smoke_generation_covers_both_tasks_and_all_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template = self._template(Path(tmpdir))
            manifest_path, jobs = prepare_suite(
                template, phase="smoke", dry_run=False
            )

            self.assertTrue(manifest_path.is_file())
            self.assertEqual(
                len(jobs),
                len(OPERATIONAL_CONDITIONS)
                + len(MATCHED_CONDITIONS)
                + len(RANDOM_MATCHED_CONDITIONS),
            )
            self.assertEqual({job["seed"] for job in jobs}, {42})
            self.assertEqual(
                {job["task"] for job in jobs},
                {
                    "operational_nonkn",
                    "kn_nuisance_matched",
                    "kn_same_source_random",
                },
            )
            matched = next(
                job
                for job in jobs
                if job["task"] == "kn_nuisance_matched"
                and job["condition"] == "spin2z_perm"
            )
            payload = json.loads(Path(matched["config"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["gallery_sizes"], [16])
            self.assertEqual(payload["gallery_trials"], 2)
            self.assertEqual(payload["max_gw_events"], 64)
            self.assertEqual(
                payload["condition"]["coordinate_mode"], "positive_shared"
            )
            combined = next(
                job
                for job in jobs
                if job["condition"] == "distance_perm__brightness_norm"
            )
            combined_payload = json.loads(
                Path(combined["config"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                combined_payload["condition"]["gw_transform"], "distance_perm"
            )
            self.assertEqual(
                combined_payload["condition"]["optical_transform"],
                "brightness_norm",
            )
            random_job = next(
                job for job in jobs if job["task"] == "kn_same_source_random"
            )
            self.assertEqual(random_job["condition"], "baseline")

    def test_prepare_refuses_to_overwrite_existing_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template = self._template(Path(tmpdir))
            prepare_suite(template, phase="smoke", dry_run=False)
            with self.assertRaises(FileExistsError):
                prepare_suite(template, phase="smoke", dry_run=False)

    def test_remaining_seed_phase_is_exactly_123_and_456(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template = self._template(Path(tmpdir))
            _, jobs = prepare_suite(
                template, phase="remaining_seeds", dry_run=True
            )
            self.assertEqual({job["seed"] for job in jobs}, {123, 456})
            self.assertTrue(all(job["payload"]["gallery_trials"] == 10 for job in jobs))


class AttributionAggregationTests(unittest.TestCase):
    @staticmethod
    def _complete_run_matrix() -> list[dict]:
        runs = []
        for task, conditions in (
            ("operational_nonkn", OPERATIONAL_CONDITIONS),
            ("kn_nuisance_matched", MATCHED_CONDITIONS),
            ("kn_same_source_random", RANDOM_MATCHED_CONDITIONS),
        ):
            for condition in conditions:
                runs.append(
                    {
                        "manifest": {
                            "task": task,
                            "condition": condition["name"],
                            "seed": 42,
                            "code_digest": "code",
                            "test_data_path": "/test.h5",
                        },
                        "result": {
                            "resolved_checkpoint": "/full.pth",
                            "resolved_config": "/full.json",
                        },
                    }
                )
        return runs

    def test_run_matrix_validation_rejects_a_missing_condition(self) -> None:
        runs = self._complete_run_matrix()
        validate_run_matrix(runs)
        with self.assertRaisesRegex(ValueError, "Incomplete condition matrix"):
            validate_run_matrix(runs[:-1])

    def test_aggregation_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "aggregation"
            prepare_output_dir(output_dir, overwrite=False)
            marker = output_dir / "existing.csv"
            marker.write_text("old\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                prepare_output_dir(output_dir, overwrite=False)

            prepare_output_dir(output_dir, overwrite=True)
            self.assertEqual(marker.read_text(encoding="utf-8"), "old\n")

    def test_run_matrix_validation_accepts_legacy_v1_matrix(self) -> None:
        factorial_conditions = {
            definition["combined"] for definition in FACTORIAL_DEFINITIONS
        }
        legacy_runs = [
            run
            for run in self._complete_run_matrix()
            if run["manifest"]["task"] != "kn_same_source_random"
            and run["manifest"]["condition"] not in factorial_conditions
        ]

        self.assertEqual(len(legacy_runs), 26)
        validate_run_matrix(legacy_runs)

    def test_physical_correlation_is_computed_within_query(self) -> None:
        rows = []
        for gw_id in (1, 2):
            for trial in (0, 1):
                for mismatch, score in ((0.1, 0.9), (0.2, 0.5), (0.3, 0.1)):
                    rows.append(
                        {
                            "gw_id": gw_id,
                            "trial": trial,
                            "score": score + gw_id * 10.0,
                            "mismatch": mismatch,
                        }
                    )

        stats = _within_query_spearman(pd.DataFrame(rows), "mismatch")

        self.assertEqual(stats["n_query_trials"], 4)
        self.assertEqual(stats["n_gw"], 2)
        self.assertAlmostEqual(stats["spearman_rho"], -1.0)

    def test_factorial_frame_computes_difference_in_differences(self) -> None:
        definition = FACTORIAL_DEFINITIONS[0]
        values = {
            "baseline": 1.0,
            definition["gw_only"]: 0.8,
            definition["optical_only"]: 0.7,
            definition["combined"]: 0.65,
        }
        rows = []
        for condition, value in values.items():
            for gw_id in (1, 2):
                for trial in (0, 1):
                    rows.append(
                        {
                            "task": "kn_nuisance_matched",
                            "condition": condition,
                            "seed": 42,
                            "trial": trial,
                            "gw_id": gw_id,
                            "source_type": "bns",
                            "gallery_size": 16,
                            "mrr": value,
                        }
                    )

        paired = _four_way_factorial_frame(
            pd.DataFrame(rows),
            definition=definition,
            source="all",
            gallery_size=16,
            metric="mrr",
        )

        np.testing.assert_allclose(paired["delta"], 0.15)

    def test_gallery_strategy_comparison_pairs_the_same_queries(self) -> None:
        rows = []
        for task, value in (
            ("kn_nuisance_matched", 0.2),
            ("kn_same_source_random", 0.3),
        ):
            for gw_id in (1, 2):
                for trial in (0, 1):
                    row = {
                        "task": task,
                        "condition": "baseline",
                        "seed": 42,
                        "trial": trial,
                        "gw_id": gw_id,
                        "source_type": "bns",
                        "gallery_size": 16,
                    }
                    row.update({metric: value for metric in METRICS})
                    rows.append(row)

        comparisons = build_gallery_strategy_comparisons(
            pd.DataFrame(rows),
            n_bootstrap=100,
            bootstrap_seed=7,
            equivalence_margin=0.01,
        )

        recall = next(
            row
            for row in comparisons
            if row["source_type"] == "all"
            and row["metric"] == "recall_at_1"
        )
        self.assertEqual(recall["n_pairs"], 4)
        self.assertAlmostEqual(recall["mean_delta_nearest_minus_random"], -0.1)

    def test_hierarchical_bootstrap_uses_equal_seed_weighting(self) -> None:
        rows = []
        for seed, seed_delta in ((42, 0.2), (123, 0.0)):
            for gw_id in (1, 2, 3):
                for trial in (0, 1):
                    rows.append(
                        {
                            "seed": seed,
                            "gw_id": gw_id,
                            "trial": trial,
                            "delta": seed_delta,
                        }
                    )
        paired = pd.DataFrame(rows)

        result = hierarchical_paired_bootstrap(
            paired, n_bootstrap=200, seed=7
        )

        self.assertAlmostEqual(result["mean_delta"], 0.1)
        self.assertAlmostEqual(result["ci_low"], 0.1)
        self.assertAlmostEqual(result["ci_high"], 0.1)

    def test_benjamini_hochberg_restores_input_order(self) -> None:
        adjusted = benjamini_hochberg([0.04, 0.001, 0.02])
        np.testing.assert_allclose(adjusted, [0.04, 0.003, 0.03])


class QueryTransformHookTests(unittest.TestCase):
    def test_streamed_negative_encoding_discards_raw_tensor_cache(self) -> None:
        class DummyOpticalEncoder:
            @staticmethod
            def encode_components(coords, _times, values, _ref, _masks, opt_err=None):
                del opt_err
                z_curve = values.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
                coord_features = coords
                h_l = values.mean(dim=1)
                return z_curve, coord_features, h_l

            @staticmethod
            def contrastive_head(z_curve, coord_features):
                return torch.cat([z_curve, coord_features], dim=1)

        class DummyModel:
            optical_encoder = DummyOpticalEncoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "negative.h5"
            with h5py.File(path, "w") as handle:
                group = handle.create_group("ELASTICC/optical_data")
                group.create_dataset("times", data=np.zeros((5, 4), np.float32))
                for field, fill in (("values", 1.0), ("masks", 1.0), ("errors", 0.1)):
                    group.create_dataset(
                        field, data=np.full((5, 4, 6), fill, np.float32)
                    )
                group.create_dataset(
                    "coordinates", data=np.arange(10, dtype=np.float32).reshape(5, 2)
                )

            encoded, audit = _encode_streamed_negative_bank(
                model=DummyModel(),
                cfg={
                    "neg_data_path": str(path),
                    "neg_group": "ELASTICC/optical_data",
                    "negative_encode_chunk_size": 2,
                },
                source_indices=np.asarray([0, 2, 4]),
                comparison_window=(-0.1, 0.2),
                transform="none",
                transform_seed=42,
                psfflux_zp=31.4,
                lupt_b_njy=np.ones(6),
                device=torch.device("cpu"),
                runtime_model_args={"n_ref": 4, "ref_start": -0.1, "ref_end": 0.2},
                amp_dtype=torch.float32,
                amp_enabled=False,
            )

        self.assertEqual(encoded["h_l_cls"].shape[0], 3)
        self.assertEqual(encoded["z_curve_cls"].shape[0], 3)
        self.assertNotIn("opt_v_raw", encoded)
        self.assertEqual(audit, [])

    def test_operational_negative_compaction_preserves_source_identity(self) -> None:
        galleries = {
            (3, 0, 7): {
                "positive_index": 0,
                "negative_indices": np.asarray([3, 1]),
            },
            (3, 1, 7): {
                "positive_index": 0,
                "negative_indices": np.asarray([1, 4]),
            },
        }
        sampled_sources = np.asarray([10, 3, 8, 1, 6])

        compacted, selected = _compact_operational_negative_galleries(
            galleries, sampled_sources
        )

        np.testing.assert_array_equal(selected, [1, 3, 6])
        np.testing.assert_array_equal(
            compacted[(3, 0, 7)]["source_negative_indices"], [1, 3]
        )
        np.testing.assert_array_equal(
            compacted[(3, 0, 7)]["negative_indices"], [0, 1]
        )
        np.testing.assert_array_equal(
            compacted[(3, 1, 7)]["negative_indices"], [1, 2]
        )

    def test_runner_reads_only_selected_skymap_distance_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            skymaps = np.arange(4 * 7 * 3, dtype=np.float32).reshape(4, 7, 3)
            with h5py.File(path, "w") as handle:
                gw = handle.create_group("events/gw_data")
                gw.create_dataset("scalars", data=np.zeros((4, 7), np.float32))
                gw.create_dataset("skymaps", data=skymaps)
                gw.create_dataset("source_type", data=np.asarray([b"bns"] * 4))
                gw.create_dataset("ids", data=np.asarray([b"bns_0"] * 4))
                gw.create_dataset("event_time_mjd", data=np.arange(4))
                optical = handle.create_group("events/optical_data")
                optical.create_dataset("parent_gw_idx", data=np.arange(4))

            tables = _read_gw_tables(str(path))
            selected = _read_selected_distance_skymaps(str(path), [1, 3])

        self.assertNotIn("skymaps", tables)
        self.assertEqual(selected.shape, (2, 2, 3))
        np.testing.assert_array_equal(selected, skymaps[[1, 3], 5:7, :])

    def test_query_cache_encodes_transformed_inputs_and_preserves_shapes(self) -> None:
        class DummyModel:
            dual_fusion = False

            def __init__(self) -> None:
                self.seen = None

            def gw_encoder(self, scalars, skymap):
                self.seen = (scalars.detach().clone(), skymap.detach().clone())
                return scalars[:, :2], skymap[:, :2].transpose(1, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            with h5py.File(path, "w") as handle:
                gw = handle.create_group("events/gw_data")
                gw.create_dataset(
                    "scalars", data=np.arange(7, dtype=np.float32).reshape(1, 7)
                )
                gw.create_dataset(
                    "skymaps", data=np.zeros((1, 7, 3), dtype=np.float32)
                )
            model = DummyModel()

            def transform(_gw_id, scalars, skymap):
                scalars[4] = -9.0
                skymap[5] = 3.0
                return scalars, skymap

            cache = _build_gallery_query_cache(
                model,
                [0],
                str(path),
                torch.device("cpu"),
                query_input_transform=transform,
            )

        self.assertAlmostEqual(float(model.seen[0][0, 4]), -9.0)
        np.testing.assert_array_equal(model.seen[1][0, 5].numpy(), [3.0] * 3)
        self.assertAlmostEqual(float(cache[0]["gw_s"][4]), -9.0)


if __name__ == "__main__":
    unittest.main()
