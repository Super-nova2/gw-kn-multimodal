from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
SCRIPT_DIR = MODEL_DIR / "scripts" / "eval"
for path in (MODEL_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


retrieval_gallery = importlib.import_module("retrieval_gallery")


def _make_two_pixel_skymap() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.9, 0.1],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )


def _make_two_pixel_skymap_xy(*, first_high: str) -> torch.Tensor:
    if first_high == "x":
        pix_xyz = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    elif first_high == "y":
        pix_xyz = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported first_high={first_high!r}")

    return torch.cat(
        [
            pix_xyz,
            torch.zeros((1, 2), dtype=torch.float32),
            torch.tensor([[0.9, 0.1]], dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float32),
        ],
        dim=0,
    )


class RetrievalGalleryTests(unittest.TestCase):
    def test_current_full_name_uses_magiks_red_and_draws_on_top(self) -> None:
        full_name = "Full (HPO v7 + Mixed Gallery)"
        methods = [full_name, "w/o Retrieval Loss", "Optical-only"]

        self.assertEqual(retrieval_gallery._plot_method_label(full_name), "MAGIKS")
        self.assertEqual(
            retrieval_gallery._plot_method_color_map(methods)[full_name], "#D62728"
        )
        self.assertEqual(
            sorted(methods, key=retrieval_gallery._plot_method_draw_order)[-1],
            full_name,
        )
        self.assertGreater(
            retrieval_gallery._plot_method_zorder(full_name),
            retrieval_gallery._plot_method_zorder("w/o Retrieval Loss"),
        )

    def test_no_classification_color_preserves_existing_method_colors(self) -> None:
        existing_methods = [
            "Fink Random Forest",
            "w/o Contrastive Loss",
            "w/o Cross-Attn",
            "w/o Fusion",
            "w/o Retrieval Loss",
            "Full",
        ]
        extended_methods = [*existing_methods, "w/o Classification Loss"]
        existing_order = sorted(existing_methods, key=retrieval_gallery._plot_method_draw_order)
        extended_order = sorted(extended_methods, key=retrieval_gallery._plot_method_draw_order)

        existing_colors = retrieval_gallery._plot_method_color_map(existing_order)
        extended_colors = retrieval_gallery._plot_method_color_map(extended_order)

        self.assertEqual(
            {method: extended_colors[method] for method in existing_methods},
            existing_colors,
        )
        self.assertEqual(extended_colors["w/o Classification Loss"], "#17BECF")
        self.assertEqual(len(set(extended_colors.values())), len(extended_colors))

    def test_build_time_sky_candidate_sequence_filters_and_sorts(self) -> None:
        seq = retrieval_gallery.build_time_sky_candidate_sequence(
            anchor_time_mjd=50.0,
            candidate_zero_time_mjd=np.asarray([55.0, 45.0, 60.0, 40.0, 49.0], dtype=np.float64),
            candidate_credible_levels=np.asarray([0.4, 0.4, 0.2, 0.95, 0.1], dtype=np.float64),
            time_window_days=10.0,
            credible_level_max=0.9,
            seed=7,
        )

        self.assertEqual(seq["candidate_indices"].tolist(), [0, 2])
        self.assertTrue(np.all(seq["abs_dt_days"] <= 10.0))
        self.assertTrue(np.all(seq["credible_levels"] <= 0.9))

        same_seed = retrieval_gallery.build_time_sky_candidate_sequence(
            anchor_time_mjd=50.0,
            candidate_zero_time_mjd=np.asarray([55.0, 45.0, 60.0, 40.0, 49.0], dtype=np.float64),
            candidate_credible_levels=np.asarray([0.4, 0.4, 0.2, 0.95, 0.1], dtype=np.float64),
            time_window_days=10.0,
            credible_level_max=0.9,
            seed=7,
        )
        self.assertTrue(np.array_equal(seq["candidate_indices"], same_seed["candidate_indices"]))

    def test_build_prefixed_gallery_specs_keeps_undersized_queries(self) -> None:
        candidate_sequences = {
            (0, 0): {
                "candidate_indices": np.asarray([10, 11], dtype=np.int64),
                "credible_levels": np.asarray([0.1, 0.2], dtype=np.float32),
                "abs_dt_days": np.asarray([1.0, 2.0], dtype=np.float32),
            },
            (0, 1): {
                "candidate_indices": np.asarray([20, 21, 22, 23], dtype=np.int64),
                "credible_levels": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                "abs_dt_days": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            },
        }
        galleries, unique_gw = retrieval_gallery.build_prefixed_gallery_specs(
            gw_positive_indices={
                0: np.asarray([100], dtype=np.int64),
                1: np.asarray([200, 201], dtype=np.int64),
            },
            candidate_sequences=candidate_sequences,
            gallery_sizes=[3, 5],
            n_trials=1,
            seed=5,
            include_undersized=True,
        )

        self.assertEqual(unique_gw, [0, 1])

        undersized = galleries[(5, 0, 0)]
        self.assertEqual(undersized["positive_index"], 100)
        self.assertEqual(undersized["negative_indices"].tolist(), [10, 11])
        self.assertEqual(undersized["actual_gallery_size"], 3)
        self.assertEqual(undersized["requested_gallery_size"], 5)
        self.assertTrue(undersized["is_undersized"])

        full = galleries[(5, 0, 1)]
        self.assertIn(full["positive_index"], {200, 201})
        self.assertEqual(full["negative_indices"].tolist(), [20, 21, 22, 23])
        self.assertEqual(full["actual_gallery_size"], 5)
        self.assertFalse(full["is_undersized"])

    def test_skymap_model_specs_allow_missing_checkpoint(self) -> None:
        model_specs = retrieval_gallery.build_comparison_model_specs(
            {
                "models": [
                    {
                        "name": "skymap-only",
                        "type": "skymap",
                    }
                ]
            },
            REPO_ROOT / "Model" / "args" / "eval",
        )

        self.assertEqual(model_specs[0]["type"], "skymap")
        self.assertIsNone(model_specs[0]["checkpoint"])
        self.assertIsNone(model_specs[0]["resolved_checkpoint"])
        self.assertIsNone(model_specs[0]["resolved_config"])

    def test_score_all_galleries_skymap_and_aggregate_metrics(self) -> None:
        candidate_sequences = {
            (0, 0): {
                "candidate_indices": np.asarray([0], dtype=np.int64),
                "credible_levels": np.asarray([1.0], dtype=np.float32),
                "abs_dt_days": np.asarray([5.0], dtype=np.float32),
            }
        }
        galleries, unique_gw = retrieval_gallery.build_prefixed_gallery_specs(
            gw_positive_indices={0: np.asarray([0], dtype=np.int64)},
            candidate_sequences=candidate_sequences,
            gallery_sizes=[2],
            n_trials=1,
            seed=13,
            include_undersized=True,
        )
        outcomes = retrieval_gallery.score_all_galleries_skymap(
            positive_bank={"opt_coords": torch.tensor([[0.0, 0.0]], dtype=torch.float32)},
            galleries=galleries,
            gw_skymaps={0: _make_two_pixel_skymap()},
        )
        metrics, by_source, coverage = retrieval_gallery.aggregate_gallery_outcomes(
            outcomes=outcomes,
            gallery_sizes=[2],
            n_trials=1,
            unique_gw=unique_gw,
            gw_source_map={0: "bns"},
        )

        self.assertEqual(outcomes[(2, 0, 0)]["rank"], 0)
        self.assertEqual(metrics["gallery_2_recall_at_1"], 1.0)
        self.assertEqual(metrics["gallery_2_mrr"], 1.0)
        self.assertEqual(by_source["bns"]["gallery_2_recall_at_1"], 1.0)
        self.assertEqual(coverage["gallery_2"]["coverage"], 1.0)
        self.assertEqual(coverage["gallery_2"]["effective_gallery_size_min"], 2)

    def test_aggregate_gallery_outcomes_uses_zero_based_rank_for_recall_and_mrr(self) -> None:
        outcomes = {
            (5, 0, 10): {"rank": 0, "actual_gallery_size": 5},
            (5, 0, 11): {"rank": 4, "actual_gallery_size": 5},
            (5, 0, 12): {"rank": 5, "actual_gallery_size": 6},
        }

        metrics, _by_source, coverage = retrieval_gallery.aggregate_gallery_outcomes(
            outcomes=outcomes,
            gallery_sizes=[5],
            n_trials=1,
            unique_gw=[10, 11, 12],
        )

        self.assertAlmostEqual(metrics["gallery_5_recall_at_1"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["gallery_5_recall_at_5"], 2.0 / 3.0)
        self.assertAlmostEqual(
            metrics["gallery_5_mrr"],
            (1.0 + 1.0 / 5.0 + 1.0 / 6.0) / 3.0,
        )
        self.assertAlmostEqual(coverage["gallery_5"]["fill_ratio_mean"], 1.0)

    def test_build_prefixed_gallery_specs_covers_all_mapped_gw_per_trial(self) -> None:
        candidate_sequences = {
            (0, 0): {"candidate_indices": np.asarray([10], dtype=np.int64), "credible_levels": np.asarray([0.1], dtype=np.float32), "abs_dt_days": np.asarray([1.0], dtype=np.float32)},
            (0, 1): {"candidate_indices": np.asarray([20], dtype=np.int64), "credible_levels": np.asarray([0.2], dtype=np.float32), "abs_dt_days": np.asarray([2.0], dtype=np.float32)},
            (0, 2): {"candidate_indices": np.asarray([30], dtype=np.int64), "credible_levels": np.asarray([0.3], dtype=np.float32), "abs_dt_days": np.asarray([3.0], dtype=np.float32)},
            (1, 0): {"candidate_indices": np.asarray([11], dtype=np.int64), "credible_levels": np.asarray([0.1], dtype=np.float32), "abs_dt_days": np.asarray([1.0], dtype=np.float32)},
            (1, 1): {"candidate_indices": np.asarray([21], dtype=np.int64), "credible_levels": np.asarray([0.2], dtype=np.float32), "abs_dt_days": np.asarray([2.0], dtype=np.float32)},
            (1, 2): {"candidate_indices": np.asarray([31], dtype=np.int64), "credible_levels": np.asarray([0.3], dtype=np.float32), "abs_dt_days": np.asarray([3.0], dtype=np.float32)},
        }

        galleries, unique_gw = retrieval_gallery.build_prefixed_gallery_specs(
            gw_positive_indices={
                0: np.asarray([100, 101], dtype=np.int64),
                1: np.asarray([200], dtype=np.int64),
                2: np.asarray([300, 301, 302], dtype=np.int64),
            },
            candidate_sequences=candidate_sequences,
            gallery_sizes=[2, 4],
            n_trials=2,
            seed=17,
            include_undersized=True,
        )

        self.assertEqual(unique_gw, [0, 1, 2])
        self.assertEqual(len(galleries), 3 * 2 * 2)
        for trial in (0, 1):
            for gw_id, allowed_pos in {0: {100, 101}, 1: {200}, 2: {300, 301, 302}}.items():
                small = galleries[(2, trial, gw_id)]
                large = galleries[(4, trial, gw_id)]
                self.assertIn(small["positive_index"], allowed_pos)
                self.assertEqual(small["positive_index"], large["positive_index"])

    def test_build_time_sky_candidate_sequences_handles_inconsistent_geometries(self) -> None:
        gw_skymaps = torch.stack(
            [
                _make_two_pixel_skymap_xy(first_high="x"),
                _make_two_pixel_skymap_xy(first_high="y"),
            ],
            dim=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("events/gw_data/event_time_mjd", data=np.asarray([100.0, 100.0], dtype=np.float32))
                f.create_dataset("events/gw_data/skymaps", data=gw_skymaps.numpy())

            candidate_sequences, gw_lookup, time_lookup = retrieval_gallery.build_time_sky_candidate_sequences(
                test_data_path=str(h5_path),
                unique_gw_ids=[0, 1],
                neg_optical_data={
                    "zero_time_mjd_cls_base": np.asarray([100.0, 100.0], dtype=np.float32),
                    "coordinates": torch.tensor([[0.0, 0.0], [90.0, 0.0]], dtype=torch.float32),
                },
                n_trials=1,
                seed=11,
                time_window_days=1.0,
                credible_level_max=1.0,
            )

        self.assertEqual(sorted(gw_lookup.keys()), [0, 1])
        self.assertEqual(time_lookup, {0: 100.0, 1: 100.0})
        self.assertEqual(candidate_sequences[(0, 0)]["candidate_indices"].tolist(), [0, 1])
        self.assertEqual(candidate_sequences[(0, 1)]["candidate_indices"].tolist(), [0, 1])

    def test_plot_helpers_write_curve_and_coverage_artifacts(self) -> None:
        curve_rows = [
            {
                "method": "skymap-only",
                "gallery_size_target": 10,
                "gallery_size_actual": 10.0,
                "coverage": 1.0,
                "metric_name": "R@1",
                "metric_value": 0.4,
            },
            {
                "method": "skymap-only",
                "gallery_size_target": 100,
                "gallery_size_actual": 80.0,
                "coverage": 0.7,
                "metric_name": "R@1",
                "metric_value": 0.2,
            },
            {
                "method": "full multimodal",
                "gallery_size_target": 10,
                "gallery_size_actual": 10.0,
                "coverage": 1.0,
                "metric_name": "R@1",
                "metric_value": 0.8,
            },
            {
                "method": "full multimodal",
                "gallery_size_target": 100,
                "gallery_size_actual": 95.0,
                "coverage": 0.9,
                "metric_name": "R@1",
                "metric_value": 0.6,
            },
            {
                "method": "skymap-only",
                "gallery_size_target": 10,
                "gallery_size_actual": 10.0,
                "coverage": 1.0,
                "metric_name": "R@5",
                "metric_value": 0.7,
            },
            {
                "method": "full multimodal",
                "gallery_size_target": 10,
                "gallery_size_actual": 10.0,
                "coverage": 1.0,
                "metric_name": "R@5",
                "metric_value": 0.9,
            },
            {
                "method": "skymap-only",
                "gallery_size_target": 10,
                "gallery_size_actual": 10.0,
                "coverage": 1.0,
                "metric_name": "R@10",
                "metric_value": 0.9,
            },
            {
                "method": "full multimodal",
                "gallery_size_target": 10,
                "gallery_size_actual": 10.0,
                "coverage": 1.0,
                "metric_name": "R@10",
                "metric_value": 1.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            retrieval_gallery.plot_retrieval_curves(curve_rows, out_dir)
            retrieval_gallery.plot_retrieval_coverage(curve_rows, out_dir)

            self.assertTrue((out_dir / "retrieval_curves.png").exists())
            self.assertTrue((out_dir / "retrieval_curves.pdf").exists())
            self.assertTrue((out_dir / "retrieval_coverage.png").exists())
            self.assertFalse((out_dir / "retrieval_coverage.pdf").exists())


if __name__ == "__main__":
    unittest.main()
