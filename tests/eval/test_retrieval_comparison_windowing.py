"""Unit tests for random-mode input window cropping and gallery stabilisation."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
SCRIPT_DIR = MODEL_DIR / "scripts" / "eval"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import apply_runtime_input_window_torch  # noqa: E402


def _make_four_pixel_skymap(probabilities=(0.60, 0.20, 0.15, 0.05), areas=None):
    """Build a tiny skymap with four equatorial pixels and known probability mass."""
    pix_xyz = np.asarray(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    dA = np.ones((1, 4), dtype=np.float32) if areas is None else np.asarray(areas, dtype=np.float32).reshape(1, 4)
    dP = np.asarray(probabilities, dtype=np.float32).reshape(1, 4)
    distmu = np.zeros((1, 4), dtype=np.float32)
    distsigma = np.ones((1, 4), dtype=np.float32)
    return np.vstack([pix_xyz, dA, dP, distmu, distsigma])


def _make_minimal_h5(path: str, n_optical: int = 20, n_gw: int = 10):
    """Create a minimal test HDF5 file with the expected structure."""
    n_t, n_bands = 10, 1
    with h5py.File(path, "w") as f:
        f.attrs["time_window_start"] = -0.3
        f.attrs["time_window_end"] = 0.6
        opt = f.create_group("events/optical_data")
        # 2D times, 3D values/masks/errors (samples × time_steps × bands)
        times = np.linspace(-0.2, 0.3, n_optical * n_t).reshape(n_optical, n_t).astype(np.float32)
        opt.create_dataset("times", data=times)
        opt.create_dataset("values", data=np.random.randn(n_optical, n_t, n_bands).astype(np.float32))
        opt.create_dataset("masks", data=np.ones((n_optical, n_t, n_bands), dtype=np.float32))
        opt.create_dataset("errors", data=np.abs(np.random.randn(n_optical, n_t, n_bands)).astype(np.float32))
        opt.create_dataset("coordinates", data=np.random.uniform(0, 360, (n_optical, 2)).astype(np.float32))
        opt.create_dataset("parent_gw_idx", data=np.random.randint(0, n_gw, n_optical).astype(np.int64))
        opt.create_dataset("zero_time_mjd_base", data=np.linspace(60001, 60051, n_optical).astype(np.float64))
        gw = f.create_group("events/gw_data")
        gw.create_dataset("scalars", data=np.random.randn(n_gw, 16).astype(np.float32))
        gw.create_dataset("skymaps", data=np.random.randn(n_gw, 5, 19200).astype(np.float32))
        gw.create_dataset("event_time_mjd", data=np.linspace(60000, 60050, n_gw).astype(np.float64))


class TestApplyRuntimeInputWindow(unittest.TestCase):
    def test_window_zeros_outside(self):
        opt_t = torch.tensor([[-0.5, -0.05, 0.0, 0.15, 0.5]], dtype=torch.float32)
        opt_v = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=torch.float32)
        opt_mask = torch.ones_like(opt_v)
        opt_err = torch.abs(opt_v)

        ct, cv, cm, ce, _ = apply_runtime_input_window_torch(
            opt_t, opt_v, opt_mask, opt_err,
            window_start=-0.1, window_end=0.2,
        )
        # Slot 0 at -0.5: outside
        self.assertEqual(float(ct[0, 0]), 0.0)
        self.assertEqual(float(cv[0, 0, 0]), 0.0)
        self.assertEqual(float(cm[0, 0, 0]), 0.0)
        # Slot 2 at 0.0: inside
        self.assertAlmostEqual(float(ct[0, 2]), 0.0)
        self.assertEqual(float(cv[0, 2, 0]), 3.0)
        self.assertEqual(float(cm[0, 2, 0]), 1.0)
        # Slot 4 at 0.5: outside
        self.assertEqual(float(ct[0, 4]), 0.0)
        self.assertEqual(float(cv[0, 4, 0]), 0.0)

    def test_window_none_skips_cropping(self):
        opt_t = torch.tensor([[-0.5, 0.0]], dtype=torch.float32)
        opt_v = torch.tensor([[[1.0], [2.0]]], dtype=torch.float32)
        opt_mask = torch.ones_like(opt_v)
        opt_err = torch.abs(opt_v)
        ct, cv, cm, ce, _ = apply_runtime_input_window_torch(
            opt_t, opt_v, opt_mask, opt_err,
            window_start=None, window_end=None,
        )
        self.assertEqual(float(ct[0, 0]), -0.5)
        self.assertEqual(float(cv[0, 0, 0]), 1.0)


class TestLoadSelectedPositiveBankWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.h5_path = os.path.join(cls.tmpdir, "test.h5")
        _make_minimal_h5(cls.h5_path, n_optical=20, n_gw=10)
        # Extend sys.path so eval_retrieval_comparison is importable
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)

    def test_no_window_keeps_all_data(self):
        from scripts.eval.eval_retrieval_comparison import load_selected_positive_bank
        bank, remap = load_selected_positive_bank(
            self.h5_path,
            np.array([0, 1, 2], dtype=np.int64),
        )
        self.assertEqual(bank["times"].shape, (3, 10))
        self.assertGreater(float(bank["times"].abs().sum()), 0.0)

    def test_window_crops_outside(self):
        from scripts.eval.eval_retrieval_comparison import load_selected_positive_bank
        bank, remap = load_selected_positive_bank(
            self.h5_path,
            np.array([0, 1, 2], dtype=np.int64),
            runtime_input_window_start=-0.1,
            runtime_input_window_end=0.2,
        )
        times = bank["times"]
        # Times outside [-0.1, 0.2] must be zeroed
        outside = (times > -0.1) & (times < 0.2)
        # All non-zero times must be inside the window
        nonzero = times.abs() > 1e-6
        all_inside = torch.all(~nonzero | outside)
        self.assertTrue(bool(all_inside))

    def test_remap_preserves_unique(self):
        from scripts.eval.eval_retrieval_comparison import load_selected_positive_bank
        selected = np.array([5, 2, 5, 2, 10], dtype=np.int64)
        bank, remap = load_selected_positive_bank(self.h5_path, selected)
        unique = sorted(set(selected.tolist()))
        self.assertEqual(len(remap), len(unique))
        for compact_idx, source_idx in enumerate(unique):
            self.assertEqual(remap[source_idx], compact_idx)

    def test_raw_fields_match_main(self):
        from scripts.eval.eval_retrieval_comparison import load_selected_positive_bank
        bank, _ = load_selected_positive_bank(
            self.h5_path,
            np.array([0, 1], dtype=np.int64),
            runtime_input_window_start=-0.1,
            runtime_input_window_end=0.2,
        )
        raw_keys = {
            "times": "opt_t_raw",
            "values": "opt_v_raw",
            "masks": "opt_mask_raw",
            "errors": "opt_err_raw",
        }
        for field in ["times", "values", "masks", "errors"]:
            main = bank[field]
            raw = bank[raw_keys[field]]
            self.assertTrue(torch.equal(main, raw), f"{field} vs raw mismatch")


class TestNormalizeSharedConfigWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)
        cls.tmpdir = tempfile.mkdtemp()
        cls.h5_path = os.path.join(cls.tmpdir, "test.h5")
        _make_minimal_h5(cls.h5_path, n_optical=4, n_gw=2)

    def test_opt_ref_preferred(self):
        from scripts.eval.eval_retrieval_comparison import normalize_shared_config
        cfg = normalize_shared_config(
            {"test_data_path": self.h5_path, "opt_ref_start": -0.3, "opt_ref_end": 0.6},
            Path("/tmp/test.json"),
        )
        self.assertEqual(cfg["comparison_window"], [-0.3, 0.6])
        self.assertEqual(cfg["comparison_window_source"], "opt_ref")

    def test_fallback_to_old_fields(self):
        from scripts.eval.eval_retrieval_comparison import normalize_shared_config
        cfg = normalize_shared_config(
            {"test_data_path": self.h5_path, "comparison_window_start": -0.2, "comparison_window_end": 0.3},
            Path("/tmp/test.json"),
        )
        self.assertEqual(cfg["comparison_window"], [-0.2, 0.3])
        self.assertEqual(cfg["comparison_window_source"], "comparison_window_start_end")

    def test_default_fallback(self):
        from scripts.eval.eval_retrieval_comparison import normalize_shared_config
        cfg = normalize_shared_config({"test_data_path": self.h5_path}, Path("/tmp/test.json"))
        self.assertEqual(cfg["comparison_window"], [-0.1, 0.2])
        self.assertEqual(cfg["comparison_window_source"], "default")

    def test_retrieval_comparison_default_gallery_window_is_post_30_days(self):
        from scripts.eval.eval_retrieval_comparison import normalize_shared_config

        cfg = normalize_shared_config({"test_data_path": self.h5_path}, Path("/tmp/test.json"))

        self.assertEqual(cfg["gallery_candidate_time_window_days"], 30.0)

    def test_gw170817a_default_gallery_window_is_post_30_days(self):
        from eval_gw170817a_retrieval import normalize_config

        cfg = normalize_config({"test_data_path": self.h5_path}, Path("/tmp/test.json"))

        self.assertEqual(cfg["gallery_candidate_time_window_days"], 30.0)


class TestPrecomputeTutorialGalleries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)

    def test_same_positive_per_trial_gw(self):
        from scripts.eval.eval_retrieval_comparison import precompute_tutorial_galleries
        gw_map = {0: [0, 1, 2], 1: [3, 4]}
        galleries, unique = precompute_tutorial_galleries(
            gw_map, n_tutorial_negatives=500,
            gallery_sizes=[10, 100], n_trials=3, seed=42,
        )
        for trial in range(3):
            for gw_id in [0, 1]:
                pos_10 = galleries[(10, trial, gw_id)]["positive_index"]
                pos_100 = galleries[(100, trial, gw_id)]["positive_index"]
                self.assertEqual(pos_10, pos_100,
                                 f"trial={trial} gw={gw_id}: positive should match across sizes")

    def test_negative_prefix(self):
        from scripts.eval.eval_retrieval_comparison import precompute_tutorial_galleries
        gw_map = {0: [0, 1, 2], 1: [3, 4]}
        galleries, _ = precompute_tutorial_galleries(
            gw_map, n_tutorial_negatives=500,
            gallery_sizes=[10, 100], n_trials=3, seed=42,
        )
        for trial in range(3):
            for gw_id in [0, 1]:
                neg_10 = galleries[(10, trial, gw_id)]["negative_indices"]
                neg_100 = galleries[(100, trial, gw_id)]["negative_indices"]
                self.assertEqual(len(neg_10), 9)
                self.assertEqual(len(neg_100), 99)
                np.testing.assert_array_equal(neg_100[:9], neg_10)


class TestGalleryCoverageFillRatio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)

    def test_gallery_coverage_is_mean_fill_ratio(self):
        from retrieval_gallery import aggregate_gallery_outcomes

        outcomes = {
            (500, 0, 1): {"rank": 0, "actual_gallery_size": 100, "coverage_met": False},
            (500, 0, 2): {"rank": 0, "actual_gallery_size": 250, "coverage_met": False},
        }

        _metrics, _by_source, coverage = aggregate_gallery_outcomes(
            outcomes=outcomes,
            gallery_sizes=[500],
            n_trials=1,
            unique_gw=[1, 2],
        )

        self.assertAlmostEqual(coverage["gallery_500"]["coverage"], 0.35)
        self.assertAlmostEqual(coverage["gallery_500"]["fill_ratio_mean"], 0.35)
        self.assertAlmostEqual(coverage["gallery_500"]["full_coverage"], 0.0)

    def test_main_redshift_coverage_is_mean_fill_ratio(self):
        from scripts.eval.eval_retrieval_comparison import _aggregate_redshift_metrics

        outcomes = {
            (500, 0, 1): {"rank": 0, "actual_gallery_size": 100, "coverage_met": False},
            (500, 0, 2): {"rank": 1, "actual_gallery_size": 250, "coverage_met": False},
        }
        redshift_metadata = {
            1: {"redshift": 0.05},
            2: {"redshift": 0.06},
        }

        rows = _aggregate_redshift_metrics(
            outcomes=outcomes,
            gallery_sizes=[500],
            n_trials=1,
            unique_gw=[1, 2],
            redshift_metadata=redshift_metadata,
            bin_edges=[0.0, 0.1],
            bin_labels=["near"],
            method_name="test",
        )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["coverage"], 0.35)
        self.assertAlmostEqual(rows[0]["full_coverage"], 0.0)

    def test_gw170817_redshift_coverage_is_mean_fill_ratio(self):
        from eval_gw170817a_retrieval import aggregate_redshift_metrics

        outcomes = {
            (500, 0, 1): {"rank": 0, "actual_gallery_size": 100, "coverage_met": False},
            (500, 0, 2): {"rank": 1, "actual_gallery_size": 250, "coverage_met": False},
        }
        redshift_metadata = {
            1: {"redshift": 0.05, "redshift_bin": 0},
            2: {"redshift": 0.06, "redshift_bin": 0},
        }

        rows = aggregate_redshift_metrics(
            outcomes=outcomes,
            gallery_sizes=[500],
            n_trials=1,
            unique_gw=[1, 2],
            redshift_metadata=redshift_metadata,
            method_name="test",
        )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["coverage"], 0.35)
        self.assertAlmostEqual(rows[0]["full_coverage"], 0.0)

    def test_gw170817_log10_weighted_macro_metrics_across_gallery_sizes(self):
        from eval_gw170817a_retrieval import aggregate_redshift_macro_metrics

        rows = [
            {
                "method": "test",
                "redshift_bin": 0,
                "redshift": 0.03,
                "gallery_size": 10,
                "n_queries": 4,
                "recall_at_1": 0.25,
                "recall_at_10": 0.75,
                "mrr": 0.5,
            },
            {
                "method": "test",
                "redshift_bin": 0,
                "redshift": 0.03,
                "gallery_size": 1000,
                "n_queries": 4,
                "recall_at_1": 0.75,
                "recall_at_10": 0.25,
                "mrr": 0.9,
            },
        ]

        macro_rows = aggregate_redshift_macro_metrics(rows)

        self.assertEqual(len(macro_rows), 1)
        macro = macro_rows[0]
        self.assertAlmostEqual(macro["macro_recall_at_1"], (0.25 * 1.0 + 0.75 * 3.0) / 4.0)
        self.assertAlmostEqual(macro["macro_recall_at_10"], (0.75 * 1.0 + 0.25 * 3.0) / 4.0)
        self.assertAlmostEqual(macro["macro_mrr"], (0.5 * 1.0 + 0.9 * 3.0) / 4.0)
        self.assertAlmostEqual(macro["gallery_weight_sum"], 4.0)
        self.assertEqual(macro["n_gallery_sizes"], 2)


class TestSkymapCredibleLevels(unittest.TestCase):
    def test_single_gw_credible_level_uses_probability_mass(self):
        from retrieval_gallery import compute_credible_levels_single_gw

        skymap = torch.from_numpy(_make_four_pixel_skymap())
        coords = torch.tensor(
            [
                [0.0, 0.0],
                [np.pi / 2.0, 0.0],
                [np.pi, 0.0],
                [3.0 * np.pi / 2.0, 0.0],
            ],
            dtype=torch.float32,
        )

        credible = compute_credible_levels_single_gw(skymap, coords)

        expected = torch.tensor([0.60, 0.80, 0.95, 1.00], dtype=torch.float32)
        torch.testing.assert_close(credible, expected, rtol=1e-6, atol=1e-6)

    def test_time_sky_candidate_sequences_filter_by_probability_mass_credible_level(self):
        from retrieval_gallery import build_time_sky_candidate_sequences

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "skymap.h5")
            with h5py.File(h5_path, "w") as f:
                gw = f.create_group("events/gw_data")
                gw.create_dataset("event_time_mjd", data=np.asarray([60000.0], dtype=np.float64))
                gw.create_dataset(
                    "skymaps",
                    data=_make_four_pixel_skymap().reshape(1, 7, 4),
                )

            neg_optical = {
                "zero_time_mjd_cls_base": np.asarray([60000.0, 60000.0, 60000.0, 60000.0], dtype=np.float64),
                "coordinates": np.asarray(
                    [
                        [0.0, 0.0],
                        [np.pi / 2.0, 0.0],
                        [np.pi, 0.0],
                        [3.0 * np.pi / 2.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            }

            candidate_sequences, _, _ = build_time_sky_candidate_sequences(
                test_data_path=h5_path,
                unique_gw_ids=[0],
                neg_optical_data=neg_optical,
                n_trials=1,
                seed=42,
                time_window_days=1.0,
                credible_level_max=0.90,
            )

        seq = candidate_sequences[(0, 0)]
        np.testing.assert_array_equal(seq["candidate_indices"], np.asarray([0, 1], dtype=np.int64))
        np.testing.assert_allclose(seq["credible_levels"], np.asarray([0.60, 0.80], dtype=np.float32))

    def test_training_credible_level_uses_probability_mass(self):
        from scripts.train.train import compute_credible_level

        single_skymap = torch.from_numpy(_make_four_pixel_skymap())
        skymaps = single_skymap.unsqueeze(0).repeat(4, 1, 1)
        coords = torch.tensor(
            [
                [0.0, 0.0],
                [np.pi / 2.0, 0.0],
                [np.pi, 0.0],
                [3.0 * np.pi / 2.0, 0.0],
            ],
            dtype=torch.float32,
        )

        credible = compute_credible_level(skymaps, coords).squeeze(-1)

        expected = torch.tensor([0.60, 0.80, 0.95, 1.00], dtype=torch.float32)
        torch.testing.assert_close(credible, expected, rtol=1e-6, atol=1e-6)

    def test_credible_level_ranks_variable_area_moc_cells_by_probability_density(self):
        from retrieval_gallery import compute_credible_levels_single_gw

        skymap = torch.from_numpy(
            _make_four_pixel_skymap(
                probabilities=(0.40, 0.30, 0.20, 0.10),
                areas=(4.0, 1.0, 1.0, 1.0),
            )
        )
        coords = torch.tensor(
            [
                [0.0, 0.0],
                [np.pi / 2.0, 0.0],
                [np.pi, 0.0],
                [3.0 * np.pi / 2.0, 0.0],
            ],
            dtype=torch.float32,
        )

        credible = compute_credible_levels_single_gw(skymap, coords)

        expected = torch.tensor([1.00, 0.30, 0.50, 1.00], dtype=torch.float32)
        torch.testing.assert_close(credible, expected, rtol=1e-6, atol=1e-6)


class TestSyntheticTimeSkyGallery(unittest.TestCase):
    def test_time_sky_candidate_sequence_keeps_only_post_gw_window(self):
        from retrieval_gallery import build_time_sky_candidate_sequence

        seq = build_time_sky_candidate_sequence(
            anchor_time_mjd=60000.0,
            candidate_zero_time_mjd=np.asarray(
                [59999.0, 60000.0, 60010.0, 60030.0, 60031.0],
                dtype=np.float64,
            ),
            candidate_credible_levels=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
            time_window_days=30.0,
            credible_level_max=0.9,
            seed=7,
        )

        np.testing.assert_array_equal(
            np.sort(seq["candidate_indices"]),
            np.asarray([1, 2, 3], dtype=np.int64),
        )
        np.testing.assert_allclose(
            np.sort(seq["abs_dt_days"]),
            np.asarray([0.0, 10.0, 30.0], dtype=np.float32),
        )

    def test_synthetic_gallery_fully_covers_sizes_and_ignores_original_negative_time_sky(self):
        from retrieval_gallery import (
            build_prefixed_gallery_specs,
            build_synthetic_time_sky_candidate_sequences,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "synthetic.h5")
            with h5py.File(h5_path, "w") as f:
                gw = f.create_group("events/gw_data")
                gw.create_dataset("event_time_mjd", data=np.asarray([60000.0, 60010.0], dtype=np.float64))
                skymaps = np.stack([_make_four_pixel_skymap(), _make_four_pixel_skymap()], axis=0)
                gw.create_dataset("skymaps", data=skymaps.astype(np.float32))

            neg_optical = {
                # Deliberately far away from both GW times; synthetic mode must ignore these.
                "zero_time_mjd_cls_base": np.linspace(50000.0, 50100.0, 20, dtype=np.float64),
                # Deliberately outside the high-probability test cells; synthetic mode must ignore these too.
                "coordinates": np.full((20, 2), 180.0, dtype=np.float32),
            }

            seqs, skymaps_a, times_a = build_synthetic_time_sky_candidate_sequences(
                test_data_path=h5_path,
                unique_gw_ids=[0, 1],
                neg_optical_data=neg_optical,
                gallery_sizes=[3, 6],
                n_trials=2,
                seed=123,
                time_window_days=5.0,
                credible_level_max=0.80,
            )
            seqs_repeat, _skymaps_b, times_b = build_synthetic_time_sky_candidate_sequences(
                test_data_path=h5_path,
                unique_gw_ids=[0, 1],
                neg_optical_data=neg_optical,
                gallery_sizes=[3, 6],
                n_trials=2,
                seed=123,
                time_window_days=5.0,
                credible_level_max=0.80,
            )

            galleries, unique_gw = build_prefixed_gallery_specs(
                gw_positive_indices={0: [100], 1: [200]},
                candidate_sequences=seqs,
                gallery_sizes=[3, 6],
                n_trials=2,
                seed=123,
                include_undersized=True,
            )

        self.assertEqual(unique_gw, [0, 1])
        self.assertEqual(sorted(skymaps_a), [0, 1])
        self.assertEqual(times_a, times_b)

        for key, seq in seqs.items():
            repeat = seqs_repeat[key]
            self.assertEqual(seq["candidate_indices"].shape[0], 5)
            np.testing.assert_array_equal(seq["candidate_indices"], repeat["candidate_indices"])
            np.testing.assert_allclose(seq["synthetic_coordinates"], repeat["synthetic_coordinates"])
            np.testing.assert_allclose(seq["synthetic_zero_time_mjd_cls_base"], repeat["synthetic_zero_time_mjd_cls_base"])
            gw_time = times_a[key[1]]
            signed_dt = seq["synthetic_zero_time_mjd_cls_base"] - gw_time
            self.assertTrue(np.all(signed_dt >= 0.0))
            self.assertTrue(np.all(signed_dt <= 5.0 + 1e-6))
            self.assertTrue(np.all(seq["abs_dt_days"] <= 5.0 + 1e-6))
            self.assertTrue(np.all(seq["credible_levels"] <= 0.80 + 1e-6))

        for trial in range(2):
            for gw_id in [0, 1]:
                small = galleries[(3, trial, gw_id)]
                large = galleries[(6, trial, gw_id)]
                self.assertEqual(small["actual_gallery_size"], 3)
                self.assertEqual(large["actual_gallery_size"], 6)
                self.assertTrue(small["coverage_met"])
                self.assertTrue(large["coverage_met"])
                self.assertFalse(small["is_undersized"])
                self.assertFalse(large["is_undersized"])
                np.testing.assert_array_equal(large["negative_indices"][:2], small["negative_indices"])
                np.testing.assert_allclose(
                    large["negative_synthetic_coordinates"][:2],
                    small["negative_synthetic_coordinates"],
                )
                np.testing.assert_allclose(
                    large["negative_synthetic_zero_time_mjd_cls_base"][:2],
                    small["negative_synthetic_zero_time_mjd_cls_base"],
                )


class TestSyntheticCoordinateScoringHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)

    def test_synthetic_coordinates_recompute_negative_z_l_from_curve_features(self):
        from scripts.eval.eval_retrieval_comparison import _candidate_z_l_for_chunk

        class FakeOpticalEncoder:
            def encode_coord_only(self, opt_coords):
                return opt_coords

            def contrastive_head(self, z_curve, coord_feat):
                return z_curve + coord_feat

        class FakeModel:
            def __init__(self):
                self.optical_encoder = FakeOpticalEncoder()

        candidate_bank = {
            "z_l_cls": torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32),
            "z_curve_cls": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        }
        chunk_idx = torch.tensor([0, 1], dtype=torch.long)
        synthetic_coords = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)

        z_l = _candidate_z_l_for_chunk(
            FakeModel(),
            candidate_bank,
            chunk_idx,
            synthetic_coords,
            torch.device("cpu"),
            use_synthetic_coords=True,
        )

        expected = torch.tensor([[1.5, 3.5], [5.5, 7.5]], dtype=torch.float32)
        torch.testing.assert_close(z_l, expected)

    def test_candidate_coords_for_chunk_prefers_synthetic_coordinates(self):
        from scripts.eval.eval_retrieval_comparison import _candidate_coords_for_chunk

        candidate_bank = {
            "opt_coords": torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32),
        }
        chunk_idx = torch.tensor([0, 1], dtype=torch.long)
        synthetic_coords = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        coords = _candidate_coords_for_chunk(
            candidate_bank,
            chunk_idx,
            synthetic_coords,
            start=0,
            end=2,
            device=torch.device("cpu"),
        )

        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        torch.testing.assert_close(coords, expected)


class TestSkymapGalleryModeSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)

    def test_synthetic_time_sky_supports_skymap_only(self):
        from scripts.eval.eval_retrieval_comparison import _gallery_mode_supports_skymap_only

        self.assertTrue(_gallery_mode_supports_skymap_only("time_sky_hard"))
        self.assertTrue(_gallery_mode_supports_skymap_only("synthetic_time_sky_hard"))
        self.assertFalse(_gallery_mode_supports_skymap_only("tutorial_random"))


# ---------------------------------------------------------------------------
# Helper to create minimal HDF5 with GW ids and source_type for redshift tests
# ---------------------------------------------------------------------------
def _make_redshift_test_h5(path: str, gw_records):
    """Create a minimal HDF5 with events/gw_data/ids and source_type.

    Args:
        path: File path for the new HDF5.
        gw_records: List of (id_str, source_str) tuples, e.g.
            [("bns_6", "bns"), ("nsbh_114", "nsbh")].
    """
    n_gw = len(gw_records)
    n_optical = max(n_gw * 2, 4)
    n_t, n_bands = 10, 1
    with h5py.File(path, "w") as f:
        f.attrs["time_window_start"] = -0.3
        f.attrs["time_window_end"] = 0.6
        opt = f.create_group("events/optical_data")
        times = np.linspace(-0.2, 0.3, n_optical * n_t).reshape(n_optical, n_t).astype(np.float32)
        opt.create_dataset("times", data=times)
        opt.create_dataset("values", data=np.random.randn(n_optical, n_t, n_bands).astype(np.float32))
        opt.create_dataset("masks", data=np.ones((n_optical, n_t, n_bands), dtype=np.float32))
        opt.create_dataset("errors", data=np.abs(np.random.randn(n_optical, n_t, n_bands)).astype(np.float32))
        opt.create_dataset("coordinates", data=np.random.uniform(0, 360, (n_optical, 2)).astype(np.float32))
        opt.create_dataset("parent_gw_idx", data=np.random.randint(0, n_gw, n_optical).astype(np.int64))
        gw = f.create_group("events/gw_data")
        gw.create_dataset("scalars", data=np.random.randn(n_gw, 7).astype(np.float32))
        gw.create_dataset("skymaps", data=np.random.randn(n_gw, 5, 19200).astype(np.float32))
        gw.create_dataset("event_time_mjd", data=np.linspace(60000, 60050, n_gw).astype(np.float64))
        gw.create_dataset("has_kn", data=np.ones(n_gw, dtype=np.int32))
        gw.create_dataset("mej_tot", data=np.random.randn(n_gw).astype(np.float32))
        gw.create_dataset("neg_type", data=np.zeros(n_gw, dtype=np.int32))
        dt = h5py.special_dtype(vlen=str)
        ids_ds = gw.create_dataset("ids", shape=(n_gw,), dtype=dt)
        src_ds = gw.create_dataset("source_type", shape=(n_gw,), dtype=dt)
        for i, (id_str, src_str) in enumerate(gw_records):
            ids_ds[i] = id_str
            src_ds[i] = src_str


# ---------------------------------------------------------------------------
# Task 1: Redshift Metadata Recovery
# ---------------------------------------------------------------------------
class TestParseHDF5GWID(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)

    def test_parse_bns_id(self):
        from scripts.eval.eval_retrieval_comparison import _parse_hdf5_gw_id
        source, event_id = _parse_hdf5_gw_id("bns_6")
        self.assertEqual(source, "bns")
        self.assertEqual(event_id, 6)

    def test_parse_nsbh_id(self):
        from scripts.eval.eval_retrieval_comparison import _parse_hdf5_gw_id
        source, event_id = _parse_hdf5_gw_id("nsbh_114")
        self.assertEqual(source, "nsbh")
        self.assertEqual(event_id, 114)

    def test_parse_stream_id(self):
        from scripts.eval.eval_retrieval_comparison import _parse_hdf5_gw_id
        source, event_id = _parse_hdf5_gw_id("bns_test_pos_15")
        self.assertEqual(source, "bns_test_pos")
        self.assertEqual(event_id, 15)

    def test_parse_bytes_input(self):
        from scripts.eval.eval_retrieval_comparison import _parse_hdf5_gw_id
        source, event_id = _parse_hdf5_gw_id(b"bns_42")
        self.assertEqual(source, "bns")
        self.assertEqual(event_id, 42)

    def test_rejects_malformed_no_underscore(self):
        from scripts.eval.eval_retrieval_comparison import _parse_hdf5_gw_id
        with self.assertRaises(ValueError):
            _parse_hdf5_gw_id("bns6")

    def test_rejects_malformed_non_numeric_event_id(self):
        from scripts.eval.eval_retrieval_comparison import _parse_hdf5_gw_id
        with self.assertRaises(ValueError):
            _parse_hdf5_gw_id("bns_abc")


class TestLoadCatalogRedshiftMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)
        cls.tmpdir = tempfile.mkdtemp()

    def test_loads_valid_csv(self):
        from scripts.eval.eval_retrieval_comparison import _load_catalog_redshift_map
        csv_path = os.path.join(self.tmpdir, "test_bns.csv")
        pd.DataFrame({
            "simulation_id": [0, 2, 6],
            "redshift": [0.05, 0.10, 0.15],
        }).to_csv(csv_path, index=False)
        z_map = _load_catalog_redshift_map(csv_path)
        self.assertEqual(z_map, {0: 0.05, 2: 0.10, 6: 0.15})

    def test_missing_redshift_column_raises(self):
        from scripts.eval.eval_retrieval_comparison import _load_catalog_redshift_map
        csv_path = os.path.join(self.tmpdir, "no_z.csv")
        pd.DataFrame({"simulation_id": [1, 2]}).to_csv(csv_path, index=False)
        with self.assertRaises(KeyError):
            _load_catalog_redshift_map(csv_path)

    def test_missing_simulation_id_column_raises(self):
        from scripts.eval.eval_retrieval_comparison import _load_catalog_redshift_map
        csv_path = os.path.join(self.tmpdir, "no_sim.csv")
        pd.DataFrame({"redshift": [0.1, 0.2]}).to_csv(csv_path, index=False)
        with self.assertRaises(KeyError):
            _load_catalog_redshift_map(csv_path)

    def test_file_not_found_raises(self):
        from scripts.eval.eval_retrieval_comparison import _load_catalog_redshift_map
        with self.assertRaises(FileNotFoundError):
            _load_catalog_redshift_map("/nonexistent/path.csv")


class TestNormalizeRedshiftCatalogPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)

    def test_relative_paths_resolved(self):
        from scripts.eval.eval_retrieval_comparison import _normalize_redshift_catalog_paths
        cfg = {"bns": "relative/path.csv", "nsbh": "/absolute/path.csv"}
        cfg_dir = Path("/tmp/config")
        result = _normalize_redshift_catalog_paths(cfg, cfg_dir)
        self.assertEqual(result["bns"], str((cfg_dir / "relative/path.csv").resolve()))
        self.assertEqual(result["nsbh"], "/absolute/path.csv")

    def test_empty_config(self):
        from scripts.eval.eval_retrieval_comparison import _normalize_redshift_catalog_paths
        result = _normalize_redshift_catalog_paths({}, Path("/tmp"))
        self.assertEqual(result, {})


class TestBuildRedshiftMetadataFromCatalogs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)
        cls.tmpdir = tempfile.mkdtemp()

    def _make_catalog(self, filename, rows):
        path = os.path.join(self.tmpdir, filename)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _make_h5(self, filename, gw_records):
        path = os.path.join(self.tmpdir, filename)
        _make_redshift_test_h5(path, gw_records)
        return path

    def test_recovers_redshift_for_bns_and_nsbh(self):
        from scripts.eval.eval_retrieval_comparison import _build_redshift_metadata_from_catalogs
        bns_path = self._make_catalog("bns.csv", [
            {"simulation_id": 6, "redshift": 0.05},
            {"simulation_id": 7, "redshift": 0.08},
        ])
        nsbh_path = self._make_catalog("nsbh.csv", [
            {"simulation_id": 114, "redshift": 0.12},
        ])
        h5_path = self._make_h5("test.h5", [
            ("bns_6", "bns"),
            ("bns_7", "bns"),
            ("nsbh_114", "nsbh"),
        ])
        catalogs = {"bns": bns_path, "nsbh": nsbh_path}
        meta = _build_redshift_metadata_from_catalogs(h5_path, catalogs, validate_scalars=False)
        self.assertEqual(len(meta), 3)
        self.assertAlmostEqual(meta[0]["redshift"], 0.05)
        self.assertAlmostEqual(meta[1]["redshift"], 0.08)
        self.assertAlmostEqual(meta[2]["redshift"], 0.12)

    def test_recovers_redshift_from_stream_specific_catalogs(self):
        from scripts.eval.eval_retrieval_comparison import _build_redshift_metadata_from_catalogs
        pos_path = self._make_catalog("bns_test_pos.csv", [
            {"simulation_id": 6, "redshift": 0.05},
        ])
        neg_path = self._make_catalog("bns_test_neg.csv", [
            {"simulation_id": 6, "redshift": 0.15},
        ])
        h5_path = self._make_h5("stream_test.h5", [
            ("bns_test_pos_6", "bns"),
            ("bns_test_neg_6", "bns"),
        ])
        catalogs = {"bns_test_pos": pos_path, "bns_test_neg": neg_path}
        meta = _build_redshift_metadata_from_catalogs(
            h5_path, catalogs, validate_scalars=False
        )
        self.assertAlmostEqual(meta[0]["redshift"], 0.05)
        self.assertAlmostEqual(meta[1]["redshift"], 0.15)

    def test_unmatched_event_id_raises(self):
        from scripts.eval.eval_retrieval_comparison import _build_redshift_metadata_from_catalogs
        bns_path = self._make_catalog("bns.csv", [
            {"simulation_id": 999, "redshift": 0.05},
        ])
        h5_path = self._make_h5("test.h5", [("bns_6", "bns")])
        with self.assertRaises(ValueError):
            _build_redshift_metadata_from_catalogs(h5_path, {"bns": bns_path}, validate_scalars=False)

    def test_missing_catalog_for_source_raises(self):
        from scripts.eval.eval_retrieval_comparison import _build_redshift_metadata_from_catalogs
        h5_path = self._make_h5("test.h5", [("bns_6", "bns")])
        with self.assertRaises(ValueError):
            _build_redshift_metadata_from_catalogs(h5_path, {}, validate_scalars=False)


def _make_redshift_test_h5_with_scalars(path: str, gw_records):
    """Create a minimal HDF5 with specific scalar values.

    Args:
        path: File path for the new HDF5.
        gw_records: List of (id_str, source_str, scalars_array) tuples.
            scalars_array must have shape (7,) matching _SCALAR_VALIDATION_COLS.
    """
    n_gw = len(gw_records)
    n_optical = max(n_gw * 2, 4)
    n_t, n_bands = 10, 1
    with h5py.File(path, "w") as f:
        f.attrs["time_window_start"] = -0.3
        f.attrs["time_window_end"] = 0.6
        opt = f.create_group("events/optical_data")
        times = np.linspace(-0.2, 0.3, n_optical * n_t).reshape(n_optical, n_t).astype(np.float32)
        opt.create_dataset("times", data=times)
        opt.create_dataset("values", data=np.random.randn(n_optical, n_t, n_bands).astype(np.float32))
        opt.create_dataset("masks", data=np.ones((n_optical, n_t, n_bands), dtype=np.float32))
        opt.create_dataset("errors", data=np.abs(np.random.randn(n_optical, n_t, n_bands)).astype(np.float32))
        opt.create_dataset("coordinates", data=np.random.uniform(0, 360, (n_optical, 2)).astype(np.float32))
        opt.create_dataset("parent_gw_idx", data=np.random.randint(0, n_gw, n_optical).astype(np.int64))
        gw = f.create_group("events/gw_data")
        scalars_data = np.stack([rec[2] for rec in gw_records], axis=0).astype(np.float32)
        gw.create_dataset("scalars", data=scalars_data)
        gw.create_dataset("skymaps", data=np.random.randn(n_gw, 5, 19200).astype(np.float32))
        gw.create_dataset("event_time_mjd", data=np.linspace(60000, 60050, n_gw).astype(np.float64))
        gw.create_dataset("has_kn", data=np.ones(n_gw, dtype=np.int32))
        gw.create_dataset("mej_tot", data=np.random.randn(n_gw).astype(np.float32))
        gw.create_dataset("neg_type", data=np.zeros(n_gw, dtype=np.int32))
        dt = h5py.special_dtype(vlen=str)
        ids_ds = gw.create_dataset("ids", shape=(n_gw,), dtype=dt)
        src_ds = gw.create_dataset("source_type", shape=(n_gw,), dtype=dt)
        for i, (id_str, src_str, _sc) in enumerate(gw_records):
            ids_ds[i] = id_str
            src_ds[i] = src_str


class TestSourceTypeMismatchDetection(unittest.TestCase):
    """Fix 2: verify that parsed-ID source is checked against source_type."""

    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)
        cls.tmpdir = tempfile.mkdtemp()

    def _make_catalog(self, filename, rows):
        path = os.path.join(self.tmpdir, filename)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_id_source_mismatches_source_type_raises(self):
        """If ids='bns_6' but source_type='nsbh', raise ValueError."""
        from scripts.eval.eval_retrieval_comparison import _build_redshift_metadata_from_catalogs

        bns_path = self._make_catalog("bns.csv", [
            {"simulation_id": 6, "redshift": 0.05},
        ])
        h5_path = os.path.join(self.tmpdir, "mismatch.h5")
        # ID says "bns_6" but source_type says "nsbh" → mismatch
        _make_redshift_test_h5(h5_path, [("bns_6", "nsbh")])

        with self.assertRaises(ValueError) as ctx:
            _build_redshift_metadata_from_catalogs(
                h5_path, {"nsbh": bns_path}, validate_scalars=False,
            )
        self.assertIn("source mismatch", str(ctx.exception))

    def test_consistent_source_passes(self):
        """If ids='bns_6' and source_type='bns', no error."""
        from scripts.eval.eval_retrieval_comparison import _build_redshift_metadata_from_catalogs

        bns_path = self._make_catalog("bns.csv", [
            {"simulation_id": 6, "redshift": 0.05},
        ])
        h5_path = os.path.join(self.tmpdir, "ok.h5")
        _make_redshift_test_h5(h5_path, [("bns_6", "bns")])

        meta = _build_redshift_metadata_from_catalogs(
            h5_path, {"bns": bns_path}, validate_scalars=False,
        )
        self.assertAlmostEqual(meta[0]["redshift"], 0.05)


class TestScalarConsistencyValidation(unittest.TestCase):
    """Fix 1: verify scalar cross-check against catalog GW parameters."""

    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)
        cls.tmpdir = tempfile.mkdtemp()

    def _make_catalog(self, filename, rows):
        path = os.path.join(self.tmpdir, filename)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _make_scalars(self, mass1, mass2, spin1z, spin2z, incl, distmean, diststd):
        """Build a (7,) scalar array matching _SCALAR_VALIDATION_COLS transforms."""
        return np.array([
            mass1,                  # 0: mass1_detector (direct)
            mass2,                  # 1: mass2_detector (direct)
            spin1z,                 # 2: spin1z (direct)
            spin2z,                 # 3: spin2z (direct)
            np.cos(incl),           # 4: cos(inclination)
            distmean / 1000.0,      # 5: distmean / 1000
            diststd / 1000.0,       # 6: diststd / 1000
        ], dtype=np.float32)

    def test_scalar_validation_passes_with_matching_data(self):
        from scripts.eval.eval_retrieval_comparison import _build_redshift_metadata_from_catalogs

        mass1, mass2 = 1.53, 1.42
        spin1z, spin2z = -0.0014, 0.0259
        incl, distmean, diststd = 1.76, 613.5, 207.6

        bns_path = self._make_catalog("bns.csv", [{
            "simulation_id": 6,
            "redshift": 0.05,
            "mass1_detector": mass1,
            "mass2_detector": mass2,
            "spin1z": spin1z,
            "spin2z": spin2z,
            "inclination": incl,
            "distmean": distmean,
            "diststd": diststd,
        }])
        scalars = self._make_scalars(mass1, mass2, spin1z, spin2z, incl, distmean, diststd)
        h5_path = os.path.join(self.tmpdir, "match.h5")
        _make_redshift_test_h5_with_scalars(h5_path, [("bns_6", "bns", scalars)])

        meta = _build_redshift_metadata_from_catalogs(
            h5_path, {"bns": bns_path}, validate_scalars=True,
        )
        self.assertAlmostEqual(meta[0]["redshift"], 0.05)

    def test_scalar_mismatch_raises(self):
        from scripts.eval.eval_retrieval_comparison import _build_redshift_metadata_from_catalogs

        mass1, mass2 = 1.53, 1.42
        spin1z, spin2z = -0.0014, 0.0259
        incl, distmean, diststd = 1.76, 613.5, 207.6

        bns_path = self._make_catalog("bns.csv", [{
            "simulation_id": 6,
            "redshift": 0.05,
            "mass1_detector": mass1,
            "mass2_detector": mass2,
            "spin1z": spin1z,
            "spin2z": spin2z,
            "inclination": incl,
            "distmean": distmean,
            "diststd": diststd,
        }])
        # Wrong scalar: mass1=999 (should be 1.53)
        wrong_scalars = self._make_scalars(999.0, mass2, spin1z, spin2z, incl, distmean, diststd)
        h5_path = os.path.join(self.tmpdir, "mismatch.h5")
        _make_redshift_test_h5_with_scalars(h5_path, [("bns_6", "bns", wrong_scalars)])

        with self.assertRaises(ValueError) as ctx:
            _build_redshift_metadata_from_catalogs(
                h5_path, {"bns": bns_path}, validate_scalars=True,
            )
        self.assertIn("Scalar mismatch", str(ctx.exception))
        self.assertIn("mass1_detector", str(ctx.exception))

    def test_scalar_validation_skipped_when_disabled(self):
        from scripts.eval.eval_retrieval_comparison import _build_redshift_metadata_from_catalogs

        bns_path = self._make_catalog("bns.csv", [{
            "simulation_id": 6,
            "redshift": 0.05,
            "mass1_detector": 1.53,
            "mass2_detector": 1.42,
            "spin1z": 0.0,
            "spin2z": 0.0,
            "inclination": 0.0,
            "distmean": 100.0,
            "diststd": 10.0,
        }])
        h5_path = os.path.join(self.tmpdir, "skip.h5")
        _make_redshift_test_h5(h5_path, [("bns_6", "bns")])

        meta = _build_redshift_metadata_from_catalogs(
            h5_path, {"bns": bns_path}, validate_scalars=False,
        )
        self.assertAlmostEqual(meta[0]["redshift"], 0.05)


class TestNormalizeRedshiftBinConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)

    def test_valid_config(self):
        from scripts.eval.eval_retrieval_comparison import _normalize_redshift_bin_config
        edges, labels = _normalize_redshift_bin_config(
            [0.0, 0.04, 0.065, 0.10],
            ["z1", "z2", "z3"],
        )
        self.assertEqual(len(edges), 4)
        self.assertEqual(len(labels), 3)

    def test_labels_inferred_from_edges(self):
        from scripts.eval.eval_retrieval_comparison import _normalize_redshift_bin_config
        edges, labels = _normalize_redshift_bin_config(
            [0.0, 0.04, 0.10],
            None,
        )
        self.assertEqual(labels, ["0.00-0.04", "0.04-0.10"])

    def test_label_edge_count_mismatch_raises(self):
        from scripts.eval.eval_retrieval_comparison import _normalize_redshift_bin_config
        with self.assertRaises(ValueError):
            _normalize_redshift_bin_config([0.0, 0.04, 0.10], ["only_one"])

    def test_too_few_edges_raises(self):
        from scripts.eval.eval_retrieval_comparison import _normalize_redshift_bin_config
        with self.assertRaises(ValueError):
            _normalize_redshift_bin_config([0.0], None)


# ---------------------------------------------------------------------------
# Task 2: Redshift Bin Aggregation
# ---------------------------------------------------------------------------
class TestAggregateRedshiftMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = str(SCRIPT_DIR)
        if cls.script_dir not in sys.path:
            sys.path.insert(0, cls.script_dir)

    def test_basic_aggregation(self):
        from scripts.eval.eval_retrieval_comparison import _aggregate_redshift_metrics

        # Two GWs: gw0 (z=0.03, bin 0), gw1 (z=0.05, bin 1)
        redshift_metadata = {
            0: {"redshift": 0.03},
            1: {"redshift": 0.05},
        }
        bin_edges = [0.0, 0.04, 0.065, 0.10]
        bin_labels = ["low", "mid", "high"]

        # gallery_size=10, trial=0, gw0 rank=0, gw1 rank=3
        # gallery_size=10, trial=1, gw0 rank=2, gw1 rank=5
        outcomes = {
            (10, 0, 0): {"rank": 0, "actual_gallery_size": 10, "coverage_met": True},
            (10, 0, 1): {"rank": 3, "actual_gallery_size": 10, "coverage_met": True},
            (10, 1, 0): {"rank": 2, "actual_gallery_size": 10, "coverage_met": True},
            (10, 1, 1): {"rank": 5, "actual_gallery_size": 10, "coverage_met": True},
        }

        rows = _aggregate_redshift_metrics(
            outcomes=outcomes,
            gallery_sizes=[10],
            n_trials=2,
            unique_gw=[0, 1],
            redshift_metadata=redshift_metadata,
            bin_edges=bin_edges,
            bin_labels=bin_labels,
            method_name="test",
        )

        # Should have 2 rows: bin "low" (gw0) and "mid" (gw1)
        self.assertEqual(len(rows), 2)
        low_row = [r for r in rows if r["redshift_bin_label"] == "low"][0]
        mid_row = [r for r in rows if r["redshift_bin_label"] == "mid"][0]

        # gw0: rank 0,2 → R@1=0.5, R@5=1.0, R@10=1.0, MRR=(1+1/3)/2=0.667
        self.assertEqual(low_row["n_queries"], 2)
        self.assertEqual(low_row["recall_at_1"], 0.5)
        self.assertEqual(low_row["recall_at_5"], 1.0)
        self.assertEqual(low_row["recall_at_10"], 1.0)
        self.assertAlmostEqual(low_row["mrr"], (1.0 + 1.0 / 3.0) / 2.0)
        self.assertAlmostEqual(low_row["redshift"], 0.03)
        self.assertEqual(low_row["gallery_size"], 10)
        self.assertEqual(low_row["method"], "test")

        # gw1: rank 3,5 → R@1=0.0, R@5=1.0, R@10=1.0, MRR=(1/4 + 1/6)/2≈0.208
        self.assertEqual(mid_row["n_queries"], 2)
        self.assertAlmostEqual(mid_row["mrr"], (0.25 + 1.0 / 6.0) / 2.0)

    def test_empty_bins_omitted(self):
        from scripts.eval.eval_retrieval_comparison import _aggregate_redshift_metrics

        redshift_metadata = {0: {"redshift": 0.03}}
        bin_edges = [0.0, 0.04, 0.10, 0.20]
        bin_labels = ["low", "mid", "high"]

        outcomes = {
            (10, 0, 0): {"rank": 0, "actual_gallery_size": 10, "coverage_met": True},
        }

        rows = _aggregate_redshift_metrics(
            outcomes=outcomes,
            gallery_sizes=[10],
            n_trials=1,
            unique_gw=[0],
            redshift_metadata=redshift_metadata,
            bin_edges=bin_edges,
            bin_labels=bin_labels,
            method_name="test",
        )
        # Only "low" should appear
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["redshift_bin_label"], "low")

    def test_gw_missing_from_metadata_skipped(self):
        from scripts.eval.eval_retrieval_comparison import _aggregate_redshift_metrics

        redshift_metadata = {0: {"redshift": 0.03}}  # gw1 not in metadata
        bin_edges = [0.0, 0.04, 0.10]
        bin_labels = ["low", "mid"]

        outcomes = {
            (10, 0, 0): {"rank": 0, "actual_gallery_size": 10, "coverage_met": True},
            (10, 0, 1): {"rank": 5, "actual_gallery_size": 10, "coverage_met": True},
        }
        rows = _aggregate_redshift_metrics(
            outcomes=outcomes,
            gallery_sizes=[10],
            n_trials=1,
            unique_gw=[0, 1],
            redshift_metadata=redshift_metadata,
            bin_edges=bin_edges,
            bin_labels=bin_labels,
            method_name="test",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["redshift_bin_label"], "low")

    def test_coverage_and_effective_size(self):
        from scripts.eval.eval_retrieval_comparison import _aggregate_redshift_metrics

        redshift_metadata = {0: {"redshift": 0.03}}
        bin_edges = [0.0, 0.04, 0.10]
        bin_labels = ["low", "mid"]

        outcomes = {
            (10, 0, 0): {"rank": 0, "actual_gallery_size": 8, "coverage_met": False},
            (10, 1, 0): {"rank": 1, "actual_gallery_size": 10, "coverage_met": True},
        }
        rows = _aggregate_redshift_metrics(
            outcomes=outcomes,
            gallery_sizes=[10],
            n_trials=2,
            unique_gw=[0],
            redshift_metadata=redshift_metadata,
            bin_edges=bin_edges,
            bin_labels=bin_labels,
            method_name="test",
        )
        self.assertEqual(rows[0]["coverage"], 0.9)
        self.assertEqual(rows[0]["full_coverage"], 0.5)
        self.assertEqual(rows[0]["effective_gallery_size_mean"], 9.0)

    def test_multiple_gallery_sizes(self):
        from scripts.eval.eval_retrieval_comparison import _aggregate_redshift_metrics

        redshift_metadata = {0: {"redshift": 0.03}}
        bin_edges = [0.0, 0.04, 0.10]
        bin_labels = ["low", "mid"]

        outcomes = {
            (10, 0, 0): {"rank": 0, "actual_gallery_size": 10, "coverage_met": True},
            (100, 0, 0): {"rank": 2, "actual_gallery_size": 100, "coverage_met": True},
        }
        rows = _aggregate_redshift_metrics(
            outcomes=outcomes,
            gallery_sizes=[10, 100],
            n_trials=1,
            unique_gw=[0],
            redshift_metadata=redshift_metadata,
            bin_edges=bin_edges,
            bin_labels=bin_labels,
            method_name="test",
        )
        sizes = {r["gallery_size"] for r in rows}
        self.assertEqual(sizes, {10, 100})

    def test_csv_fieldnames_match_expected(self):
        from scripts.eval.eval_retrieval_comparison import _aggregate_redshift_metrics

        redshift_metadata = {0: {"redshift": 0.03}}
        bin_edges = [0.0, 0.04]
        bin_labels = ["z0"]

        outcomes = {
            (10, 0, 0): {"rank": 0, "actual_gallery_size": 10, "coverage_met": True},
        }
        rows = _aggregate_redshift_metrics(
            outcomes=outcomes,
            gallery_sizes=[10],
            n_trials=1,
            unique_gw=[0],
            redshift_metadata=redshift_metadata,
            bin_edges=bin_edges,
            bin_labels=bin_labels,
            method_name="test",
        )
        expected_fields = {
            "method", "redshift_bin_label", "redshift", "bin_left", "bin_right",
            "gallery_size", "n_queries", "recall_at_1", "recall_at_5",
            "recall_at_10", "mrr", "coverage", "effective_gallery_size_mean",
        }
        self.assertTrue(expected_fields.issubset(set(rows[0].keys())))

    def test_log10_weighted_macro_metrics_across_gallery_sizes(self):
        from scripts.eval.eval_retrieval_comparison import aggregate_redshift_macro_metrics

        rows = [
            {
                "method": "test",
                "redshift_bin_label": "low",
                "bin_left": 0.0,
                "bin_right": 0.04,
                "redshift": 0.03,
                "gallery_size": 10,
                "n_queries": 4,
                "recall_at_1": 0.2,
                "recall_at_10": 0.8,
                "mrr": 0.4,
            },
            {
                "method": "test",
                "redshift_bin_label": "low",
                "bin_left": 0.0,
                "bin_right": 0.04,
                "redshift": 0.03,
                "gallery_size": 1000,
                "n_queries": 4,
                "recall_at_1": 0.8,
                "recall_at_10": 0.2,
                "mrr": 0.7,
            },
        ]

        macro_rows = aggregate_redshift_macro_metrics(rows)

        self.assertEqual(len(macro_rows), 1)
        macro = macro_rows[0]
        # log10(10)=1 and log10(1000)=3, so the larger gallery receives 3x weight.
        self.assertAlmostEqual(macro["macro_recall_at_1"], (0.2 * 1.0 + 0.8 * 3.0) / 4.0)
        self.assertAlmostEqual(macro["macro_recall_at_10"], (0.8 * 1.0 + 0.2 * 3.0) / 4.0)
        self.assertAlmostEqual(macro["macro_mrr"], (0.4 * 1.0 + 0.7 * 3.0) / 4.0)
        self.assertAlmostEqual(macro["gallery_weight_sum"], 4.0)
        self.assertEqual(macro["n_gallery_sizes"], 2)


if __name__ == "__main__":
    unittest.main()
