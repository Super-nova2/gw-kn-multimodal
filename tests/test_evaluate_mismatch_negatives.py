from pathlib import Path
import argparse
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
sys.path.insert(0, str(MODEL_DIR))

from scripts.eval import evaluate as te  # noqa: E402


class TestEvaluateMismatchNegatives(unittest.TestCase):
    def test_easy_mismatch_sampler_uses_single_random_circular_shift(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1234)

        idx = te.sample_easy_mismatch_negatives(12, torch.device("cpu"), generator=generator)

        rows = torch.arange(12)
        shifts = (idx.cpu() - rows) % 12
        self.assertEqual(torch.unique(shifts).numel(), 1)
        self.assertIn(int(shifts[0].item()), range(1, 12))
        self.assertFalse(torch.any(idx.cpu() == rows))

    def test_time_windowed_mismatch_sampler_prefers_actual_in_window_pairs(self):
        rng = np.random.default_rng(123)
        gw_ids = torch.arange(4)
        anchor_times = torch.full((4,), 60000.0)
        candidate_times = torch.tensor([60001.0, 60002.0, 60003.0, 60004.0])
        fallback = torch.tensor([1, 2, 3, 0])

        idx, dt, actual = te.sample_time_windowed_mismatch_negatives(
            gw_ids,
            gw_ids,
            anchor_times,
            candidate_times,
            window_days=30.0,
            rng=rng,
            fallback_indices=fallback,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.all(actual))
        self.assertTrue(torch.all(gw_ids[idx] != gw_ids))
        self.assertTrue(torch.all(dt >= 0.0))
        self.assertTrue(torch.all(dt <= 30.0))
        expected_dt = candidate_times[idx] - anchor_times
        self.assertTrue(torch.allclose(dt, expected_dt.to(dt.dtype)))

    def test_time_windowed_mismatch_sampler_redefines_dt_when_actual_pairs_missing(self):
        rng = np.random.default_rng(456)
        gw_ids = torch.arange(3)
        anchor_times = torch.full((3,), 60000.0)
        candidate_times = torch.full((3,), 59000.0)
        fallback = torch.tensor([1, 2, 0])

        idx, dt, actual = te.sample_time_windowed_mismatch_negatives(
            gw_ids,
            gw_ids,
            anchor_times,
            candidate_times,
            window_days=30.0,
            rng=rng,
            fallback_indices=fallback,
            device=torch.device("cpu"),
        )

        self.assertFalse(torch.any(actual))
        self.assertTrue(torch.equal(idx, fallback))
        self.assertTrue(torch.all(gw_ids[idx] != gw_ids))
        self.assertTrue(torch.all(dt >= 0.0))
        self.assertTrue(torch.all(dt <= 30.0))

    def test_logits_distribution_plot_copy_uses_mismatch_negative_language(self):
        self.assertEqual(te.MISMATCH_NEGATIVE_LABEL, "Mismatched Negatives")
        self.assertEqual(
            te.LOGIT_DISTRIBUTION_TITLE,
            "Classification Logit Distribution by Sample Pairs",
        )
        self.assertEqual(te.LOGIT_AXIS_LABEL, "Logit")

    def test_logits_distribution_plot_legend_is_upper_left_without_pair_details(self):
        source = (MODEL_DIR / "scripts" / "eval" / "evaluate.py").read_text()
        start = source.index("def generate_logits_distribution_plot")
        end = source.index("def generate_gw_shuffle_comparison_plot")
        plot_source = source[start:end]

        self.assertIn("ax.legend(loc='upper left'", plot_source)
        self.assertNotIn("Positive (GW, KN)", plot_source)
        self.assertNotIn("Optical Negatives (GW, nonKN)", plot_source)
        self.assertNotIn("GW Negatives (GW_has_kn0, KN)", plot_source)
        self.assertNotIn("KN_mismatch", plot_source)

    def test_build_test_dataloader_ignores_external_negative_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pos_path = Path(tmpdir) / "positive.h5"
            neg_path = Path(tmpdir) / "negative_without_time_fields.h5"
            n_lc = 3
            n_steps = 4
            n_bands = 2

            with h5py.File(pos_path, "w") as f:
                opt = f.create_group("events/optical_data")
                opt.create_dataset(
                    "values", data=np.zeros((n_lc, n_steps, n_bands), dtype=np.float32)
                )
                opt.create_dataset(
                    "errors", data=np.ones((n_lc, n_steps, n_bands), dtype=np.float32)
                )
                opt.create_dataset(
                    "masks", data=np.ones((n_lc, n_steps, n_bands), dtype=np.float32)
                )
                opt.create_dataset("times", data=np.zeros((n_lc, n_steps), dtype=np.float32))
                opt.create_dataset("coordinates", data=np.zeros((n_lc, 2), dtype=np.float32))
                opt.create_dataset("parent_gw_idx", data=np.arange(n_lc, dtype=np.int64))
                opt.create_dataset(
                    "zero_time_mjd_base", data=np.full(n_lc, 60000.0, dtype=np.float64)
                )
                opt.create_dataset(
                    "first_detection_mjd", data=np.full(n_lc, 60001.0, dtype=np.float64)
                )
                gw = f.create_group("events/gw_data")
                gw.create_dataset("scalars", data=np.zeros((n_lc, 4), dtype=np.float32))
                gw.create_dataset("skymaps", data=np.zeros((n_lc, 1, 4, 4), dtype=np.float32))

            with h5py.File(neg_path, "w") as f:
                neg = f.create_group("ELASTICC/optical_data")
                neg.create_dataset("values", data=np.zeros((2, n_steps, n_bands), dtype=np.float32))
                neg.create_dataset("errors", data=np.ones((2, n_steps, n_bands), dtype=np.float32))
                neg.create_dataset("masks", data=np.ones((2, n_steps, n_bands), dtype=np.float32))
                neg.create_dataset("times", data=np.zeros((2, n_steps), dtype=np.float32))
                neg.create_dataset("coordinates", data=np.zeros((2, 2), dtype=np.float32))

            args = argparse.Namespace(
                test_data_path=str(pos_path),
                neg_data_path=str(neg_path),
                neg_group="ELASTICC/optical_data",
                batch_size=2,
                test_steps=1,
                n_neg_samples=2,
                num_workers=0,
            )

            loader, dataset = te.build_test_dataloader(
                args, {}, return_zero_time_mjd=True, nonkn_cls_base_field="zero_time_mjd_cls_base"
            )

            self.assertIsNone(dataset.negative_h5_path)
            batch = next(iter(loader))
            self.assertEqual(len(batch), 10)


if __name__ == "__main__":
    unittest.main()
