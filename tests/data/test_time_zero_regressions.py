import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
SCRIPT_DIR = MODEL_DIR / "scripts" / "eval"
GW170817A_DATASET_DIR = REPO_ROOT / "kn_simulation" / "gw170817a"

for path in (MODEL_DIR, SCRIPT_DIR, GW170817A_DATASET_DIR):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)


class TimeZeroRegressionTests(unittest.TestCase):
    def test_first_detection_uses_psfflux_snr_before_photflag_or_head_time(self):
        from data_loader import resolve_first_detection_mjd

        first_detection = resolve_first_detection_mjd(
            snr_mjd=[100.0, 101.0, 102.0],
            snr_flux=[0.0, 60.0, 0.0],
            snr_fluxerr=[1.0, 10.0, 1.0],
            photflag_mjd=[100.0, 101.0, 102.0],
            photflag=[4096, 0, 0],
            head_mjd_detect_first=102.0,
            snr_threshold=5.0,
        )

        self.assertEqual(first_detection, 101.0)

    def test_first_detection_uses_photflag_when_psfflux_snr_has_no_detection(self):
        from data_loader import resolve_first_detection_mjd

        first_detection = resolve_first_detection_mjd(
            snr_mjd=[100.0, 101.0, 102.0],
            snr_flux=[0.0, 40.0, 0.0],
            snr_fluxerr=[10.0, 10.0, 10.0],
            photflag_mjd=[100.0, 101.0, 102.0],
            photflag=[0, 4096, 0],
            head_mjd_detect_first=102.0,
            snr_threshold=5.0,
        )

        self.assertEqual(first_detection, 101.0)

    def test_first_detection_uses_head_mjd_detect_first_as_last_fallback(self):
        from data_loader import resolve_first_detection_mjd

        first_detection = resolve_first_detection_mjd(
            snr_mjd=[100.0, 101.0, 102.0],
            snr_flux=[0.0, 40.0, 0.0],
            snr_fluxerr=[10.0, 10.0, 10.0],
            photflag_mjd=[100.0, 101.0, 102.0],
            photflag=[0, 0, 0],
            head_mjd_detect_first=102.0,
            snr_threshold=5.0,
        )

        self.assertEqual(first_detection, 102.0)

    def test_gw170817_formatter_anchors_time_to_raw_psfflux_snr_not_luptitude_snr(self):
        import build_gw170817a_retrieval_h5 as build_h5

        formatted = build_h5._format_lightcurve(
            mjd=np.asarray([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64),
            fluxcal=np.asarray([10.0, 1000.0, 20.0, 30.0, 40.0], dtype=np.float64),
            fluxcalerr=np.asarray([100.0, 10.0, 100.0, 100.0, 100.0], dtype=np.float64),
            band=np.asarray(["r", "r", "r", "r", "r"]),
            photflag=np.asarray([0, 0, 0, 0, 0], dtype=np.int64),
            head_mjd_detect_first=103.0,
            fluxcal_to_psfflux_factor=1.0,
            psfflux_zp=31.4,
            lupt_b_njy=[100.0] * 6,
            min_nobs=5,
        )

        self.assertIsNotNone(formatted)
        _values, _errors, _masks, times, first_detection_mjd = formatted
        self.assertEqual(first_detection_mjd, 101.0)
        self.assertAlmostEqual(float(times[1]), 0.0, places=6)

    def test_candidate_time_helpers_require_first_detection_without_event_fallback(self):
        from scripts.train.train import (
            gather_candidate_optical_time_mjd,
            resolve_optical_candidate_time_mjd,
        )

        first_detection = torch.tensor([101.5, 203.0, 304.0], dtype=torch.float32)
        event_time = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32)
        candidate_idx = torch.tensor([2, 0], dtype=torch.long)

        resolved = resolve_optical_candidate_time_mjd(first_detection, event_time)
        gathered = gather_candidate_optical_time_mjd(first_detection, event_time, candidate_idx)
        no_fallback = gather_candidate_optical_time_mjd(None, event_time, candidate_idx)

        self.assertTrue(torch.equal(resolved, first_detection))
        self.assertTrue(torch.equal(gathered, torch.tensor([304.0, 101.5])))
        self.assertIsNone(no_fallback)

    def test_retrieval_batch_cache_requests_time_metadata(self):
        import eval_retrieval_comparison as comp

        captured_kwargs = {}

        def fake_build_loader(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return [(torch.tensor([1]),)]

        with mock.patch.object(comp, "_build_test_loader", side_effect=fake_build_loader):
            cached, used_workers = comp.build_cached_batches(
                test_data_path="dummy.h5",
                neg_data_path=None,
                neg_group="dummy/group",
                comparison_window=(-0.1, 0.2),
                batch_size=1,
                test_steps=1,
                num_workers=0,
                target_samples=None,
                nonkn_cls_base_field="zero_time_mjd_cls_base",
            )

        self.assertEqual(len(cached), 1)
        self.assertEqual(used_workers, 0)
        self.assertIs(captured_kwargs.get("return_zero_time_mjd"), True)


if __name__ == "__main__":
    unittest.main()
