import sys
from pathlib import Path

import h5py
import numpy as np

OPTICAL_ONLY_DIR = Path(__file__).resolve().parents[2] / "optical_only"
if str(OPTICAL_ONLY_DIR) not in sys.path:
    sys.path.insert(0, str(OPTICAL_ONLY_DIR))

from scripts.data import create_datasets as builder

NUM_BANDS = 6
MAX_LC_LENGTH = 200
PSFFLUX_ZP = 31.4
LUPT_B = np.array(
    [
        200.0,
        72.6156109540202,
        95.7260184645276,
        182.40216787118175,
        347.56016574987456,
        1049.6149204995424,
    ],
    dtype=np.float64,
)
ASINH_MAG_FACTOR = 2.5 / np.log(10.0)


def _lupt_from_flux(flux, fluxerr, band):
    b = LUPT_B[band]
    value = PSFFLUX_ZP - ASINH_MAG_FACTOR * (
        np.arcsinh(flux / (2.0 * b)) + np.log(b)
    )
    error = ASINH_MAG_FACTOR * fluxerr / np.sqrt(flux * flux + (2.0 * b) ** 2)
    return value, error


def _row(flux, fluxerr, band=0, n_slots=4):
    times = np.zeros(MAX_LC_LENGTH, dtype=np.float32)
    values = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    errors = np.zeros_like(values)
    masks = np.zeros_like(values)
    for slot in range(n_slots):
        v, e = _lupt_from_flux(flux, fluxerr, band)
        values[slot, band] = v
        errors[slot, band] = e
        masks[slot, band] = 1.0
        times[slot] = float(slot - 1) / 100.0
    return values, errors, masks, times, np.array([10.0, -20.0], dtype=np.float32)


def _write_albef_h5(path):
    with h5py.File(path, "w") as f:
        gw = f.create_group("events/gw_data")
        gw.create_dataset("has_kn", data=np.array([1, 1, 0], dtype=np.int32))
        gw.create_dataset("source_type", data=np.array([b"bns", b"nsbh", b"bns"]))
        opt = f.create_group("events/optical_data")
        rows = [
            _row(500.0, 10.0, band=0, n_slots=4),  # SNR=50, parent 0
            _row(10.0, 10.0, band=1, n_slots=3),   # SNR=1, parent 0
            _row(800.0, 20.0, band=2, n_slots=5),  # SNR=40, parent 1
            _row(900.0, 10.0, band=3, n_slots=6),  # SNR=90, parent 2 (has_kn=0)
        ]
        opt.create_dataset("values", data=np.stack([r[0] for r in rows]), dtype=np.float32)
        opt.create_dataset("errors", data=np.stack([r[1] for r in rows]), dtype=np.float32)
        opt.create_dataset("masks", data=np.stack([r[2] for r in rows]), dtype=np.float32)
        opt.create_dataset("times", data=np.stack([r[3] for r in rows]), dtype=np.float32)
        opt.create_dataset("coordinates", data=np.stack([r[4] for r in rows]), dtype=np.float32)
        opt.create_dataset(
            "parent_gw_idx", data=np.array([0, 0, 1, 2], dtype=np.int32)
        )
        opt.create_dataset(
            "zero_time_mjd_base", data=np.full(len(rows), 60000.0, dtype=np.float64)
        )
        attrs = {
            "psfflux_zp": PSFFLUX_ZP,
            "lupt_b_njy": LUPT_B,
            "lupt_m5_mag": [23.9, 25.0, 24.7, 24.0, 23.3, 22.1],
            "lupt_f5sigma_njy": [
                1000.0,
                363.078054770101,
                478.630092322638,
                912.0108393559087,
                1737.8008287493728,
                5248.0746024977125,
            ],
            "fluxcal_zp": 27.5,
            "fluxcal_to_psfflux_factor": 36.3078054770101,
            "lupt_k": 1.0,
            "first_detection_policy": "psfflux_snr5_then_photflag_then_head_mjd_detect_first",
            "time_scale_divisor_days": 100.0,
            "photometry_representation": "luptitude",
            "flux_input_column": "FLUXCAL",
            "fluxerr_input_column": "FLUXCALERR",
            "lupt_band_order": "u,g,r,i,z,Y",
            "values_semantics": "luptitude",
            "errors_semantics": "luptitude_sigma",
            "lightcurve_merge_window_hours": 2.0,
            "lightcurve_merge_mode": "inverse_variance_weighted_same_band_psfflux",
            "lightcurve_merge_flux_domain": "psfflux",
            "min_nobs_stage": "post_merge",
        }
        for key, value in attrs.items():
            f.attrs[key] = value


def test_luptitude_to_psfflux_roundtrip():
    flux = np.array([[500.0, 10.0]], dtype=np.float64)
    ferr = np.array([[10.0, 10.0]], dtype=np.float64)
    bands = np.array([[0, 1]], dtype=np.int64)
    values, errors = _lupt_from_flux(flux, ferr, bands)
    flux2, ferr2 = builder.luptitude_to_psfflux(
        values,
        errors,
        bands,
        psfflux_zp=PSFFLUX_ZP,
        lupt_b_njy=LUPT_B,
    )
    np.testing.assert_allclose(flux2, flux, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(ferr2, ferr, rtol=1e-10, atol=1e-8)


def test_albef_import_filters_negatives_and_computes_detection_meta(tmp_path):
    src = tmp_path / "albef_train.h5"
    _write_albef_h5(src)
    out = tmp_path / "combined_dataset_train.h5"
    builder.create_positive_h5_from_albef(
        out,
        src,
        snr_threshold=5.0,
        write_meta_features=True,
        buffer_limit=2,
        enforce_positive_only=True,
        archive_existing=False,
    )

    with h5py.File(out, "r") as f:
        g = f["events/optical_data"]
        assert g["values"].shape == (3, MAX_LC_LENGTH, NUM_BANDS)
        np.testing.assert_array_equal(g["parent_gw_idx"][:], np.array([0, 0, 1]))
        slot = g["slot_is_detection"][:]
        np.testing.assert_array_equal(slot[0, :4], np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(slot[0, 4:], np.zeros(MAX_LC_LENGTH - 4, dtype=np.float32))
        assert not np.any(slot[1, :3])
        np.testing.assert_array_equal(slot[2, :5], np.ones(5, dtype=np.float32))

        meta_n_obs = g["meta_n_obs"][:]
        meta_n_det_snr5 = g["meta_n_det_snr5"][:]
        np.testing.assert_array_equal(meta_n_obs, np.array([4, 3, 5]))
        np.testing.assert_array_equal(meta_n_det_snr5, np.array([4, 0, 5]))

        assert f.attrs["n_total_optical"] == 3
        assert f.attrs["n_total_gw"] == 2
        assert f.attrs["source_albef_n_total_gw"] == 3
        assert f.attrs["source_albef_n_total_optical"] == 4
        assert f.attrs["bns_events_written"] == 1
        assert f.attrs["nsbh_events_written"] == 1
        assert f.attrs["slot_is_detection_semantics"] == "derived_snr_gt_5_any_band"
        assert f.attrs["first_detection_rule"] == "psfflux_snr5_then_photflag_then_head_mjd_detect_first"
        np.testing.assert_allclose(f.attrs["lupt_b_njy"], LUPT_B)


def test_albef_import_keeps_negatives_when_disabled(tmp_path):
    src = tmp_path / "albef_train.h5"
    _write_albef_h5(src)
    out = tmp_path / "combined_dataset_train_all.h5"
    builder.create_positive_h5_from_albef(
        out,
        src,
        snr_threshold=5.0,
        write_meta_features=False,
        buffer_limit=2,
        enforce_positive_only=False,
        archive_existing=False,
    )
    with h5py.File(out, "r") as f:
        g = f["events/optical_data"]
        assert g["values"].shape[0] == 4
        np.testing.assert_array_equal(g["parent_gw_idx"][:], np.array([0, 0, 1, 2]))
        assert f.attrs["n_total_gw"] == 3
        assert f.attrs["n_total_optical"] == 4


def test_albef_import_archives_existing_output(tmp_path):
    src = tmp_path / "albef_train.h5"
    _write_albef_h5(src)
    out = tmp_path / "combined_dataset_train.h5"
    out.write_bytes(b"old-bytes")
    builder.create_positive_h5_from_albef(
        out,
        src,
        snr_threshold=5.0,
        write_meta_features=False,
        buffer_limit=4,
        enforce_positive_only=True,
        archive_existing=True,
    )
    assert out.stat().st_size > 4
    archive = out.parent / "archive" / f"{out.name}.pre_albef_import"
    assert archive.read_bytes() == b"old-bytes"
