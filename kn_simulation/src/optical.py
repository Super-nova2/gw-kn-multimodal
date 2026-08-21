"""Normalized raw SNANA optical payloads for inode-efficient HDF5 storage."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

REALIZATION_DTYPES = {
    "ra_deg": np.float64,
    "dec_deg": np.float64,
    "nobs": np.int64,
    "mjd_detect_first": np.float64,
    "observation_start": np.int64,
    "observation_count": np.int64,
}
OBSERVATION_DTYPES = {
    "mjd": np.float64,
    "fluxcal": np.float64,
    "fluxcalerr": np.float64,
    "band_code": np.uint8,
    "photflag": np.int32,
}
_MJD_EXPLODE_PATTERN = re.compile(r"^\s*MJD_EXPLODE:\s*([-+0-9.eE]+)", re.MULTILINE)


@dataclass(frozen=True)
class OpticalPayload:
    """One event's lossless normalized subset of SNANA HEAD/PHOT products."""

    simulation_id: int
    mjd_explode: float
    realizations: dict[str, np.ndarray]
    observations: dict[str, np.ndarray]
    checksum: str

    @property
    def realization_count(self) -> int:
        return len(self.realizations["ra_deg"])

    @property
    def observation_count(self) -> int:
        return len(self.observations["mjd"])


def _native_array(values: Any, dtype: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=dtype))


def _checksum(
    simulation_id: int,
    mjd_explode: float,
    realizations: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([simulation_id], dtype="<i8").tobytes())
    digest.update(np.asarray([mjd_explode], dtype="<f8").tobytes())
    for name, dtype in REALIZATION_DTYPES.items():
        digest.update(name.encode("ascii") + b"\0")
        digest.update(
            _native_array(realizations[name], dtype)
            .astype(np.dtype(dtype).newbyteorder("<"), copy=False)
            .tobytes()
        )
    for name, dtype in OBSERVATION_DTYPES.items():
        digest.update(name.encode("ascii") + b"\0")
        digest.update(
            _native_array(observations[name], dtype)
            .astype(np.dtype(dtype).newbyteorder("<"), copy=False)
            .tobytes()
        )
    return digest.hexdigest()


def normalize_optical_payload(
    payload: OpticalPayload | None,
    simulation_id: int,
) -> OpticalPayload | None:
    """Validate and normalize an optical payload for deterministic storage."""

    if payload is None:
        return None
    if int(payload.simulation_id) != int(simulation_id):
        raise ValueError(
            f"Optical payload belongs to simulation_id={payload.simulation_id}, "
            f"expected {simulation_id}"
        )
    missing_realizations = set(REALIZATION_DTYPES) - payload.realizations.keys()
    missing_observations = set(OBSERVATION_DTYPES) - payload.observations.keys()
    if missing_realizations or missing_observations:
        raise ValueError(
            "Optical payload is missing fields: "
            f"realizations={sorted(missing_realizations)}, "
            f"observations={sorted(missing_observations)}"
        )
    realizations = {
        name: _native_array(payload.realizations[name], dtype)
        for name, dtype in REALIZATION_DTYPES.items()
    }
    observations = {
        name: _native_array(payload.observations[name], dtype)
        for name, dtype in OBSERVATION_DTYPES.items()
    }
    realization_lengths = {len(values) for values in realizations.values()}
    observation_lengths = {len(values) for values in observations.values()}
    if len(realization_lengths) != 1 or len(observation_lengths) != 1:
        raise ValueError("Optical payload columns have inconsistent lengths")
    observation_total = next(iter(observation_lengths))
    starts = realizations["observation_start"]
    counts = realizations["observation_count"]
    if (
        np.any(starts < 0)
        or np.any(counts < 0)
        or np.any(starts + counts > observation_total)
    ):
        raise ValueError("Optical realization observation offsets are invalid")
    if not np.array_equal(realizations["nobs"], counts):
        raise ValueError("SNANA NOBS differs from normalized observation_count")
    checksum = _checksum(
        int(simulation_id), float(payload.mjd_explode), realizations, observations
    )
    if payload.checksum and payload.checksum != checksum:
        raise ValueError("Optical payload checksum mismatch")
    return OpticalPayload(
        simulation_id=int(simulation_id),
        mjd_explode=float(payload.mjd_explode),
        realizations=realizations,
        observations=observations,
        checksum=checksum,
    )


def _mjd_explode(path: Path) -> float:
    match = _MJD_EXPLODE_PATTERN.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"README does not contain MJD_EXPLODE: {path}")
    return float(match.group(1))


def _band_codes(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.uint8)
    for index, value in enumerate(values):
        if isinstance(value, (bytes, np.bytes_)):
            text = bytes(value).decode("ascii").strip()
        else:
            text = str(value).strip()
        if text.startswith("LSST-"):
            text = text.split("-", 1)[1]
        if text == "y":
            text = "Y"
        if len(text) != 1 or ord(text) > 127:
            raise ValueError(f"Unsupported SNANA BAND value: {value!r}")
        result[index] = ord(text)
    return result


def read_snana_optical(
    sndata_sim_dir: Path,
    sim_name: str,
    simulation_id: int,
) -> OpticalPayload:
    """Read and validate one event's raw SNANA products."""

    simulation_id = int(simulation_id)
    version = f"{sim_name}_{simulation_id}"
    event_dir = Path(sndata_sim_dir) / version
    head_path = event_dir / f"{version}_HEAD.FITS"
    phot_path = event_dir / f"{version}_PHOT.FITS"
    readme_path = event_dir / f"{version}.README"
    for path in (head_path, phot_path, readme_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"SNANA optical product is missing or empty: {path}"
            )

    with fits.open(head_path, memmap=True) as handle:
        head = handle[1].data
        head_names = set(head.columns.names)
        required = {"RA", "DEC", "NOBS", "PTROBS_MIN", "PTROBS_MAX"}
        missing = required - head_names
        if missing:
            raise ValueError(f"SNANA HEAD is missing columns {sorted(missing)}")
        ptrobs_min = _native_array(head["PTROBS_MIN"], np.int64)
        ptrobs_max = _native_array(head["PTROBS_MAX"], np.int64)
        starts = ptrobs_min - 1
        counts = ptrobs_max - ptrobs_min + 1
        realizations = {
            "ra_deg": _native_array(head["RA"], np.float64),
            "dec_deg": _native_array(head["DEC"], np.float64),
            "nobs": _native_array(head["NOBS"], np.int64),
            "mjd_detect_first": (
                _native_array(head["MJD_DETECT_FIRST"], np.float64)
                if "MJD_DETECT_FIRST" in head_names
                else np.full(len(head), np.nan, dtype=np.float64)
            ),
            "observation_start": starts,
            "observation_count": counts,
        }

    with fits.open(phot_path, memmap=True) as handle:
        phot = handle[1].data
        phot_names = set(phot.columns.names)
        required = {"MJD", "FLUXCAL", "FLUXCALERR", "BAND"}
        missing = required - phot_names
        if missing:
            raise ValueError(f"SNANA PHOT is missing columns {sorted(missing)}")
        observations = {
            "mjd": _native_array(phot["MJD"], np.float64),
            "fluxcal": _native_array(phot["FLUXCAL"], np.float64),
            "fluxcalerr": _native_array(phot["FLUXCALERR"], np.float64),
            "band_code": _band_codes(np.asarray(phot["BAND"])),
            "photflag": (
                _native_array(phot["PHOTFLAG"], np.int32)
                if "PHOTFLAG" in phot_names
                else np.zeros(len(phot), dtype=np.int32)
            ),
        }

    payload = OpticalPayload(
        simulation_id=simulation_id,
        mjd_explode=_mjd_explode(readme_path),
        realizations=realizations,
        observations=observations,
        checksum="",
    )
    return normalize_optical_payload(payload, simulation_id)  # type: ignore[return-value]


def optical_event_dir(sndata_sim_dir: Path, sim_name: str, simulation_id: int) -> Path:
    return Path(sndata_sim_dir) / f"{sim_name}_{int(simulation_id)}"
