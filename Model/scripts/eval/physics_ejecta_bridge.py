"""Empirical, physically interpretable GW--KN ejecta bridge.

Both a recovered-GW query and an observed light curve are represented as
weighted posteriors over log ejecta masses, inclination, and distance. Runtime
scoring only accepts frozen training references; test ejecta labels are never
part of the interface.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BAND_NAMES = ("u", "g", "r", "i", "z", "Y")
PHASE_BINS_DAYS = ((-10.0, 0.0), (0.0, 5.0), (5.0, 20.0))
GW_FEATURE_NAMES = {
    "bns": ("chirp_mass_detector", "mass_ratio", "chi_eff"),
    "nsbh": ("chirp_mass_detector", "mass_ratio", "primary_spin_z"),
}
PSI_NAMES = (
    "log10_mej_dynamic",
    "log10_mej_wind",
    "abs_costheta",
    "log10_distance_gpc",
)
CADENCE_NAMES = ("log1p_nobs", "band_fraction", "log1p_time_span_days")


def lightcurve_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for band in BAND_NAMES:
        names.extend(
            [
                f"{band}_peak_asinh_flux",
                f"{band}_peak_phase_days",
                f"{band}_pre_mean_asinh_flux",
                f"{band}_early_mean_asinh_flux",
                f"{band}_late_mean_asinh_flux",
                f"{band}_late_minus_early",
            ]
        )
    names.extend(
        f"early_color_{left}_{right}" for left, right in itertools.pairwise(BAND_NAMES)
    )
    return tuple(names)


LIGHTCURVE_FEATURE_NAMES = lightcurve_feature_names()


def decode_strings(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [
            (
                value.decode("utf-8", errors="ignore")
                if isinstance(value, (bytes, np.bytes_))
                else str(value)
            )
            for value in values
        ],
        dtype=object,
    )


def stable_hash_int(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def deterministic_event_split(
    source_types: Sequence[Any],
    simulation_ids: Sequence[int],
    *,
    seed: int,
    fit_fraction: float = 0.8,
) -> np.ndarray:
    """Return a stable event-level fit mask."""
    sources = decode_strings(source_types)
    simulations = np.asarray(simulation_ids, dtype=np.int64)
    if sources.shape != simulations.shape:
        raise ValueError("source_types and simulation_ids must have identical shape")
    if not 0.0 < float(fit_fraction) < 1.0:
        raise ValueError("fit_fraction must be in (0, 1)")
    threshold = int(float(fit_fraction) * 10_000)
    return np.asarray(
        [
            stable_hash_int(seed, source, int(simulation)) % 10_000 < threshold
            for source, simulation in zip(sources, simulations)
        ],
        dtype=bool,
    )


def derive_gw_feature_matrix(scalars: np.ndarray, source: str) -> np.ndarray:
    """Derive the three source-specific recovered GW coordinates."""
    values = np.asarray(scalars, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 7:
        raise ValueError("scalars must have shape [N, >=7]")
    m1, m2, s1, s2 = (values[:, index] for index in range(4))
    high = np.maximum(m1, m2)
    low = np.minimum(m1, m2)
    total = high + low
    chirp = np.power(high * low, 3.0 / 5.0) / np.power(total, 1.0 / 5.0)
    ratio = low / high
    source_key = str(source).lower()
    if source_key == "bns":
        third = (m1 * s1 + m2 * s2) / (m1 + m2)
    elif source_key == "nsbh":
        third = np.where(m1 >= m2, s1, s2)
    else:
        raise ValueError(f"Unsupported source {source!r}")
    result = np.column_stack([chirp, ratio, third])
    if not np.all(np.isfinite(result)):
        raise ValueError("Derived GW features contain non-finite values")
    return result


def build_psi(
    mej_dynamic: np.ndarray,
    mej_wind: np.ndarray,
    scalars: np.ndarray,
) -> np.ndarray:
    scalars = np.asarray(scalars, dtype=np.float64)
    dynamic = np.asarray(mej_dynamic, dtype=np.float64)
    wind = np.asarray(mej_wind, dtype=np.float64)
    if np.any(dynamic <= 0.0) or np.any(wind <= 0.0):
        raise ValueError("Bridge reference ejecta masses must be positive")
    result = np.column_stack(
        [
            np.log10(dynamic),
            np.log10(wind),
            np.abs(scalars[:, 4]),
            np.log10(scalars[:, 5]),
        ]
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("Bridge psi contains non-finite values")
    return result


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Expected a non-empty finite feature matrix")
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-8, scale, 1.0)
    return center, scale


def masked_robust_location_scale(
    values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    available = np.asarray(mask, dtype=bool)
    if values.shape != available.shape or values.ndim != 2:
        raise ValueError("values and mask must be matching two-dimensional arrays")
    center = np.zeros(values.shape[1], dtype=np.float64)
    scale = np.ones(values.shape[1], dtype=np.float64)
    for index in range(values.shape[1]):
        observed = values[available[:, index], index]
        if observed.size:
            center[index] = np.median(observed)
            q25, q75 = np.quantile(observed, [0.25, 0.75])
            if q75 - q25 > 1e-8:
                scale[index] = q75 - q25
    return center, scale


def luptitude_to_flux_sigma(
    values: np.ndarray,
    errors: np.ndarray,
    *,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the repository luptitude representation into nJy."""
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if values.shape != errors.shape or values.shape[-1] != len(lupt_b_njy):
        raise ValueError("Invalid luptitude values/errors/band softening shape")
    b = np.asarray(lupt_b_njy, dtype=np.float64).reshape(
        (1,) * (values.ndim - 1) + (values.shape[-1],)
    )
    factor = 2.5 / math.log(10.0)
    x = (float(psfflux_zp) - values) / factor - np.log(b)
    flux = 2.0 * b * np.sinh(x)
    sigma = np.abs(errors) * np.sqrt(flux**2 + (2.0 * b) ** 2) / factor
    return flux, sigma


def _weighted_mean(values: np.ndarray, sigma: np.ndarray) -> float:
    weights = 1.0 / np.maximum(np.asarray(sigma, dtype=np.float64) ** 2, 1e-12)
    if weights.size > 1:
        weights = np.minimum(weights, np.quantile(weights, 0.95))
    return float(np.sum(weights * values) / np.sum(weights))


def extract_lightcurve_features(
    times: np.ndarray,
    values: np.ndarray,
    masks: np.ndarray,
    errors: np.ndarray,
    *,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract masked physical photometry features and cadence descriptors."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float64) > 0
    errors = np.asarray(errors, dtype=np.float64)
    if values.ndim != 3 or values.shape != masks.shape or values.shape != errors.shape:
        raise ValueError("values, masks and errors must have shape [N, T, 6]")
    if times.shape != values.shape[:2] or values.shape[2] != len(BAND_NAMES):
        raise ValueError(
            "times must have shape [N, T] and photometry must have 6 bands"
        )
    flux, sigma = luptitude_to_flux_sigma(
        values,
        errors,
        psfflux_zp=psfflux_zp,
        lupt_b_njy=lupt_b_njy,
    )
    b = np.asarray(lupt_b_njy, dtype=np.float64)
    n_rows = values.shape[0]
    features = np.zeros((n_rows, len(LIGHTCURVE_FEATURE_NAMES)), dtype=np.float64)
    available = np.zeros_like(features, dtype=bool)
    cadence = np.zeros((n_rows, len(CADENCE_NAMES)), dtype=np.float64)
    for row in range(n_rows):
        valid_slots = np.any(masks[row], axis=1)
        n_obs = int(np.count_nonzero(valid_slots))
        band_hits = np.any(masks[row], axis=0)
        span = (
            float(np.max(times[row, valid_slots]) - np.min(times[row, valid_slots]))
            * 100.0
            if n_obs
            else 0.0
        )
        cadence[row] = [
            math.log1p(n_obs),
            float(np.mean(band_hits)),
            math.log1p(max(span, 0.0)),
        ]
        early_values = np.zeros(len(BAND_NAMES), dtype=np.float64)
        early_ok = np.zeros(len(BAND_NAMES), dtype=bool)
        for band in range(len(BAND_NAMES)):
            valid = masks[row, :, band] & np.isfinite(flux[row, :, band])
            offset = band * 6
            if not np.any(valid):
                continue
            band_flux = flux[row, valid, band]
            band_sigma = sigma[row, valid, band]
            band_times_days = times[row, valid] * 100.0
            peak_index = int(np.argmax(band_flux))
            features[row, offset] = np.arcsinh(band_flux[peak_index] / (2.0 * b[band]))
            features[row, offset + 1] = band_times_days[peak_index]
            available[row, offset : offset + 2] = True
            means: list[float | None] = []
            for bin_index, (lower, upper) in enumerate(PHASE_BINS_DAYS):
                selected = (band_times_days >= lower) & (
                    band_times_days <= upper
                    if bin_index == len(PHASE_BINS_DAYS) - 1
                    else band_times_days < upper
                )
                if np.any(selected):
                    mean_flux = _weighted_mean(
                        band_flux[selected], band_sigma[selected]
                    )
                    transformed = float(np.arcsinh(mean_flux / (2.0 * b[band])))
                    features[row, offset + 2 + bin_index] = transformed
                    available[row, offset + 2 + bin_index] = True
                    means.append(transformed)
                else:
                    means.append(None)
            if means[1] is not None:
                early_values[band] = float(means[1])
                early_ok[band] = True
            if means[1] is not None and means[2] is not None:
                features[row, offset + 5] = float(means[2] - means[1])
                available[row, offset + 5] = True
        color_offset = 6 * len(BAND_NAMES)
        for color in range(len(BAND_NAMES) - 1):
            if early_ok[color] and early_ok[color + 1]:
                features[row, color_offset + color] = (
                    early_values[color] - early_values[color + 1]
                )
                available[row, color_offset + color] = True
    return features, available, cadence


def select_curves_per_event(
    parent_gw_idx: np.ndarray,
    allowed_events: np.ndarray,
    *,
    seed: int,
    max_curves_per_event: int,
) -> np.ndarray:
    parent = np.asarray(parent_gw_idx, dtype=np.int64)
    allowed = set(np.asarray(allowed_events, dtype=np.int64).tolist())
    by_event: dict[int, list[int]] = {}
    for optical_index, event in enumerate(parent.tolist()):
        if event in allowed:
            by_event.setdefault(event, []).append(optical_index)
    selected: list[int] = []
    for event in sorted(by_event):
        ranked = sorted(
            by_event[event],
            key=lambda index: (stable_hash_int(seed, index), index),
        )
        selected.extend(ranked[: int(max_curves_per_event)])
    return np.asarray(sorted(selected), dtype=np.int64)


def masked_feature_distances(
    query_values: np.ndarray,
    query_mask: np.ndarray,
    reference_values: np.ndarray,
    reference_mask: np.ndarray,
) -> np.ndarray:
    q = np.asarray(query_values, dtype=np.float64).reshape(-1)
    qm = np.asarray(query_mask, dtype=bool).reshape(-1)
    refs = np.asarray(reference_values, dtype=np.float64)
    rm = np.asarray(reference_mask, dtype=bool)
    if refs.shape != rm.shape or refs.shape[1] != q.size or qm.size != q.size:
        raise ValueError("Masked feature distance shapes are inconsistent")
    common = rm & qm[None, :]
    count = common.sum(axis=1)
    squared = np.where(common, (refs - q[None, :]) ** 2, 0.0).sum(axis=1)
    result = np.full(len(refs), np.inf, dtype=np.float64)
    valid = count > 0
    result[valid] = np.sqrt(squared[valid] / count[valid])
    return result


def cadence_candidate_mask(
    query: np.ndarray,
    references: np.ndarray,
    *,
    max_abs: float,
    l1_max: float,
) -> np.ndarray:
    differences = np.abs(
        np.asarray(references, dtype=np.float64)
        - np.asarray(query, dtype=np.float64)[None, :]
    )
    return (np.max(differences, axis=1) <= float(max_abs)) & (
        np.sum(differences, axis=1) <= float(l1_max)
    )


@dataclass(frozen=True)
class NeighborPosterior:
    indices: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    fallback_level: int = 0


def select_unique_neighbors(
    distances: np.ndarray,
    parent_ids: np.ndarray,
    *,
    k: int,
    alpha: float,
    eligible: np.ndarray | None = None,
) -> NeighborPosterior:
    distances = np.asarray(distances, dtype=np.float64)
    parents = np.asarray(parent_ids, dtype=np.int64)
    if distances.shape != parents.shape or int(k) < 2:
        raise ValueError("Invalid neighbor inputs")
    allowed = np.isfinite(distances)
    if eligible is not None:
        allowed &= np.asarray(eligible, dtype=bool)
    order = np.argsort(np.where(allowed, distances, np.inf), kind="stable")
    chosen: list[int] = []
    seen: set[int] = set()
    for index in order.tolist():
        if not allowed[index]:
            break
        parent = int(parents[index])
        if parent in seen:
            continue
        chosen.append(index)
        seen.add(parent)
        if len(chosen) == int(k):
            break
    if len(chosen) < int(k):
        raise ValueError(f"Only {len(chosen)} unique eligible neighbors; need {k}")
    indices = np.asarray(chosen, dtype=np.int64)
    selected_distances = distances[indices]
    bandwidth = float(alpha) * max(float(selected_distances[-1]), 1e-8)
    weights = np.exp(-0.5 * (selected_distances / bandwidth) ** 2)
    weights /= np.sum(weights)
    return NeighborPosterior(indices, selected_distances, weights)


def weighted_gaussian(
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    covariance_floor: float,
    extra_diagonal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if samples.ndim != 2 or weights.shape != (len(samples),):
        raise ValueError("Invalid weighted posterior shapes")
    weights = weights / np.sum(weights)
    mean = np.sum(weights[:, None] * samples, axis=0)
    centered = samples - mean
    covariance = (centered * weights[:, None]).T @ centered
    effective = 1.0 - float(np.sum(weights**2))
    if effective > 1e-8:
        covariance /= effective
    covariance += np.eye(samples.shape[1]) * float(covariance_floor) ** 2
    if extra_diagonal is not None:
        covariance += np.diag(np.asarray(extra_diagonal, dtype=np.float64))
    return mean, covariance


def bhattacharyya_score(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
    *,
    jitter: float = 1e-10,
) -> float:
    """Return negative Gaussian Bhattacharyya distance."""
    mean_a = np.asarray(mean_a, dtype=np.float64)
    mean_b = np.asarray(mean_b, dtype=np.float64)
    cov_a = np.asarray(covariance_a, dtype=np.float64)
    cov_b = np.asarray(covariance_b, dtype=np.float64)
    midpoint = 0.5 * (cov_a + cov_b)

    def factor(matrix: np.ndarray) -> tuple[np.ndarray, float]:
        identity = np.eye(matrix.shape[0])
        for multiplier in (1.0, 10.0):
            try:
                chol = np.linalg.cholesky(matrix + identity * jitter * multiplier)
                return chol, 2.0 * float(np.log(np.diag(chol)).sum())
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError("Bridge covariance is not positive definite")

    chol_mid, logdet_mid = factor(midpoint)
    _, logdet_a = factor(cov_a)
    _, logdet_b = factor(cov_b)
    delta = mean_a - mean_b
    solved = np.linalg.solve(chol_mid, delta)
    distance = 0.125 * float(solved @ solved) + 0.5 * (
        logdet_mid - 0.5 * (logdet_a + logdet_b)
    )
    return -float(max(distance, 0.0))


class PhysicsEjectaBridge:
    """Frozen source-specific empirical bridge artifact."""

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        manifest_path = self.artifact_dir / "bridge_fit_manifest.json"
        arrays_path = self.artifact_dir / "bridge_reference.npz"
        if not manifest_path.is_file() or not arrays_path.is_file():
            raise FileNotFoundError(
                f"Incomplete Physics Ejecta Bridge artifact: {self.artifact_dir}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("status") != "complete":
            raise ValueError("Physics Ejecta Bridge artifact is not complete")
        self.arrays = np.load(arrays_path, allow_pickle=False)

    def _key(self, source: str, name: str) -> np.ndarray:
        return np.asarray(self.arrays[f"{str(source).lower()}__{name}"])

    def hyperparameters(self, source: str, side: str) -> Mapping[str, Any]:
        return self.manifest["sources"][str(source).lower()][f"{side}_hyperparameters"]

    def gw_posterior(
        self, scalars: np.ndarray, source: str
    ) -> tuple[np.ndarray, np.ndarray, NeighborPosterior]:
        query_scalars = np.asarray(scalars, dtype=np.float64).reshape(1, -1)
        query = derive_gw_feature_matrix(query_scalars, source)[0]
        center = self._key(source, "gw_center")
        scale = self._key(source, "gw_scale")
        references = self._key(source, "gw_features")
        distances = np.linalg.norm((query - center) / scale - references, axis=1)
        hp = self.hyperparameters(source, "gw")
        neighbors = select_unique_neighbors(
            distances,
            self._key(source, "gw_parent_ids"),
            k=int(hp["k"]),
            alpha=float(hp["alpha"]),
        )
        samples = self._key(source, "gw_psi")[neighbors.indices].copy()
        psi_center = self._key(source, "psi_center")
        psi_scale = self._key(source, "psi_scale")
        samples[:, 2] = (abs(float(query_scalars[0, 4])) - psi_center[2]) / psi_scale[2]
        samples[:, 3] = (
            math.log10(float(query_scalars[0, 5])) - psi_center[3]
        ) / psi_scale[3]
        fractional = float(query_scalars[0, 6]) / float(query_scalars[0, 5])
        sigma_log_distance = fractional / math.log(10.0) / psi_scale[3]
        extra = np.zeros(4, dtype=np.float64)
        extra[3] = sigma_log_distance**2
        mean, covariance = weighted_gaussian(
            samples,
            neighbors.weights,
            covariance_floor=float(hp["covariance_floor"]),
            extra_diagonal=extra,
        )
        return mean, covariance, neighbors

    def lc_posterior(
        self,
        feature_values: np.ndarray,
        feature_mask: np.ndarray,
        cadence: np.ndarray,
        source: str,
    ) -> tuple[np.ndarray, np.ndarray, NeighborPosterior]:
        center = self._key(source, "lc_center")
        scale = self._key(source, "lc_scale")
        query = np.clip((np.asarray(feature_values) - center) / scale, -8.0, 8.0)
        references = self._key(source, "lc_features")
        reference_mask = self._key(source, "lc_feature_mask").astype(bool)
        distances = masked_feature_distances(
            query, feature_mask, references, reference_mask
        )
        cadence_center = self._key(source, "cadence_center")
        cadence_scale = self._key(source, "cadence_scale")
        query_cadence = (np.asarray(cadence) - cadence_center) / cadence_scale
        reference_cadence = self._key(source, "lc_cadence")
        hp = self.hyperparameters(source, "lc")
        required = 8 * int(hp["k"])
        fallback = 0
        eligible = cadence_candidate_mask(
            query_cadence, reference_cadence, max_abs=1.0, l1_max=3.0
        )
        if np.count_nonzero(eligible) < required:
            fallback = 1
            eligible = cadence_candidate_mask(
                query_cadence,
                reference_cadence,
                max_abs=1.5,
                l1_max=4.5,
            )
        if np.count_nonzero(eligible) < required:
            fallback = 2
            eligible = np.ones(len(reference_cadence), dtype=bool)
        neighbors = select_unique_neighbors(
            distances,
            self._key(source, "lc_parent_ids"),
            k=int(hp["k"]),
            alpha=float(hp["alpha"]),
            eligible=eligible,
        )
        neighbors = NeighborPosterior(
            neighbors.indices,
            neighbors.distances,
            neighbors.weights,
            fallback,
        )
        mean, covariance = weighted_gaussian(
            self._key(source, "lc_psi")[neighbors.indices],
            neighbors.weights,
            covariance_floor=float(hp["covariance_floor"]),
        )
        return mean, covariance, neighbors

    def lc_posteriors_batch(
        self,
        feature_values: np.ndarray,
        feature_masks: np.ndarray,
        cadences: np.ndarray,
        source: str,
        *,
        device: str = "cuda",
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray, list[NeighborPosterior]]:
        """Compute many light-curve posteriors with batched masked distances."""
        import torch

        center = self._key(source, "lc_center")
        scale = self._key(source, "lc_scale")
        queries = np.clip((np.asarray(feature_values) - center) / scale, -8.0, 8.0)
        query_masks = np.asarray(feature_masks, dtype=bool)
        cadence_center = self._key(source, "cadence_center")
        cadence_scale = self._key(source, "cadence_scale")
        query_cadence = (np.asarray(cadences) - cadence_center) / cadence_scale
        references = self._key(source, "lc_features").astype(np.float32)
        reference_masks = self._key(source, "lc_feature_mask").astype(bool)
        reference_cadence = self._key(source, "lc_cadence").astype(np.float32)
        parents = self._key(source, "lc_parent_ids").astype(np.int64)
        psi = self._key(source, "lc_psi")
        hp = self.hyperparameters(source, "lc")
        k = int(hp["k"])
        required = 8 * k
        torch_device = torch.device(device)
        rv = torch.as_tensor(references, device=torch_device)
        rm = torch.as_tensor(reference_masks.astype(np.float32), device=torch_device)
        rvm = rv * rm
        rv2m = rv.square() * rm
        rc = torch.as_tensor(reference_cadence, device=torch_device)
        means: list[np.ndarray] = []
        covariances: list[np.ndarray] = []
        all_neighbors: list[NeighborPosterior] = []
        for start in range(0, len(queries), int(batch_size)):
            stop = min(start + int(batch_size), len(queries))
            qv = torch.as_tensor(
                queries[start:stop], dtype=torch.float32, device=torch_device
            )
            qm = torch.as_tensor(
                query_masks[start:stop].astype(np.float32), device=torch_device
            )
            qvm = qv * qm
            common = qm @ rm.T
            squared = (qv.square() * qm) @ rm.T + qm @ rv2m.T - 2.0 * (qvm @ rvm.T)
            distance = torch.sqrt(
                torch.clamp(squared, min=0.0) / torch.clamp(common, min=1.0)
            )
            distance = torch.where(
                common > 0, distance, torch.full_like(distance, float("inf"))
            )
            qc = torch.as_tensor(
                query_cadence[start:stop], dtype=torch.float32, device=torch_device
            )
            cadence_diff = torch.abs(qc[:, None, :] - rc[None, :, :])
            strict = (cadence_diff.amax(dim=2) <= 1.0) & (
                cadence_diff.sum(dim=2) <= 3.0
            )
            relaxed = (cadence_diff.amax(dim=2) <= 1.5) & (
                cadence_diff.sum(dim=2) <= 4.5
            )
            strict_count = strict.sum(dim=1)
            relaxed_count = relaxed.sum(dim=1)
            eligible = torch.ones_like(strict)
            fallback = np.full(stop - start, 2, dtype=np.int64)
            strict_rows = strict_count >= required
            relaxed_rows = (~strict_rows) & (relaxed_count >= required)
            eligible[strict_rows] = strict[strict_rows]
            eligible[relaxed_rows] = relaxed[relaxed_rows]
            fallback[np.asarray(strict_rows.cpu())] = 0
            fallback[np.asarray(relaxed_rows.cpu())] = 1
            distance = torch.where(
                eligible, distance, torch.full_like(distance, float("inf"))
            )
            search_k = min(max(required * 2, 2048), distance.shape[1])
            top_distance, top_index = torch.topk(
                distance, k=search_k, largest=False, sorted=True
            )
            top_distance_np = np.asarray(top_distance.cpu(), dtype=np.float64)
            top_index_np = np.asarray(top_index.cpu(), dtype=np.int64)
            for local in range(stop - start):
                chosen: list[int] = []
                seen: set[int] = set()
                for ref_index, ref_distance in zip(
                    top_index_np[local], top_distance_np[local]
                ):
                    if not np.isfinite(ref_distance):
                        break
                    parent = int(parents[ref_index])
                    if parent in seen:
                        continue
                    chosen.append(int(ref_index))
                    seen.add(parent)
                    if len(chosen) == k:
                        break
                if len(chosen) < k:
                    raise ValueError(
                        f"Only {len(chosen)} unique batched LC neighbors; need {k}"
                    )
                indices = np.asarray(chosen, dtype=np.int64)
                selected_distances = np.asarray(
                    [
                        top_distance_np[
                            local, np.flatnonzero(top_index_np[local] == index)[0]
                        ]
                        for index in indices
                    ],
                    dtype=np.float64,
                )
                bandwidth = float(hp["alpha"]) * max(
                    float(selected_distances[-1]), 1e-8
                )
                weights = np.exp(-0.5 * (selected_distances / bandwidth) ** 2)
                weights /= weights.sum()
                neighbors = NeighborPosterior(
                    indices, selected_distances, weights, int(fallback[local])
                )
                mean, covariance = weighted_gaussian(
                    psi[indices],
                    weights,
                    covariance_floor=float(hp["covariance_floor"]),
                )
                means.append(mean)
                covariances.append(covariance)
                all_neighbors.append(neighbors)
        return np.asarray(means), np.asarray(covariances), all_neighbors
