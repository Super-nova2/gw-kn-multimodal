"""Sky-coordinate sampling helpers for SNANA kilonova simulations."""

from __future__ import annotations

import healpy as hp
import numpy as np
from ligo.skymap.distance import conditional_ppf
from ligo.skymap.moc import rasterize
from ligo.skymap.postprocess import find_greedy_credible_levels


def _validate_nside(nside: int) -> int:
    nside = int(nside)
    if nside <= 0 or nside & (nside - 1):
        raise ValueError("nside must be a positive power of two")
    return nside


def _rasterized_posterior_data(mocmap, nside: int):
    """Return the rasterized map, pixel probabilities, and credible levels."""
    nside = _validate_nside(nside)
    order = int(np.log2(nside))
    raster = rasterize(np.asarray(mocmap), order=order)
    probability_density = np.asarray(raster["PROBDENSITY"], dtype=float)
    pixel_area = hp.nside2pixarea(nside)
    probability = probability_density * pixel_area

    valid = np.isfinite(probability) & np.isfinite(probability_density)
    valid &= probability > 0
    if not np.any(valid):
        raise ValueError("Sky map has no finite positive posterior probability")

    probability = np.where(valid, probability, 0.0)
    probability /= probability.sum()
    credible_level = find_greedy_credible_levels(
        probability,
        np.where(valid, probability_density, -np.inf),
    )
    return raster, probability, credible_level


def rasterized_posterior(mocmap, nside: int = 256):
    """Return fixed-order pixel probabilities and credible levels."""
    _, probability, credible_level = _rasterized_posterior_data(mocmap, nside)
    return probability, credible_level


def sample_posterior_coordinates(
    mocmap,
    n_samples: int,
    *,
    level: float = 0.9,
    nside: int = 256,
    seed: int | None = None,
):
    """Sample fixed-order HEALPix pixel centers from a 2D sky posterior."""
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    if not 0 < level <= 1:
        raise ValueError("level must be in (0, 1]")
    if n_samples == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty

    nside = _validate_nside(nside)
    probability, credible_level = rasterized_posterior(mocmap, nside=nside)
    eligible = np.flatnonzero((credible_level <= level) & (probability > 0))
    if len(eligible) == 0:
        raise ValueError(f"Sky map has no pixels inside the {level:g} credible region")

    conditional_probability = probability[eligible]
    conditional_probability /= conditional_probability.sum()
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        eligible,
        size=int(n_samples),
        replace=True,
        p=conditional_probability,
    )
    theta, phi = hp.pix2ang(nside, selected, nest=True)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)
    return ra, dec, probability[selected]


def sample_posterior_3d(
    mocmap,
    n_samples: int,
    *,
    level: float = 0.9,
    nside: int = 256,
    seed: int | None = None,
):
    """Sample sky position and conditional distance from a 3D GW posterior."""
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    if not 0 < level <= 1:
        raise ValueError("level must be in (0, 1]")
    if n_samples == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty, empty

    nside = _validate_nside(nside)
    raster, probability, credible_level = _rasterized_posterior_data(mocmap, nside)
    names = set(raster.dtype.names or ())
    required = {"DISTMU", "DISTSIGMA", "DISTNORM"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            f"Sky map is missing 3D distance columns: {', '.join(missing)}"
        )

    distmu = np.asarray(raster["DISTMU"], dtype=float)
    distsigma = np.asarray(raster["DISTSIGMA"], dtype=float)
    distnorm = np.asarray(raster["DISTNORM"], dtype=float)
    valid_distance = (
        np.isfinite(distmu)
        & np.isfinite(distsigma)
        & np.isfinite(distnorm)
        & (distmu > 0)
        & (distsigma > 0)
        & (distnorm > 0)
    )
    eligible = np.flatnonzero(
        (credible_level <= level) & (probability > 0) & valid_distance
    )
    if len(eligible) == 0:
        raise ValueError(
            f"Sky map has no valid 3D pixels inside the {level:g} credible region"
        )

    conditional_probability = probability[eligible]
    conditional_probability /= conditional_probability.sum()
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        eligible,
        size=int(n_samples),
        replace=True,
        p=conditional_probability,
    )
    theta, phi = hp.pix2ang(nside, selected, nest=True)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)

    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    quantile = rng.uniform(lower, upper, size=int(n_samples))
    distance = conditional_ppf(
        quantile,
        distmu[selected],
        distsigma[selected],
        distnorm[selected],
    )
    if not np.all(np.isfinite(distance) & (distance > 0)):
        raise ValueError("Conditional distance sampling produced invalid distances")
    return ra, dec, distance, probability[selected]


def build_posterior_coordinate_samples(
    mocmap,
    *,
    samples_per_event: int,
    level: float = 0.9,
    nside: int = 256,
    seed: int | None = None,
):
    """Build posterior-only 3D coordinate samples for a GW event."""
    if samples_per_event < 1:
        raise ValueError("samples_per_event must be >= 1")

    ra, dec, distance, probability = sample_posterior_3d(
        mocmap,
        samples_per_event,
        level=level,
        nside=nside,
        seed=seed,
    )
    is_true_position = np.zeros(samples_per_event, dtype=bool)
    return ra, dec, distance, probability, is_true_position


def build_test_coordinate_samples(
    mocmap,
    *,
    true_ra: float,
    true_dec: float,
    true_distance_mpc: float,
    samples_per_event: int = 64,
    level: float = 0.9,
    nside: int = 256,
    seed: int | None = None,
):
    """Build one true position plus posterior samples for a GW event."""
    if samples_per_event < 1:
        raise ValueError("samples_per_event must be >= 1")
    if not np.isfinite(true_distance_mpc) or true_distance_mpc <= 0:
        raise ValueError("true_distance_mpc must be finite and positive")

    posterior_count = samples_per_event - 1
    (
        posterior_ra,
        posterior_dec,
        posterior_distance,
        posterior_probability,
    ) = sample_posterior_3d(
        mocmap,
        posterior_count,
        level=level,
        nside=nside,
        seed=seed,
    )
    ra = np.concatenate(([float(true_ra)], posterior_ra))
    dec = np.concatenate(([float(true_dec)], posterior_dec))
    distance = np.concatenate(([float(true_distance_mpc)], posterior_distance))
    probability = np.concatenate(([np.nan], posterior_probability))
    is_true_position = np.zeros(samples_per_event, dtype=bool)
    is_true_position[0] = True
    return ra, dec, distance, probability, is_true_position
