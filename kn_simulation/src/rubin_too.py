"""Rubin Target-of-Opportunity helpers for GW-associated KN simulations.

The cadence implemented here follows the Rubin 2024 ToO workshop report and
keeps production dataset generation self-contained in this repository.
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from astropy.utils import iers
from ligo.skymap.moc import uniq2nest, uniq2pixarea
from ligo.skymap.postprocess import find_greedy_credible_levels
from opsimsummaryv2 import utils as opsim_utils
from opsimsummaryv2.sim_io import SNANA_Simlib
from sklearn.neighbors import BallTree, KDTree
from sky_sampling import rasterized_posterior

SECONDS_PER_DAY = 86400.0
SIMLIB_PIXEL_SIZE_ARCSEC = 0.2

# Batch jobs must not attempt to download IERS tables.
iers.conf.auto_download = False
iers.conf.auto_max_age = None


@dataclass(frozen=True)
class ToODecision:
    """The optical-observation branch selected for one GW event."""

    mode: str
    strategy: str | None
    reason: str


@dataclass
class ToOSchedule:
    """An event-level tiled schedule and its audit information."""

    observations: pd.DataFrame
    visibility: pd.DataFrame
    busy_intervals: list[tuple[float, float]]
    scan_intervals: list[dict[str, Any]]
    n_tiles: int
    night0_branch: str | None
    probability_coverage_by_night: dict[str, float]


def load_too_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the shared Rubin ToO YAML configuration."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "version",
        "source_url",
        "credible_level",
        "snr_too",
        "effective_fov_deg2",
        "visit_overhead_sec",
        "filter_change_sec",
        "number_of_nights",
        "visibility_grid_minutes",
        "observatory",
        "conditions",
        "strategies",
    }
    missing = sorted(required.difference(config or {}))
    if missing:
        raise ValueError(f"ToO configuration is missing keys: {missing}")
    if not 0 < float(config["credible_level"]) <= 1:
        raise ValueError("credible_level must be in (0, 1]")
    for name in ("gold_five_filter", "gold", "silver"):
        if name not in config["strategies"]:
            raise ValueError(f"ToO configuration is missing strategy {name}")
    return config


def classify_event(
    snr: float,
    area90_deg2: float,
    config: Mapping[str, Any],
) -> ToODecision:
    """Classify one event into baseline or baseline-plus-ToO."""

    snr = float(snr)
    area90_deg2 = float(area90_deg2)
    if not np.isfinite(snr):
        raise ValueError("GW SNR must be finite")
    if not np.isfinite(area90_deg2) or area90_deg2 <= 0:
        raise ValueError("90% credible area must be finite and positive")
    if snr <= float(config["snr_too"]):
        return ToODecision("baseline", None, "snr_not_above_12")

    strategies = config["strategies"]
    if area90_deg2 <= float(strategies["gold_five_filter"]["area_max_deg2"]):
        return ToODecision("baseline_plus_too", "gold_five_filter", "too_triggered")
    if area90_deg2 <= float(strategies["gold"]["area_max_deg2"]):
        return ToODecision("baseline_plus_too", "gold", "too_triggered")
    if area90_deg2 < float(strategies["silver"]["area_max_deg2"]):
        return ToODecision("baseline_plus_too", "silver", "too_triggered")
    return ToODecision("baseline", None, "area_not_below_500_deg2")


def _moc_probability_data(mocmap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq = np.asarray(mocmap["UNIQ"])
    density = np.asarray(mocmap["PROBDENSITY"], dtype=float)
    area_sr = np.asarray(uniq2pixarea(uniq), dtype=float)
    probability = density * area_sr
    valid = np.isfinite(density) & np.isfinite(probability) & (probability > 0)
    if not np.any(valid):
        raise ValueError("Sky map has no finite positive posterior probability")
    probability = np.where(valid, probability, 0.0)
    probability /= probability.sum()
    credible_level = find_greedy_credible_levels(
        probability,
        np.where(valid, density, -np.inf),
    )
    return probability, credible_level, area_sr


def credible_area_deg2(mocmap, level: float = 0.9) -> float:
    """Compute the sky area inside a greedy credible region of a MOC map."""

    if not 0 < level <= 1:
        raise ValueError("level must be in (0, 1]")
    probability, credible_level, area_sr = _moc_probability_data(mocmap)
    selected = (credible_level <= level) & (probability > 0)
    if not np.any(selected):
        selected[np.argmax(probability)] = True
    return float(np.degrees(1.0) ** 2 * area_sr[selected].sum())


def maximum_posterior_coordinate(mocmap) -> tuple[float, float]:
    """Return the center of the highest posterior-density MOC pixel."""

    density = np.asarray(mocmap["PROBDENSITY"], dtype=float)
    if not np.any(np.isfinite(density)):
        raise ValueError("Sky map has no finite posterior density")
    index = int(np.nanargmax(density))
    order, ipix = uniq2nest(np.asarray(mocmap["UNIQ"])[[index]])
    theta, phi = hp.pix2ang(2 ** int(order[0]), int(ipix[0]), nest=True)
    return float(np.degrees(phi)), float(90.0 - np.degrees(theta))


def probability_rank_tiles(
    mocmap,
    ra: Sequence[float],
    dec: Sequence[float],
    *,
    area90_deg2: float,
    level: float = 0.9,
    nside: int = 256,
    effective_fov_deg2: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign samples to probability-prioritized equal-area tile ranks.

    This is an intentionally cheap approximation to a geometric Rubin tiling:
    fixed-order pixels are sorted by posterior density and grouped in blocks of
    ``effective_fov_deg2``.  Coordinates outside the credible region receive -1.
    """

    if effective_fov_deg2 <= 0:
        raise ValueError("effective_fov_deg2 must be positive")
    probability, credible_level = rasterized_posterior(mocmap, nside=nside)
    eligible = np.flatnonzero((credible_level <= level) & (probability > 0))
    if eligible.size == 0:
        raise ValueError(f"Sky map has no pixels inside the {level:g} credible region")

    pixel_area_deg2 = float(hp.nside2pixarea(nside, degrees=True))
    n_tiles = max(1, math.ceil(float(area90_deg2) / effective_fov_deg2))
    ordered = eligible[np.argsort(-probability[eligible], kind="stable")]
    pixel_to_tile = np.full(len(probability), -1, dtype=int)
    tile_rank = np.floor(
        np.arange(len(ordered), dtype=float) * pixel_area_deg2 / effective_fov_deg2
    ).astype(int)
    tile_rank = np.clip(tile_rank, 0, n_tiles - 1)
    pixel_to_tile[ordered] = tile_rank

    theta = np.radians(90.0 - np.asarray(dec, dtype=float))
    phi = np.radians(np.mod(np.asarray(ra, dtype=float), 360.0))
    sample_pixels = hp.ang2pix(nside, theta, phi, nest=True)
    sample_tiles = pixel_to_tile[sample_pixels]
    tile_probability = np.bincount(
        tile_rank,
        weights=probability[ordered],
        minlength=n_tiles,
    ).astype(float)
    return sample_tiles, tile_probability


def _earth_location(observatory: Mapping[str, Any]) -> EarthLocation:
    return EarthLocation.from_geodetic(
        float(observatory["longitude_deg"]) * u.deg,
        float(observatory["latitude_deg"]) * u.deg,
        float(observatory["height_m"]) * u.m,
    )


def find_visibility_windows(
    trigger_mjd: float,
    ra_deg: float,
    dec_deg: float,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Find the first configured post-trigger night windows for one sky anchor."""

    observatory = config["observatory"]
    number_of_nights = int(config["number_of_nights"])
    step_minutes = float(config["visibility_grid_minutes"])
    if step_minutes <= 0:
        raise ValueError("visibility_grid_minutes must be positive")
    step_days = step_minutes / (24.0 * 60.0)
    search_days = number_of_nights + 2.5
    mjd = float(trigger_mjd) + np.arange(
        0.0,
        search_days + step_days / 2.0,
        step_days,
    )
    times = Time(mjd, format="mjd", scale="utc")
    frame = AltAz(obstime=times, location=_earth_location(observatory))
    target = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg)
    target_alt = target.transform_to(frame).alt.deg
    sun_alt = get_sun(times).transform_to(frame).alt.deg
    visible = (
        (target_alt >= float(observatory["target_altitude_min_deg"]))
        & (sun_alt <= float(observatory["sun_altitude_max_deg"]))
        & (mjd >= float(trigger_mjd))
    )
    indices = np.flatnonzero(visible)
    columns = [
        "night_index",
        "start_mjd",
        "end_mjd",
        "duration_hours",
        "start_utc",
        "end_utc",
    ]
    if indices.size == 0:
        return pd.DataFrame(columns=columns)
    starts = indices[np.r_[True, np.diff(indices) > 1]]
    ends = indices[np.r_[np.diff(indices) > 1, True]]
    rows = []
    for night_index, (start, end) in enumerate(zip(starts, ends)):
        if night_index >= number_of_nights:
            break
        rows.append(
            {
                "night_index": int(night_index),
                "start_mjd": float(mjd[start]),
                "end_mjd": float(mjd[end]),
                "duration_hours": float((mjd[end] - mjd[start]) * 24.0),
                "start_utc": str(times[start].isot),
                "end_utc": str(times[end].isot),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def scan_duration_seconds(
    filters: Sequence[str],
    exposure_sec: float,
    n_tiles: int,
    visit_overhead_sec: float,
    filter_change_sec: float,
) -> float:
    """Return the execution time of one full tiled multi-filter scan."""

    if not filters or exposure_sec <= 0 or n_tiles <= 0:
        raise ValueError(
            "A scan needs filters, positive exposure, and at least one tile"
        )
    visits = len(filters) * n_tiles * (float(exposure_sec) + visit_overhead_sec)
    changes = max(0, len(filters) - 1) * filter_change_sec
    return float(visits + changes)


def _scan_rows(
    *,
    scan_id: int,
    scan_start_mjd: float,
    window_end_mjd: float,
    filters: Sequence[str],
    exposure_sec: float,
    n_tiles: int,
    visit_overhead_sec: float,
    filter_change_sec: float,
    night_index: int,
    scan_index: int,
    scan_kind: str,
    condition_group: str,
) -> tuple[list[dict[str, Any]], tuple[float, float] | None, dict[str, Any]]:
    duration_sec = scan_duration_seconds(
        filters,
        exposure_sec,
        n_tiles,
        visit_overhead_sec,
        filter_change_sec,
    )
    scan_end_mjd = scan_start_mjd + duration_sec / SECONDS_PER_DAY
    execution_end = min(scan_end_mjd, window_end_mjd)
    rows: list[dict[str, Any]] = []
    band_block_start_sec = 0.0
    for band_order, band in enumerate(filters):
        for tile_index in range(n_tiles):
            midpoint_sec = band_block_start_sec
            midpoint_sec += tile_index * (float(exposure_sec) + visit_overhead_sec)
            midpoint_sec += float(exposure_sec) / 2.0
            mjd = scan_start_mjd + midpoint_sec / SECONDS_PER_DAY
            exposure_end_mjd = mjd + float(exposure_sec) / (2.0 * SECONDS_PER_DAY)
            if exposure_end_mjd <= window_end_mjd + 1e-12:
                rows.append(
                    {
                        "scan_id": int(scan_id),
                        "night_index": int(night_index),
                        "scan_index": int(scan_index),
                        "scan_kind": str(scan_kind),
                        "band_order": int(band_order),
                        "band": str(band),
                        "tile_index": int(tile_index),
                        "exposure_sec": float(exposure_sec),
                        "mjd": float(mjd),
                        "condition_group": str(condition_group),
                    }
                )
        band_block_start_sec += n_tiles * (float(exposure_sec) + visit_overhead_sec)
        if band_order < len(filters) - 1:
            band_block_start_sec += filter_change_sec
    interval = None
    if rows:
        interval = (float(scan_start_mjd), float(execution_end))
    audit = {
        "scan_id": int(scan_id),
        "night_index": int(night_index),
        "scan_index": int(scan_index),
        "scan_kind": str(scan_kind),
        "start_mjd": float(scan_start_mjd),
        "planned_end_mjd": float(scan_end_mjd),
        "executed_end_mjd": float(execution_end),
        "complete": bool(scan_end_mjd <= window_end_mjd + 1e-12),
        "filters": list(filters),
        "exposure_sec": float(exposure_sec),
    }
    return rows, interval, audit


def build_event_schedule(
    *,
    trigger_mjd: float,
    anchor_ra_deg: float,
    anchor_dec_deg: float,
    strategy_name: str,
    area90_deg2: float,
    tile_probability: np.ndarray,
    config: Mapping[str, Any],
) -> ToOSchedule:
    """Construct the event-level probability-prioritized ToO scan schedule."""

    strategy = config["strategies"][strategy_name]
    n_tiles = max(
        1,
        math.ceil(float(area90_deg2) / float(config["effective_fov_deg2"])),
    )
    if len(tile_probability) != n_tiles:
        raise ValueError("tile_probability length does not match the number of tiles")
    visibility = find_visibility_windows(
        trigger_mjd,
        anchor_ra_deg,
        anchor_dec_deg,
        config,
    )
    empty_columns = [
        "observation_id",
        "scan_id",
        "night_index",
        "scan_index",
        "scan_kind",
        "band_order",
        "band",
        "tile_index",
        "exposure_sec",
        "mjd",
        "condition_group",
    ]
    if visibility.empty:
        return ToOSchedule(
            pd.DataFrame(columns=empty_columns),
            visibility,
            [],
            [],
            n_tiles,
            None,
            {},
        )

    overhead = float(config["visit_overhead_sec"])
    filter_change = float(config["filter_change_sec"])
    rows: list[dict[str, Any]] = []
    busy_intervals: list[tuple[float, float]] = []
    scan_intervals: list[dict[str, Any]] = []
    next_scan_id = 0

    def add_scan(**kwargs) -> None:
        nonlocal next_scan_id
        scan_rows, interval, audit = _scan_rows(scan_id=next_scan_id, **kwargs)
        rows.extend(scan_rows)
        if interval is not None:
            busy_intervals.append(interval)
        scan_intervals.append(audit)
        next_scan_id += 1

    first_window = visibility.iloc[0]
    first_start = float(first_window["start_mjd"])
    first_end = float(first_window["end_mjd"])
    first_window_sec = (first_end - first_start) * SECONDS_PER_DAY
    night0_filters = [str(item) for item in strategy["night0_filters"]]
    night0_exposure = float(strategy["night0_exposure_sec"])
    night0_duration = scan_duration_seconds(
        night0_filters,
        night0_exposure,
        n_tiles,
        overhead,
        filter_change,
    )
    condition_group = str(strategy["condition_group"])

    if strategy_name.startswith("gold"):
        if first_window_sec >= 3.0 * night0_duration:
            n_full_scans = 3
            branch = "night0_three_full_scans"
        elif first_window_sec >= 2.0 * night0_duration:
            n_full_scans = 2
            branch = "night0_two_full_scans"
        elif first_window_sec >= night0_duration:
            n_full_scans = 1
            branch = "night0_one_full_scan_plus_r_if_time"
        else:
            n_full_scans = 0
            branch = "night0_partial_scan"
        if n_full_scans <= 1:
            starts = [first_start]
        else:
            gap = (first_window_sec - n_full_scans * night0_duration) / (
                n_full_scans - 1
            )
            starts = [
                first_start + index * (night0_duration + gap) / SECONDS_PER_DAY
                for index in range(n_full_scans)
            ]
        for scan_index, start in enumerate(starts):
            add_scan(
                scan_start_mjd=start,
                window_end_mjd=first_end,
                filters=night0_filters,
                exposure_sec=night0_exposure,
                n_tiles=n_tiles,
                visit_overhead_sec=overhead,
                filter_change_sec=filter_change,
                night_index=0,
                scan_index=scan_index,
                scan_kind="night0_full",
                condition_group=condition_group,
            )
        if n_full_scans == 1:
            full_end = first_start + night0_duration / SECONDS_PER_DAY
            if full_end < first_end:
                repeat_filter = str(strategy["night0_repeat_filter"])
                repeat_duration = scan_duration_seconds(
                    [repeat_filter],
                    night0_exposure,
                    n_tiles,
                    overhead,
                    filter_change,
                )
                repeat_start = max(
                    full_end,
                    first_end - repeat_duration / SECONDS_PER_DAY,
                )
                add_scan(
                    scan_start_mjd=repeat_start,
                    window_end_mjd=first_end,
                    filters=[repeat_filter],
                    exposure_sec=night0_exposure,
                    n_tiles=n_tiles,
                    visit_overhead_sec=overhead,
                    filter_change_sec=filter_change,
                    night_index=0,
                    scan_index=1,
                    scan_kind="night0_asteroid_rejection_repeat",
                    condition_group=condition_group,
                )
    else:
        branch = "night0_one_silver_scan"
        add_scan(
            scan_start_mjd=first_start,
            window_end_mjd=first_end,
            filters=night0_filters,
            exposure_sec=night0_exposure,
            n_tiles=n_tiles,
            visit_overhead_sec=overhead,
            filter_change_sec=filter_change,
            night_index=0,
            scan_index=0,
            scan_kind="night0_full",
            condition_group=condition_group,
        )

    later_filters = [str(item) for item in strategy["later_filters"]]
    later_exposure = float(strategy["later_exposure_sec"])
    for window in visibility.iloc[1:].itertuples(index=False):
        add_scan(
            scan_start_mjd=float(window.start_mjd),
            window_end_mjd=float(window.end_mjd),
            filters=later_filters,
            exposure_sec=later_exposure,
            n_tiles=n_tiles,
            visit_overhead_sec=overhead,
            filter_change_sec=filter_change,
            night_index=int(window.night_index),
            scan_index=0,
            scan_kind="later_night_full",
            condition_group=condition_group,
        )

    observations = pd.DataFrame(rows)
    if observations.empty:
        observations = pd.DataFrame(columns=empty_columns[1:])
    observations = observations.sort_values(
        ["mjd", "band_order", "tile_index"],
        kind="stable",
    ).reset_index(drop=True)
    observations.insert(
        0,
        "observation_id",
        np.arange(900000001, 900000001 + len(observations), dtype=np.int64),
    )
    coverage = {}
    for night_index, group in observations.groupby("night_index"):
        tiles = group["tile_index"].drop_duplicates().to_numpy(dtype=int)
        coverage[str(int(night_index))] = float(tile_probability[tiles].sum())
    return ToOSchedule(
        observations,
        visibility,
        busy_intervals,
        scan_intervals,
        n_tiles,
        branch,
        coverage,
    )


def _sky_geometry(
    mjd: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    observatory: Mapping[str, Any],
) -> pd.DataFrame:
    times = Time(mjd, format="mjd", scale="utc")
    frame = AltAz(obstime=times, location=_earth_location(observatory))
    targets = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    target_altaz = targets.transform_to(frame)
    sun_altaz = get_sun(times).transform_to(frame)
    moon_altaz = get_body(
        "moon",
        times,
        location=_earth_location(observatory),
    ).transform_to(frame)
    return pd.DataFrame(
        {
            "target_altitude_deg": target_altaz.alt.deg,
            "airmass": np.asarray(target_altaz.secz, dtype=float),
            "sun_altitude_deg": sun_altaz.alt.deg,
            "moon_altitude_deg": moon_altaz.alt.deg,
            "moon_distance_deg": target_altaz.separation(moon_altaz).deg,
        }
    )


class ToOConditionLibrary:
    """Nearest-neighbor empirical observing conditions from an OpSim database."""

    def __init__(self, templates: pd.DataFrame, config: Mapping[str, Any]):
        self.config = config
        self.scales = np.array(
            [
                float(config["airmass_scale"]),
                float(config["sun_altitude_scale_deg"]),
                float(config["moon_altitude_scale_deg"]),
                float(config["moon_distance_scale_deg"]),
            ],
            dtype=float,
        )
        self.groups: dict[tuple[str, str], tuple[pd.DataFrame, KDTree]] = {}
        for (condition_group, band), frame in templates.groupby(
            ["condition_group", "band"],
            sort=False,
        ):
            frame = frame.reset_index(drop=True)
            features = frame[["airmass", "sunAlt", "moonAlt", "moonDistance"]].to_numpy(
                dtype=float
            )
            self.groups[(str(condition_group), str(band))] = (
                frame,
                KDTree(features / self.scales),
            )

    @classmethod
    def from_database(
        cls,
        database: str | Path,
        config: Mapping[str, Any],
    ) -> ToOConditionLibrary:
        database_path = Path(database).expanduser().resolve()
        if not database_path.is_file():
            raise FileNotFoundError(f"OpSim database not found: {database_path}")
        labels = {
            "gold": str(config["gold_template_label"]),
            "silver": str(config["silver_template_label"]),
        }
        query = """
            SELECT observationId, observationStartMJD, band, visitExposureTime,
                   airmass, seeingFwhmEff, skyBrightness, fiveSigmaDepth,
                   sunAlt, moonAlt, moonDistance, observation_reason
            FROM observations
            WHERE observation_reason LIKE ?
        """
        frames = []
        connection_uri = f"file:{database_path}?mode=ro"
        with sqlite3.connect(connection_uri, uri=True) as connection:
            for group, label in labels.items():
                frame = pd.read_sql_query(
                    query,
                    connection,
                    params=(f"too_{label}_%",),
                )
                frame["condition_group"] = group
                frames.append(frame)
        templates = pd.concat(frames, ignore_index=True)
        required = [
            "visitExposureTime",
            "airmass",
            "seeingFwhmEff",
            "skyBrightness",
            "fiveSigmaDepth",
            "sunAlt",
            "moonAlt",
            "moonDistance",
        ]
        templates = templates.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        templates = templates.loc[templates["visitExposureTime"] > 0].copy()
        for group, required_bands in {
            "gold": set("ugriz"),
            "silver": set("gi"),
        }.items():
            available = set(
                templates.loc[templates["condition_group"] == group, "band"].astype(str)
            )
            missing = sorted(required_bands - available)
            if missing:
                raise RuntimeError(
                    f"No {group} ToO condition templates for bands: {missing}"
                )
        return cls(templates, config)

    def assign(self, visits: pd.DataFrame) -> pd.DataFrame:
        """Attach empirical m5/seeing/sky and SIMLIB calibration to visits."""

        if visits.empty:
            return visits.copy()
        result = visits.copy().reset_index(drop=True)
        result["five_sigma_depth"] = np.nan
        result["seeing_fwhm_eff_arcsec"] = np.nan
        result["sky_brightness_mag_arcsec2"] = np.nan
        result["condition_template_observation_id"] = -1
        result["condition_match_distance"] = np.nan
        for (condition_group, band), indices in result.groupby(
            ["condition_group", "band"]
        ).groups.items():
            key = (str(condition_group), str(band))
            if key not in self.groups:
                raise RuntimeError(f"No condition templates for {key}")
            templates, tree = self.groups[key]
            idx = np.asarray(list(indices), dtype=int)
            features = result.loc[
                idx,
                [
                    "airmass",
                    "sun_altitude_deg",
                    "moon_altitude_deg",
                    "moon_distance_deg",
                ],
            ].to_numpy(dtype=float)
            distance, matched = tree.query(features / self.scales, k=1)
            matched_frame = templates.iloc[matched[:, 0]].reset_index(drop=True)
            exposure_scale = result.loc[idx, "exposure_sec"].to_numpy(
                dtype=float
            ) / matched_frame["visitExposureTime"].to_numpy(dtype=float)
            result.loc[idx, "five_sigma_depth"] = matched_frame[
                "fiveSigmaDepth"
            ].to_numpy(dtype=float) + 1.25 * np.log10(exposure_scale)
            result.loc[idx, "seeing_fwhm_eff_arcsec"] = matched_frame[
                "seeingFwhmEff"
            ].to_numpy(dtype=float)
            result.loc[idx, "sky_brightness_mag_arcsec2"] = matched_frame[
                "skyBrightness"
            ].to_numpy(dtype=float)
            result.loc[idx, "condition_template_observation_id"] = matched_frame[
                "observationId"
            ].to_numpy(dtype=int)
            result.loc[idx, "condition_match_distance"] = distance[:, 0]
        return opsim_to_simlib_calibration(result)


def opsim_to_simlib_calibration(visits: pd.DataFrame) -> pd.DataFrame:
    """Convert m5, sky brightness, and seeing into SIMLIB quantities."""

    result = visits.copy()
    sigma_psf_arcsec = result["seeing_fwhm_eff_arcsec"] / (
        2.0 * np.sqrt(2.0 * np.log(2.0))
    )
    psf_pixels = sigma_psf_arcsec / SIMLIB_PIXEL_SIZE_ARCSEC
    noise_area = 4.0 * np.pi * sigma_psf_arcsec**2
    m5 = result["five_sigma_depth"]
    sky = result["sky_brightness_mag_arcsec2"]
    delta_mag = m5 - sky
    zpt = 2.0 * m5 - sky
    zpt += 2.5 * np.log10(25.0 * noise_area)
    zpt += 2.5 * np.log10(1.0 + 10.0 ** (-0.4 * delta_mag) / noise_area)
    sky_sigma = np.sqrt(10.0 ** (-0.4 * (sky - zpt)) * SIMLIB_PIXEL_SIZE_ARCSEC**2)
    result["PSF"] = psf_pixels
    result["ZPT"] = zpt
    result["SKYSIG"] = sky_sigma
    columns = ["PSF", "ZPT", "SKYSIG"]
    if not np.all(np.isfinite(result[columns].to_numpy(dtype=float))):
        raise ValueError("Non-finite ToO SIMLIB calibration value")
    return result


def materialize_sample_visits(
    schedule: ToOSchedule,
    samples: pd.DataFrame,
    condition_library: ToOConditionLibrary,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Materialize event scans for the tile assigned to every coordinate sample."""

    columns = [
        "sample_index",
        "expMJD",
        "PSF",
        "ZPT",
        "SKYSIG",
        "BAND",
        "ObsID",
        "NEXPOSE",
    ]
    if schedule.observations.empty or samples.empty:
        return pd.DataFrame(columns=columns)
    eligible = samples.loc[samples["tile_index"] >= 0].copy()
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    visits = eligible.merge(
        schedule.observations,
        on="tile_index",
        how="inner",
        validate="many_to_many",
    )
    geometry = _sky_geometry(
        visits["mjd"].to_numpy(dtype=float),
        visits["ra"].to_numpy(dtype=float),
        visits["dec"].to_numpy(dtype=float),
        config["observatory"],
    )
    visits = pd.concat([visits.reset_index(drop=True), geometry], axis=1)
    observatory = config["observatory"]
    visible = (
        np.isfinite(visits["airmass"])
        & (
            visits["target_altitude_deg"]
            >= float(observatory["target_altitude_min_deg"])
        )
        & (visits["sun_altitude_deg"] <= float(observatory["sun_altitude_max_deg"]))
    )
    visits = visits.loc[visible].reset_index(drop=True)
    visits = condition_library.assign(visits)
    result = pd.DataFrame(
        {
            "sample_index": visits["sample_index"].astype(int),
            "expMJD": visits["mjd"].astype(float),
            "PSF": visits["PSF"].astype(float),
            "ZPT": visits["ZPT"].astype(float),
            "SKYSIG": visits["SKYSIG"].astype(float),
            "BAND": visits["band"].astype(str),
            "ObsID": visits["observation_id"].astype(np.int64),
            "NEXPOSE": 1,
        }
    )
    return result.sort_values(["sample_index", "expMJD"], kind="stable").reset_index(
        drop=True
    )


def filter_unrelated_too_observations(survey) -> int:
    """Remove built-in OpSim ToOs and rebuild the survey spatial index."""

    original = survey.opsimdf
    if "observation_reason" not in original:
        return 0
    reason = original["observation_reason"].fillna("").astype(str).str.lower()
    keep = ~reason.str.startswith("too_")
    removed = int((~keep).sum())
    filtered = original.loc[keep].copy()
    filtered.attrs = original.attrs.copy()
    survey.opsimdf = filtered.sort_values("observationStartMJD")
    survey.tree = BallTree(
        survey.opsimdf[["_dec", "_ra"]].to_numpy(),
        leaf_size=50,
        metric="haversine",
    )
    survey._hp_rep = None
    survey._survey = None
    return removed


def sample_coordinates_with_indices(
    survey,
    ra: np.ndarray,
    dec: np.ndarray,
    redshift: np.ndarray,
    *,
    nside: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the baseline footprint and retain sample-index/LIBID alignment."""

    theta = np.radians(90.0 - np.asarray(dec, dtype=float))
    phi = np.radians(np.mod(np.asarray(ra, dtype=float), 360.0))
    ring_pixels = hp.ang2pix(nside, theta, phi, nest=False)
    in_footprint = np.isin(ring_pixels, survey.hp_rep.index.to_numpy())
    survey.sample_coordinates(
        ra,
        dec,
        redshift,
        nsides=nside,
        is_deg=True,
    )
    selected_indices = np.flatnonzero(in_footprint)
    if len(survey.survey) != len(selected_indices):
        raise RuntimeError("OpSim footprint sampling lost sample-index alignment")
    survey._survey["sample_index"] = selected_indices
    libid = np.full(len(ra), -1, dtype=int)
    libid[selected_indices] = survey.survey.index.to_numpy(dtype=int)
    return in_footprint, libid


class AugmentedSNANASimlib(SNANA_Simlib):
    """SNANA SIMLIB writer that replaces baseline time with event-specific ToO."""

    def __init__(
        self,
        *args,
        too_observations: pd.DataFrame | None = None,
        busy_intervals: Sequence[tuple[float, float]] = (),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.too_observations = (
            too_observations.copy() if too_observations is not None else pd.DataFrame()
        )
        self.busy_intervals = list(busy_intervals)
        self.removed_baseline_visits = 0
        self.too_visits_written = 0

    def _remove_busy_baseline(self, observations: pd.DataFrame) -> pd.DataFrame:
        if observations.empty or not self.busy_intervals:
            return observations
        busy = np.zeros(len(observations), dtype=bool)
        mjd = observations["expMJD"].to_numpy(dtype=float)
        for start, end in self.busy_intervals:
            busy |= (mjd >= float(start)) & (mjd <= float(end))
        self.removed_baseline_visits += int(busy.sum())
        return observations.loc[~busy].copy()

    def LIBdata(self, observations: pd.DataFrame) -> str:
        lines = []
        for row in observations.itertuples(index=False):
            band = "Y" if str(row.BAND).lower() == "y" else str(row.BAND)
            lines.append(
                opsim_utils.dataline(
                    float(row.expMJD),
                    int(row.ObsID),
                    band,
                    self.CCDgain,
                    self.CCDnoise,
                    float(row.SKYSIG),
                    float(row.PSF),
                    float(row.ZPT),
                    self.ZPTNoise,
                    nexpose=int(row.NEXPOSE),
                )
            )
        return "\n".join(lines) + ("\n" if lines else "")

    def get_SIMLIB_footer(self) -> str:
        return f"END_OF_SIMLIB:    {len(self.OpSimSurvey.survey):10d} ENTRIES"

    def write_SIMLIB(self, write_batch_size: int = 10, buffer_size: int = 8192):
        tstart = time.time()
        print(f"Writing augmented SIMLIB in {self.out_path}")
        grouped_too = (
            {
                int(sample_index): group.drop(columns="sample_index").copy()
                for sample_index, group in self.too_observations.groupby("sample_index")
            }
            if not self.too_observations.empty
            else {}
        )
        with open(self.out_path, "w", buffering=buffer_size) as simlib_file:
            simlib_file.write(self.get_SIMLIB_doc())
            simlib_file.write(self.get_SIMLIB_header())
            buffered = []
            for (libid, field), baseline in zip(
                self.OpSimSurvey.survey.iterrows(),
                self.OpSimSurvey.get_survey_obs(),
            ):
                baseline = baseline.copy()
                baseline["NEXPOSE"] = int(self.nexpose)
                baseline = self._remove_busy_baseline(baseline)
                sample_index = int(field["sample_index"])
                too = grouped_too.get(sample_index, pd.DataFrame())
                self.too_visits_written += len(too)
                observations = pd.concat([baseline, too], ignore_index=True)
                observations = observations.sort_values("expMJD", kind="stable")
                field_label = field.get("field_label", None)
                if not too.empty:
                    field_label = f"{field_label}+ToO" if field_label else "ToO"
                buffered.append(
                    self.LIBheader(
                        int(libid),
                        np.degrees(field["hp_ra"]),
                        np.degrees(field["hp_dec"]),
                        observations,
                        field.get("redshift", None),
                        field_label=field_label,
                    )
                )
                buffered.append(self.LIBdata(observations))
                buffered.append(self.LIBfooter(int(libid)))
                if (int(libid) + 1) % write_batch_size == 0:
                    simlib_file.write("".join(buffered))
                    buffered = []
            simlib_file.write("".join(buffered))
            simlib_file.write(self.get_SIMLIB_footer())
        print(f"Augmented SIMLIB wrote in {time.time() - tstart:.2f} sec.\n")
