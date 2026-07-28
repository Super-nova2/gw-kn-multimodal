#!/usr/bin/env python3
"""Unified ablation comparison for retrieval and triplet classification.

This script evaluates six model settings under the same runtime optical
window, shared gallery construction, and shared evaluation data:

1. optical-only
2. w/o contrastive learning
3. w/o cross-attention
4. w/o fusion branch (contrastive-only scoring)
5. full multimodal (w/o hard mining)
6. full multimodal + hard mining

Outputs a single JSON artifact containing paper-table-ready retrieval rows,
per-model retrieval metrics, and multimodal triplet-classification metrics.
"""

import argparse
import gc
import hashlib
import h5py
import json
import os
import random
import re
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parents[1]
WORKSPACE_DIR = MODEL_DIR.parent.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
FINK_RF_SRC_DIR = WORKSPACE_DIR / "fink-rf-reproduction" / "src"
if FINK_RF_SRC_DIR.exists() and str(FINK_RF_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(FINK_RF_SRC_DIR))

try:
    from fink_rf_repro.features import extract_feature_matrix_from_batch as fink_extract_feature_matrix_from_batch
    from fink_rf_repro.model_io import load_artifact as load_fink_rf_artifact
except Exception:  # pragma: no cover - optional standalone baseline dependency
    fink_extract_feature_matrix_from_batch = None
    load_fink_rf_artifact = None

from data_loader import (  # noqa: E402
    RelationalHDF5Dataset,
    BalancedGWBatchedSampler,
    _build_dataloader,
    _read_root_time_window_attrs,
    apply_runtime_input_window_torch,
    build_effective_input_window_metadata,
    build_gw_to_lc_mapping,
)
from model import OpticalKNClassifier, migrate_time_embed_state_dict  # noqa: E402
from retrieval_gallery import (  # noqa: E402
    aggregate_gallery_outcomes,
    build_comparison_model_specs,
    build_curve_rows,
    build_prefixed_gallery_specs,
    build_synthetic_time_sky_candidate_sequences,
    build_time_sky_candidate_sequences,
    compute_source_macro_and_gap,
    gallery_outcome_metric_contributions,
    uniform_random_tie_metric_contributions,
    extract_gallery_negative_abs_dt_days,
    PLOT_FONT_BASE,
    RETRIEVAL_TWO_ROW_LEGEND_FIGSIZE,
    _plot_method_color_map,
    _plot_method_draw_order,
    _plot_method_label,
    plot_retrieval_curves,
    score_all_galleries_skymap,
)
from plot_style import add_panel_labels_below, apply_mnras_style, layout_top_below_legend  # noqa: E402
from scripts.eval.evaluate import (  # noqa: E402
    _autocast_context,
    _build_gallery_query_cache,
    _compute_credible_level_single_gw,
    _is_dual_fusion_model,
    _lookup_source_type,
    _model_requires_cred_level,
    _model_requires_time_delta,
    _resolve_eval_amp,
    build_ref_time,
    evaluate_classification_triplet,
    evaluate_classification_triplet_by_source,
    extract_triplet_logits,
    load_gw_event_time_mjd_table,
    load_gw_source_types,
    load_model,
    load_negative_gw_indices,
    load_negative_optical_samples,
    parse_day_windows,
    parse_dt_bin_edges,
)
from scripts.eval.eval_run_io import (  # noqa: E402
    AtomicGzipCsvWriter,
    CLASSIFICATION_FIELDS,
    RETRIEVAL_FIELDS,
    classification_prediction_rows,
    mark_run_success,
    prepare_output_directory,
    retrieval_outcome_rows,
    stable_digest,
    source_tree_digest,
)


DEFAULT_COMPARISON_WINDOW = (-0.1, 0.2)
DEFAULT_DT_BIN_EDGES = "0,3,5,10,30,100,300,inf"
TABLE_METRIC_LABELS = ["R@1", "R@5", "R@10", "MRR"]
CANDIDATE_TIME_DELTA_MODES = {"native", "zero"}


def normalize_candidate_time_delta_mode(value: Any) -> str:
    """Validate the runtime GW-to-candidate time-delta treatment."""
    mode = str(value or "native").strip().lower()
    if mode not in CANDIDATE_TIME_DELTA_MODES:
        raise ValueError(
            "candidate_time_delta_mode must be one of "
            f"{sorted(CANDIDATE_TIME_DELTA_MODES)}; got {value!r}."
        )
    return mode


def apply_candidate_time_delta_mode(
    dt_days: Optional[np.ndarray],
    *,
    mode: str,
    n_candidates: int,
) -> Optional[np.ndarray]:
    """Apply an inference-only ablation without changing model architecture."""
    normalized_mode = normalize_candidate_time_delta_mode(mode)
    n_candidates = int(n_candidates)
    if n_candidates < 0:
        raise ValueError("n_candidates must be non-negative.")
    if normalized_mode == "zero":
        return np.zeros((n_candidates,), dtype=np.float32)
    return dt_days


def _lookup_event_time_mjd(gw_event_time_mjd_table, gw_id: int) -> Optional[float]:
    if gw_event_time_mjd_table is None:
        return None
    if isinstance(gw_event_time_mjd_table, torch.Tensor):
        value = gw_event_time_mjd_table[int(gw_id)].detach().float().cpu().item()
    else:
        value = np.asarray(gw_event_time_mjd_table, dtype=np.float64).reshape(-1)[int(gw_id)]
    if not np.isfinite(float(value)):
        return None
    return float(value)
TABLE_METRIC_KEYS = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
DEFAULT_TUTORIAL_NEG_DATA_PATH = str(
    (WORKSPACE_DIR / "data" / "Optical_Only_dataset" / "Tutorial_negative_dataset.h5").resolve()
)
DEFAULT_TUTORIAL_NEG_GROUP = "Tutorial/optical_data"

# Scalar column validation mapping for redshift catalog cross-check.
# Each entry: (catalog_column, scalar_index, transform)
#   transform "direct"     → scalar == catalog_value
#   transform "cos"        → scalar == cos(catalog_value)
#   transform "scale_1000" → scalar == catalog_value / 1000
_SCALAR_VALIDATION_COLS: List[Tuple[str, int, str]] = [
    ("mass1_detector", 0, "direct"),
    ("mass2_detector", 1, "direct"),
    ("spin1z",         2, "direct"),
    ("spin2z",         3, "direct"),
    ("inclination",    4, "cos"),
    ("distmean",       5, "scale_1000"),
    ("diststd",        6, "scale_1000"),
]
_SCALAR_VALIDATION_RTOL = 1e-4
_SCALAR_VALIDATION_ATOL = 1e-6


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(base_dir: Path, value: Optional[str]) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    jobfs_dir = os.environ.get("JOBFS_DIR")
    if jobfs_dir:
        staged = Path(jobfs_dir) / path.name
        if staged.is_file():
            return str(staged)
    return str(path)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _parse_gallery_sizes(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value or "100,500")
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or [100, 500]


def _parse_float_sequence(value: Any, *, expected_len: int, name: str) -> Tuple[float, ...]:
    if isinstance(value, str):
        vals = [float(part.strip()) for part in value.split(",") if part.strip()]
    else:
        vals = [float(part) for part in value]
    if len(vals) != int(expected_len):
        raise ValueError(f"{name} must contain exactly {expected_len} values.")
    if np.any(~np.isfinite(np.asarray(vals, dtype=np.float64))):
        raise ValueError(f"{name} must contain finite values.")
    return tuple(vals)


def _parse_n_neg_samples(value: Any, default: int = -1) -> int:
    if value is None:
        return int(default)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "all", "full"}:
            return -1
    out = int(value)
    return -1 if out <= 0 else out


def _resolve_target_samples(n_neg_samples: int) -> Optional[int]:
    value = int(n_neg_samples)
    return value if value > 0 else None


def _gallery_mode_supports_skymap_only(gallery_candidate_mode: str) -> bool:
    return str(gallery_candidate_mode).strip().lower() in {
        "time_sky_hard",
        "synthetic_time_sky_hard",
    }


# ---------------------------------------------------------------------------
# Redshift metadata recovery helpers
# ---------------------------------------------------------------------------
def _parse_hdf5_gw_id(gw_id) -> Tuple[str, int]:
    """Parse an HDF5 GW ID of the form ``<source>_<event_id>``.

    Args:
        gw_id: String or bytes, e.g. ``"bns_6"`` or ``b"nsbh_114"``.

    Returns:
        (source, event_id) tuple, e.g. ``("bns", 6)``.

    Raises:
        ValueError: If the format is malformed.
    """
    if isinstance(gw_id, bytes):
        gw_id = gw_id.decode("ascii")
    gw_id = str(gw_id).strip()
    if "_" not in gw_id:
        raise ValueError(f"GW ID must contain '_' separating source and event_id: {gw_id!r}")
    parts = gw_id.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"GW ID format error: {gw_id!r}")
    source = parts[0].strip()
    if not source:
        raise ValueError(f"GW ID has empty source: {gw_id!r}")
    try:
        event_id = int(parts[1])
    except ValueError:
        raise ValueError(f"GW ID event_id must be an integer: {gw_id!r}")
    return source, event_id


def _load_catalog_redshift_map(catalog_path: str) -> Dict[int, float]:
    """Load ``simulation_id`` → ``redshift`` from a CSV catalog.

    Args:
        catalog_path: Path to a CSV with ``simulation_id`` and ``redshift`` columns.

    Returns:
        Dict mapping integer simulation_id to float redshift.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        KeyError: If required columns are missing.
    """
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Redshift catalog not found: {catalog_path}")
    df = pd.read_csv(catalog_path)
    for col in ("simulation_id", "redshift"):
        if col not in df.columns:
            raise KeyError(f"Catalog {catalog_path} is missing required column '{col}'")
    z_map: Dict[int, float] = {}
    for _, row in df.iterrows():
        sim_id = int(row["simulation_id"])
        z_map[sim_id] = float(row["redshift"])
    return z_map


def _normalize_redshift_catalog_paths(
    redshift_catalogs: Optional[Dict[str, str]],
    cfg_dir: Path,
) -> Dict[str, str]:
    """Resolve relative catalog paths against the config directory.

    Args:
        redshift_catalogs: Source label → CSV path mapping, or None/empty.
        cfg_dir: Config file directory for resolving relative paths.

    Returns:
        Resolved source → absolute path mapping.  Empty dict if input is None/empty.
    """
    if not redshift_catalogs:
        return {}
    resolved: Dict[str, str] = {}
    for source, path_str in redshift_catalogs.items():
        p = Path(str(path_str)).expanduser()
        if not p.is_absolute():
            p = (cfg_dir / p).resolve()
        resolved[str(source)] = str(p)
    return resolved


def _build_redshift_metadata_from_catalogs(
    test_data_path: str,
    redshift_catalogs: Dict[str, str],
    *,
    validate_scalars: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Recover redshift for each GW event from source catalogs.

    Reads HDF5 ``events/gw_data/ids`` and ``events/gw_data/source_type``,
    parses each GW ID, and looks up the redshift in the appropriate source
    catalog (BNS or NSBH).

    Args:
        test_data_path: Path to the combined HDF5 test dataset.
        redshift_catalogs: Source label → CSV path mapping.
        validate_scalars: If True, cross-check HDF5 ``gw_data/scalars`` against
            catalog GW parameters to catch wrong-catalog misconfigurations.

    Returns:
        Dict mapping GW index (int) to ``{"redshift": float}``.

    Raises:
        ValueError: If a GW event's source has no configured catalog, its
            event_id is not found in the catalog, the parsed-ID source does not
            match ``source_type``, or scalar validation fails.
    """
    with h5py.File(test_data_path, "r") as f:
        if "events/gw_data/ids" not in f or "events/gw_data/source_type" not in f:
            raise KeyError("HDF5 missing events/gw_data/ids or events/gw_data/source_type")
        ids_arr = np.asarray(f["events/gw_data/ids"][:])
        source_arr = np.asarray(f["events/gw_data/source_type"][:])
        scalars_arr = np.asarray(f["events/gw_data/scalars"][:]) if validate_scalars else None

    if validate_scalars and scalars_arr is not None and scalars_arr.shape[1] < len(_SCALAR_VALIDATION_COLS):
        raise ValueError(
            f"HDF5 scalars has {scalars_arr.shape[1]} columns; "
            f"expected at least {len(_SCALAR_VALIDATION_COLS)} for validation"
        )

    # Load catalog redshift maps lazily; also cache DataFrames for scalar validation
    catalog_maps: Dict[str, Dict[int, float]] = {}
    catalog_dfs: Dict[str, "pd.DataFrame"] = {}

    metadata: Dict[int, Dict[str, Any]] = {}
    for idx in range(ids_arr.shape[0]):
        raw_id = ids_arr[idx]
        if isinstance(raw_id, bytes):
            raw_id = raw_id.decode("ascii")
        source_str = source_arr[idx]
        if isinstance(source_str, bytes):
            source_str = source_str.decode("ascii")

        id_source, event_id = _parse_hdf5_gw_id(str(raw_id))

        # --- Fix 2: cross-check parsed-ID source against HDF5 source_type ---
        if id_source != str(source_str):
            raise ValueError(
                f"GW ID source mismatch at index {idx}: "
                f"ids={raw_id!r} parses to source={id_source!r}, "
                f"but source_type={source_str!r}"
            )

        if str(source_str) not in redshift_catalogs:
            raise ValueError(
                f"No redshift catalog configured for source '{source_str}' "
                f"(GW idx {idx}, id={raw_id}). Available: {list(redshift_catalogs.keys())}"
            )

        catalog_path = redshift_catalogs[str(source_str)]
        if str(source_str) not in catalog_maps:
            catalog_maps[str(source_str)] = _load_catalog_redshift_map(catalog_path)
            if validate_scalars:
                catalog_dfs[str(source_str)] = pd.read_csv(catalog_path)

        z_map = catalog_maps[str(source_str)]
        if event_id not in z_map:
            raise ValueError(
                f"Event ID {event_id} (source={source_str}, id={raw_id}) "
                f"not found in catalog {catalog_path}"
            )

        # --- Fix 1: scalar consistency validation ---
        if validate_scalars and scalars_arr is not None:
            cat_df = catalog_dfs[str(source_str)]
            cat_row = cat_df[cat_df["simulation_id"] == event_id]
            if len(cat_row) != 1:
                raise ValueError(
                    f"Expected exactly 1 catalog row for simulation_id={event_id} "
                    f"in {catalog_path}; got {len(cat_row)}"
                )
            row = cat_row.iloc[0]
            gw_scalars = scalars_arr[idx]
            for cat_col, s_idx, transform in _SCALAR_VALIDATION_COLS:
                if cat_col not in cat_df.columns:
                    raise KeyError(f"Catalog {catalog_path} missing column '{cat_col}'")
                catalog_val = float(row[cat_col])
                scalar_val = float(gw_scalars[s_idx])
                if transform == "direct":
                    expected = catalog_val
                elif transform == "cos":
                    expected = float(np.cos(catalog_val))
                elif transform == "scale_1000":
                    expected = catalog_val / 1000.0
                else:
                    raise ValueError(f"Unknown scalar transform: {transform}")
                if not np.isclose(scalar_val, expected, rtol=_SCALAR_VALIDATION_RTOL, atol=_SCALAR_VALIDATION_ATOL):
                    raise ValueError(
                        f"Scalar mismatch for GW idx {idx} (id={raw_id}, source={source_str}): "
                        f"catalog '{cat_col}' (scalar_idx={s_idx}, transform={transform}) → "
                        f"expected={expected:.8g}, got scalar={scalar_val:.8g}. "
                        f"Check that the correct catalog is configured for source '{source_str}'."
                    )

        metadata[int(idx)] = {"redshift": float(z_map[event_id])}

    return metadata


def _normalize_redshift_bin_config(
    edges: Optional[List[float]],
    labels: Optional[List[str]],
) -> Tuple[List[float], List[str]]:
    """Validate and normalize redshift bin edges and labels.

    Args:
        edges: Redshift bin edges, e.g. ``[0.0, 0.04, 0.065, 0.10]``.
        labels: Optional per-bin labels.  If None, auto-generated from edges.

    Returns:
        (edges, labels) tuple, both lists of equal length (labels one shorter).

    Raises:
        ValueError: If edges/labels are invalid.
    """
    if edges is None or len(edges) < 2:
        raise ValueError("redshift_bin_edges must have at least 2 values")
    edges = [float(e) for e in edges]
    for i in range(len(edges) - 1):
        if edges[i] >= edges[i + 1]:
            raise ValueError(
                f"redshift_bin_edges must be strictly increasing; "
                f"got {edges[i]} >= {edges[i + 1]} at index {i}"
            )
    if labels is None:
        labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
    else:
        labels = [str(lbl) for lbl in labels]
        if len(labels) != len(edges) - 1:
            raise ValueError(
                f"redshift_bin_labels length ({len(labels)}) must be "
                f"one less than edges ({len(edges)})"
            )
    return edges, labels


def _aggregate_redshift_metrics(
    *,
    outcomes: Mapping[Tuple[int, int, int], Mapping[str, Any]],
    gallery_sizes: Sequence[int],
    n_trials: int,
    unique_gw: Sequence[int],
    redshift_metadata: Mapping[int, Mapping[str, Any]],
    bin_edges: List[float],
    bin_labels: List[str],
    method_name: str = "",
) -> List[Dict[str, Any]]:
    """Aggregate per-gallery outcomes by redshift bin.

    Only positive-query GW events present in ``redshift_metadata`` contribute.
    Gallery distractors are not binned by redshift.

    Returns:
        List of per-bin rows, one per (gallery_size, bin_index).
        Bins with zero queries are omitted.
    """
    rows: List[Dict[str, Any]] = []
    for gallery_size in [int(s) for s in gallery_sizes]:
        # Collect GW-level values per redshift bin
        buckets: Dict[int, Dict[str, List[float]]] = {}
        for trial in range(int(n_trials)):
            for gw_id in [int(g) for g in unique_gw]:
                key = (gallery_size, trial, gw_id)
                if key not in outcomes or gw_id not in redshift_metadata:
                    continue
                outcome = outcomes[key]
                z = float(redshift_metadata[gw_id]["redshift"])
                # Determine bin index from edges
                bin_idx = _find_redshift_bin(z, bin_edges)
                if bin_idx < 0:
                    continue
                bucket = buckets.setdefault(
                    bin_idx,
                    {
                        "redshifts": [],
                        "recalls": {1: [], 5: [], 10: []},
                        "mrrs": [],
                        "coverage": [],
                        "full_coverage": [],
                        "actual_sizes": [],
                    },
                )
                metric_contributions = gallery_outcome_metric_contributions(outcome)
                actual_size = float(outcome.get("actual_gallery_size", gallery_size))
                coverage_met = bool(outcome.get("coverage_met", actual_size >= gallery_size))
                fill_ratio = min(1.0, max(0.0, actual_size / float(max(int(gallery_size), 1))))
                bucket["redshifts"].append(z)
                for k in bucket["recalls"]:
                    bucket["recalls"][k].append(metric_contributions[f"recall_at_{k}"])
                bucket["mrrs"].append(metric_contributions["mrr"])
                bucket["coverage"].append(fill_ratio)
                bucket["full_coverage"].append(1.0 if coverage_met else 0.0)
                bucket["actual_sizes"].append(actual_size)

        for bin_idx in sorted(buckets):
            bucket = buckets[bin_idx]
            n = int(len(bucket["mrrs"]))
            rows.append({
                "method": str(method_name),
                "redshift_bin_label": str(bin_labels[bin_idx]),
                "bin_left": float(bin_edges[bin_idx]),
                "bin_right": float(bin_edges[bin_idx + 1]),
                "redshift": float(np.mean(bucket["redshifts"])) if n else 0.0,
                "gallery_size": int(gallery_size),
                "n_queries": n,
                "recall_at_1": float(np.mean(bucket["recalls"][1])) if n else 0.0,
                "recall_at_5": float(np.mean(bucket["recalls"][5])) if n else 0.0,
                "recall_at_10": float(np.mean(bucket["recalls"][10])) if n else 0.0,
                "mrr": float(np.mean(bucket["mrrs"])) if n else 0.0,
                "coverage": float(np.mean(bucket["coverage"])) if n else 0.0,
                "fill_ratio_mean": float(np.mean(bucket["coverage"])) if n else 0.0,
                "full_coverage": float(np.mean(bucket["full_coverage"])) if n else 0.0,
                "effective_gallery_size_mean": float(np.mean(bucket["actual_sizes"])) if n else 0.0,
            })
    return rows


def _find_redshift_bin(z: float, edges: List[float]) -> int:
    """Return the bin index for a redshift value, or -1 if outside all bins."""
    for i in range(len(edges) - 1):
        if edges[i] <= z < edges[i + 1]:
            return i
    # Right-inclusive for the last bin
    if z == edges[-1]:
        return len(edges) - 2
    return -1


# ---------------------------------------------------------------------------
# Redshift CSV and plot writers (adapted from eval_gw170817a_retrieval.py)
# ---------------------------------------------------------------------------
def write_redshift_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Write redshift-binned metrics to a CSV file."""
    import csv

    rows = list(rows)
    if not rows:
        return
    fieldnames = [
        "method",
        "redshift_bin_label",
        "bin_left",
        "bin_right",
        "redshift",
        "gallery_size",
        "n_queries",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "coverage",
        "fill_ratio_mean",
        "full_coverage",
        "effective_gallery_size_mean",
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate_redshift_macro_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate redshift-binned metrics across gallery sizes with log10(G) weights."""
    grouped: Dict[Tuple[str, str, float, float], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["method"]),
            str(row["redshift_bin_label"]),
            float(row["bin_left"]),
            float(row["bin_right"]),
        )
        grouped.setdefault(key, []).append(row)

    macro_rows: List[Dict[str, Any]] = []
    metric_pairs = [
        ("recall_at_1", "macro_recall_at_1"),
        ("recall_at_10", "macro_recall_at_10"),
        ("mrr", "macro_mrr"),
    ]
    for (method, label, bin_left, bin_right), group in grouped.items():
        weighted_rows: List[Tuple[Mapping[str, Any], float]] = []
        for row in group:
            gallery_size = int(row["gallery_size"])
            weight = float(np.log10(gallery_size)) if gallery_size > 1 else 0.0
            if weight > 0.0 and np.isfinite(weight):
                weighted_rows.append((row, weight))
        if not weighted_rows:
            continue
        weight_sum = float(sum(weight for _, weight in weighted_rows))
        out_row: Dict[str, Any] = {
            "method": method,
            "redshift_bin_label": label,
            "bin_left": bin_left,
            "bin_right": bin_right,
            "redshift": float(
                sum(float(row["redshift"]) * weight for row, weight in weighted_rows) / weight_sum
            ),
            "weight_scheme": "log10(gallery_size)",
            "n_gallery_sizes": int(len({int(row["gallery_size"]) for row, _ in weighted_rows})),
            "gallery_weight_sum": weight_sum,
            "n_queries_mean": float(
                sum(float(row.get("n_queries", 0.0)) * weight for row, weight in weighted_rows) / weight_sum
            ),
        }
        for source_key, target_key in metric_pairs:
            out_row[target_key] = float(
                sum(float(row[source_key]) * weight for row, weight in weighted_rows) / weight_sum
            )
        macro_rows.append(out_row)

    return sorted(
        macro_rows,
        key=lambda row: (
            str(row["method"]),
            float(row["bin_left"]),
            float(row["bin_right"]),
        ),
    )


def write_redshift_macro_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Write log10(G)-weighted redshift macro metrics to a CSV file."""
    import csv

    rows = list(rows)
    if not rows:
        return
    fieldnames = [
        "method",
        "redshift_bin_label",
        "bin_left",
        "bin_right",
        "redshift",
        "weight_scheme",
        "n_gallery_sizes",
        "gallery_weight_sum",
        "n_queries_mean",
        "macro_recall_at_1",
        "macro_recall_at_10",
        "macro_mrr",
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_redshift_metrics(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    _plot_method_label_fn=_plot_method_label,
) -> None:
    """Plot R@1, R@10, MRR vs redshift per gallery size.

    Style matches the GW170817A redshift evaluation: one figure per gallery
    size, three panels showing the three metrics against mean bin redshift.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        apply_mnras_style(plt, base_font_size=PLOT_FONT_BASE)
    except Exception:
        return
    rows = list(rows)
    if not rows:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in rows}, key=_plot_method_draw_order)
    method_colors = _plot_method_color_map(methods)
    metrics = [("recall_at_1", "R@1"), ("recall_at_10", "R@10"), ("mrr", "MRR")]
    all_gallery_sizes = sorted({int(row["gallery_size"]) for row in rows})

    label_fn = _plot_method_label_fn or (lambda x: x)

    for gallery_size in all_gallery_sizes:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
        for ax, (key, metric_label) in zip(axes, metrics):
            for method in methods:
                method_rows = sorted(
                    [
                        row
                        for row in rows
                        if str(row["method"]) == method
                        and int(row["gallery_size"]) == gallery_size
                    ],
                    key=lambda row: float(row["redshift"]),
                )
                if method_rows:
                    ax.plot(
                        [float(row["redshift"]) for row in method_rows],
                        [float(row[key]) for row in method_rows],
                        marker="o",
                        linewidth=2,
                        label=label_fn(method),
                        color=method_colors[method],
                    )
            ax.set_xlabel("Redshift")
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
        add_panel_labels_below(axes, fontsize=PLOT_FONT_BASE)
        fig.tight_layout(rect=(0, 0.1, 1, 0.86))
        fig.savefig(str(out / f"redshift_retrieval_metrics_g{gallery_size}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_redshift_macro_metrics(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    _plot_method_label_fn=_plot_method_label,
) -> None:
    """Plot log10(G)-weighted Macro R@1, R@10, and MRR vs redshift."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        apply_mnras_style(plt, base_font_size=PLOT_FONT_BASE)
    except Exception:
        return
    rows = list(rows)
    if not rows:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in rows}, key=_plot_method_draw_order)
    method_colors = _plot_method_color_map(methods)
    metrics = [
        ("macro_recall_at_1", "Macro Recall@1"),
        ("macro_recall_at_10", "Macro Recall@10"),
        ("macro_mrr", "Macro MRR"),
    ]
    label_fn = _plot_method_label_fn or (lambda x: x)

    fig, axes = plt.subplots(1, 3, figsize=RETRIEVAL_TWO_ROW_LEGEND_FIGSIZE, sharex=False, sharey=False)
    for ax, (key, metric_label) in zip(axes, metrics):
        for method in methods:
            method_rows = sorted(
                [row for row in rows if str(row["method"]) == method],
                key=lambda row: float(row["redshift"]),
            )
            if not method_rows:
                continue
            ax.plot(
                [float(row["redshift"]) for row in method_rows],
                [float(row[key]) for row in method_rows],
                marker="o",
                linewidth=2,
                label=label_fn(method),
                color=method_colors[method],
            )
        ax.set_xlabel("Redshift")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = None
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=max(1, min(4, len(labels))),
            frameon=False,
        )
    add_panel_labels_below(axes, y=-0.24, fontsize=PLOT_FONT_BASE)
    layout_top = layout_top_below_legend(fig, legend) if legend is not None else 0.94
    fig.tight_layout(rect=(0, 0.06, 1, layout_top))
    fig.savefig(
        str(out / "redshift_macro_metrics_log10_weighted.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        str(out / "redshift_macro_metrics_log10_weighted.pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def plot_redshift_coverage(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Plot coverage vs redshift per gallery size."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        apply_mnras_style(plt, base_font_size=PLOT_FONT_BASE)
    except Exception:
        return
    rows = list(rows)
    if not rows:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in rows}, key=_plot_method_draw_order)
    method_colors = _plot_method_color_map(methods)
    all_gallery_sizes = sorted({int(row["gallery_size"]) for row in rows})

    for gallery_size in all_gallery_sizes:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for method in methods:
            method_rows = sorted(
                [
                    row
                    for row in rows
                    if str(row["method"]) == method
                    and int(row["gallery_size"]) == gallery_size
                ],
                key=lambda row: float(row["redshift"]),
            )
            if method_rows:
                ax.plot(
                    [float(row["redshift"]) for row in method_rows],
                    [float(row["coverage"]) for row in method_rows],
                    marker="o",
                    linewidth=2,
                    label=method,
                    color=method_colors[method],
                )
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Mean Fill Ratio")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(str(out / f"redshift_coverage_g{gallery_size}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


_PARTIAL_REFRESH_COMPAT_FIELDS = (
    "seed",
    "gallery_sizes",
    "gallery_trials",
    "gallery_candidate_mode",
    "gallery_candidate_time_window_days",
    "gallery_candidate_credible_level_max",
    "gallery_include_undersized",
    "n_neg_samples",
    "negative_sample_strategy",
    "negative_sample_block_rows",
    "negative_sample_shuffle",
    "nonkn_cls_base_field",
    "comparison_window",
)


def _normalized_compat_value(key: str, value: Any) -> Any:
    if key in {"test_data_path", "neg_data_path"}:
        return Path(str(value)).name if value not in (None, "") else None
    if isinstance(value, tuple):
        return list(value)
    return value


def validate_partial_refresh_compatibility(
    *,
    base_output: Mapping[str, Any],
    cfg: Mapping[str, Any],
    configured_model_specs: Sequence[Mapping[str, Any]],
    refresh_model_type: str,
) -> None:
    """Fail closed when a partial refresh does not match its base experiment."""
    base_cfg = base_output.get("config")
    if not isinstance(base_cfg, Mapping):
        raise ValueError("Merge base artifact is missing a config mapping.")

    fields = ["test_data_path", "neg_data_path", "neg_group", *_PARTIAL_REFRESH_COMPAT_FIELDS]
    mismatches = []
    for key in fields:
        if key not in base_cfg or key not in cfg:
            continue
        old = _normalized_compat_value(key, base_cfg.get(key))
        new = _normalized_compat_value(key, cfg.get(key))
        if old != new:
            mismatches.append(f"{key}: base={old!r}, current={new!r}")
    if mismatches:
        raise ValueError("Partial-refresh config mismatch:\n  " + "\n  ".join(mismatches))

    target_specs = [spec for spec in configured_model_specs if str(spec["type"]) == refresh_model_type]
    if len(target_specs) != 1:
        raise ValueError(
            f"Expected exactly one configured model with type={refresh_model_type!r}; got {len(target_specs)}."
        )

    base_models = base_output.get("models")
    if not isinstance(base_models, Mapping):
        raise ValueError("Merge base artifact is missing a models mapping.")
    configured_non_target = {
        str(spec["name"]): str(spec["type"])
        for spec in configured_model_specs
        if str(spec["type"]) != refresh_model_type
    }
    for name, model_type in configured_non_target.items():
        if name not in base_models:
            raise ValueError(f"Merge base artifact is missing non-target model {name!r}.")
        if str(base_models[name].get("type")) != model_type:
            raise ValueError(
                f"Model type mismatch for {name!r}: base={base_models[name].get('type')!r}, "
                f"current={model_type!r}."
            )
    unexpected = [
        name
        for name, result in base_models.items()
        if str(result.get("type")) != refresh_model_type and name not in configured_non_target
    ]
    if unexpected:
        raise ValueError(f"Merge base artifact has unexpected non-target model(s): {unexpected}")


def merge_partial_model_results(
    *,
    base_output: Mapping[str, Any],
    fresh_model_results: Mapping[str, Mapping[str, Any]],
    fresh_curve_rows: Sequence[Mapping[str, Any]],
    fresh_redshift_rows: Sequence[Mapping[str, Any]],
    configured_model_specs: Sequence[Mapping[str, Any]],
    refresh_model_type: str,
) -> Tuple[OrderedDict, OrderedDict, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Replace one model type while preserving all other model results from a base artifact."""
    target_names = [
        str(spec["name"])
        for spec in configured_model_specs
        if str(spec["type"]) == refresh_model_type
    ]
    if len(target_names) != 1:
        raise ValueError(f"Partial refresh requires exactly one target model; got {target_names}.")
    target_name = target_names[0]
    if target_name not in fresh_model_results:
        raise ValueError(f"Fresh results are missing target model {target_name!r}.")

    base_models = base_output["models"]
    combined_models: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    for spec in configured_model_specs:
        name = str(spec["name"])
        if name == target_name:
            combined_models[name] = dict(fresh_model_results[name])
        else:
            combined_models[name] = dict(base_models[name])

    combined_retrieval: OrderedDict[str, Dict[str, float]] = OrderedDict(
        (name, dict(result.get("retrieval", {}))) for name, result in combined_models.items()
    )
    combined_curve_rows = [
        dict(row) for row in base_output.get("curve_rows", []) if str(row.get("method")) != target_name
    ]
    combined_curve_rows.extend(dict(row) for row in fresh_curve_rows)
    combined_redshift_rows = [
        dict(row) for row in base_output.get("redshift_rows", []) if str(row.get("method")) != target_name
    ]
    combined_redshift_rows.extend(dict(row) for row in fresh_redshift_rows)
    return (
        combined_models,
        combined_retrieval,
        combined_curve_rows,
        combined_redshift_rows,
        list(combined_models.keys()),
    )


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def summarize_gallery_identity(
    galleries: Mapping[Tuple[int, int, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    """Hash every non-time candidate identity used by the paired ablation."""
    digest = hashlib.sha256()
    n_negative_assignments = 0
    for key in sorted(galleries):
        spec = galleries[key]
        digest.update(np.asarray(key, dtype="<i8").tobytes())
        digest.update(
            np.asarray(
                [
                    int(spec["positive_index"]),
                    int(spec.get("requested_gallery_size", key[0])),
                    int(spec.get("actual_gallery_size", 1)),
                ],
                dtype="<i8",
            ).tobytes()
        )
        negative_indices = np.asarray(
            spec.get("negative_indices", []), dtype="<i8"
        ).reshape(-1)
        digest.update(negative_indices.tobytes())
        n_negative_assignments += int(negative_indices.size)
        for field in ("negative_synthetic_coordinates", "negative_credible_levels"):
            values = np.asarray(spec.get(field, []), dtype="<f4")
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(values.tobytes())
    return {
        "sha256": digest.hexdigest(),
        "n_galleries": int(len(galleries)),
        "n_negative_assignments": int(n_negative_assignments),
        "fields": [
            "gallery_key",
            "positive_index",
            "requested_gallery_size",
            "actual_gallery_size",
            "negative_indices",
            "negative_synthetic_coordinates",
            "negative_credible_levels",
        ],
        "excludes": ["candidate_time_delta"],
    }


_BASELINE_COMPARISON_FIELDS = (
    "seed",
    "gallery_sizes",
    "gallery_trials",
    "gallery_candidate_mode",
    "gallery_candidate_time_window_days",
    "gallery_candidate_credible_level_max",
    "gallery_include_undersized",
    "positive_selection",
    "max_gw_events",
    "n_neg_samples",
    "negative_sample_strategy",
    "negative_sample_block_rows",
    "negative_sample_shuffle",
    "nonkn_cls_base_field",
    "comparison_window",
)


def validate_time_delta_baseline_compatibility(
    baseline_output: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> None:
    """Require the prior artifact to describe the same deterministic galleries."""
    baseline_cfg = baseline_output.get("config")
    if not isinstance(baseline_cfg, Mapping):
        raise ValueError("Baseline result is missing its config mapping.")
    mismatches = []
    for field in _BASELINE_COMPARISON_FIELDS:
        if field not in baseline_cfg or field not in cfg:
            continue
        old = _normalized_compat_value(field, baseline_cfg.get(field))
        new = _normalized_compat_value(field, cfg.get(field))
        if old != new:
            mismatches.append(f"{field}: baseline={old!r}, current={new!r}")
    if mismatches:
        raise ValueError(
            "Time-delta baseline is not gallery-compatible:\n  "
            + "\n  ".join(mismatches)
        )


_RETRIEVAL_METRIC_RE = re.compile(
    r"^gallery_(?P<gallery_size>\d+)_(?P<metric>recall_at_1|recall_at_5|recall_at_10|mrr)$"
)


def build_time_delta_metric_delta_rows(
    baseline_model: Mapping[str, Any],
    ablated_model: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Build overall and source-stratified paired metric differences."""
    rows: List[Dict[str, Any]] = []
    scopes = [("overall", baseline_model.get("retrieval", {}), ablated_model.get("retrieval", {}))]
    baseline_by_source = baseline_model.get("retrieval_by_source", {})
    ablated_by_source = ablated_model.get("retrieval_by_source", {})
    for source in sorted(set(baseline_by_source) & set(ablated_by_source)):
        scopes.append((str(source), baseline_by_source[source], ablated_by_source[source]))

    for scope, baseline_metrics, ablated_metrics in scopes:
        for key, baseline_value_raw in baseline_metrics.items():
            match = _RETRIEVAL_METRIC_RE.match(str(key))
            if match is None or key not in ablated_metrics:
                continue
            baseline_value = float(baseline_value_raw)
            ablated_value = float(ablated_metrics[key])
            absolute_change = ablated_value - baseline_value
            relative_change = (
                absolute_change / baseline_value if baseline_value != 0.0 else None
            )
            rows.append(
                {
                    "scope": scope,
                    "gallery_size": int(match.group("gallery_size")),
                    "metric": str(match.group("metric")),
                    "baseline": baseline_value,
                    "dt_zero": ablated_value,
                    "absolute_change": absolute_change,
                    "relative_change": relative_change,
                }
            )
    return sorted(rows, key=lambda row: (row["scope"], row["gallery_size"], row["metric"]))


def build_time_delta_redshift_delta_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    ablated_rows: Sequence[Mapping[str, Any]],
    *,
    method_name: str,
) -> List[Dict[str, Any]]:
    """Build paired redshift-bin metric differences for one model."""
    metrics = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")
    baseline_lookup = {
        (str(row["redshift_bin_label"]), int(row["gallery_size"])): row
        for row in baseline_rows
        if str(row.get("method")) == str(method_name)
    }
    output: List[Dict[str, Any]] = []
    for row in ablated_rows:
        if str(row.get("method")) != str(method_name):
            continue
        key = (str(row["redshift_bin_label"]), int(row["gallery_size"]))
        baseline_row = baseline_lookup.get(key)
        if baseline_row is None:
            continue
        for metric in metrics:
            baseline_value = float(baseline_row[metric])
            ablated_value = float(row[metric])
            absolute_change = ablated_value - baseline_value
            output.append(
                {
                    "redshift_bin_label": key[0],
                    "redshift": float(row["redshift"]),
                    "gallery_size": key[1],
                    "metric": metric,
                    "baseline": baseline_value,
                    "dt_zero": ablated_value,
                    "absolute_change": absolute_change,
                    "relative_change": (
                        absolute_change / baseline_value if baseline_value != 0.0 else None
                    ),
                }
            )
    return output


def write_rows_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    import csv

    rows = list(rows)
    if not rows:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_time_delta_ablation(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Plot paired Full-model retrieval curves for baseline and zero-delta input."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        apply_mnras_style(plt, base_font_size=PLOT_FONT_BASE)
    except Exception:
        return
    overall = [row for row in rows if row.get("scope") == "overall"]
    if not overall:
        return
    metrics = [
        ("recall_at_1", "R@1"),
        ("recall_at_5", "R@5"),
        ("recall_at_10", "R@10"),
        ("mrr", "MRR"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        metric_rows = sorted(
            [row for row in overall if row["metric"] == metric],
            key=lambda row: int(row["gallery_size"]),
        )
        gallery_sizes = [int(row["gallery_size"]) for row in metric_rows]
        ax.plot(gallery_sizes, [row["baseline"] for row in metric_rows], marker="o", label="Baseline")
        ax.plot(gallery_sizes, [row["dt_zero"] for row in metric_rows], marker="s", label=r"$\Delta t=0$")
        ax.set_xscale("log")
        ax.set_xlabel("Gallery size")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir = Path(output_dir)
    fig.savefig(output_dir / "time_delta_ablation_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "time_delta_ablation_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_time_delta_redshift_ablation(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    """Plot baseline and zero-delta retrieval versus redshift for every gallery size."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        apply_mnras_style(plt, base_font_size=PLOT_FONT_BASE)
    except Exception:
        return

    rows = list(rows)
    if not rows:
        return
    gallery_sizes = sorted({int(row["gallery_size"]) for row in rows})
    metrics = [
        ("recall_at_1", "R@1"),
        ("recall_at_10", "R@10"),
        ("mrr", "MRR"),
    ]
    baseline_color = "#1f77b4"
    zero_color = "#e07a1f"
    fig, axes = plt.subplots(
        len(gallery_sizes),
        len(metrics),
        figsize=(14, 3.0 * len(gallery_sizes) + 1.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_idx, gallery_size in enumerate(gallery_sizes):
        for col_idx, (metric, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            metric_rows = sorted(
                [
                    row
                    for row in rows
                    if int(row["gallery_size"]) == gallery_size
                    and str(row["metric"]) == metric
                ],
                key=lambda row: float(row["redshift"]),
            )
            redshift = [float(row["redshift"]) for row in metric_rows]
            ax.plot(
                redshift,
                [float(row["baseline"]) for row in metric_rows],
                color=baseline_color,
                linestyle="-",
                marker="o",
                linewidth=2,
                markersize=5,
                label="Baseline",
            )
            ax.plot(
                redshift,
                [float(row["dt_zero"]) for row in metric_rows],
                color=zero_color,
                linestyle="--",
                marker="s",
                markerfacecolor="white",
                linewidth=2,
                markersize=5,
                label=r"$\Delta t=0$",
            )
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(metric_label)
            if col_idx == 0:
                ax.set_ylabel(f"Gallery={gallery_size}\nScore")
            if row_idx == len(gallery_sizes) - 1:
                ax.set_xlabel("Redshift")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.992),
    )
    fig.suptitle(
        "Full-model retrieval versus redshift: baseline and zero time-delta input",
        y=0.965,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=1.0, w_pad=0.8)
    output_dir = Path(output_dir)
    fig.savefig(
        output_dir / "time_delta_ablation_redshift_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "time_delta_ablation_redshift_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _resolve_checkpoint_path(checkpoint_path: str, model_type: str) -> str:
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        return str(path.resolve())
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Unsupported checkpoint path: {checkpoint_path}")

    preferred_names = ["optical_only_best.pth", "best.pth"] if model_type == "optical" else ["albef_best.pth", "best.pth"]
    for filename in preferred_names:
        matches = sorted(path.rglob(filename))
        if matches:
            return str(matches[0].resolve())

    epoch_candidates: List[Tuple[int, Path]] = []
    epoch_patterns = [
        re.compile(r"albef_epoch_(\d+)\.pth$"),
        re.compile(r"optical_only_epoch_(\d+)\.pth$"),
    ]
    for candidate in path.rglob("*.pth"):
        for pattern in epoch_patterns:
            match = pattern.fullmatch(candidate.name)
            if match:
                epoch_candidates.append((int(match.group(1)), candidate))
                break
    if epoch_candidates:
        return str(max(epoch_candidates, key=lambda item: item[0])[1].resolve())

    fallback_names = ["optical_only_last.pth", "albef_last.pth", "last.pth"]
    for filename in fallback_names:
        matches = sorted(path.rglob(filename))
        if matches:
            return str(matches[0].resolve())

    any_pth = sorted(path.rglob("*.pth"))
    if any_pth:
        return str(any_pth[0].resolve())

    raise FileNotFoundError(f"No checkpoint file found under directory: {checkpoint_path}")


def _build_model_specs(cfg: Dict[str, Any], cfg_dir: Path) -> List[Dict[str, Any]]:
    return build_comparison_model_specs(cfg, cfg_dir)


def load_optical_model(checkpoint_path: str, device: torch.device) -> Tuple[OpticalKNClassifier, Dict[str, Any]]:
    """Load optical-only model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = vars(ckpt_args)

    model = OpticalKNClassifier(
        optical_input_dim=int(ckpt_args.get("optical_input_dim", 6)),
        ref_time_dim=int(ckpt_args.get("ref_dim", ckpt_args.get("n_ref", 64))),
        enc_dim=int(ckpt_args.get("enc_dim", 128)),
        optical_curve_dim=ckpt_args.get("optical_curve_dim"),
        optical_curve_hidden_dim=ckpt_args.get("optical_curve_hidden_dim"),
        num_heads=int(ckpt_args.get("num_heads", 4)),
        k_dim=int(ckpt_args.get("k_dim", 64)),
        opt_dropout=0.0,
        feature_dropout=0.0,
        head_hidden_dim=ckpt_args.get("head_hidden_dim"),
        head_dropout=0.0,
        universal_aux_enable=bool(ckpt_args.get("universal_aux_enable", False)),
        proj_dim=int(ckpt_args.get("proj_dim", 64)),
        grl_lambda=float(ckpt_args.get("grl_lambda", 1.0)),
        mtan_snr_s0=float(ckpt_args.get("mtan_snr_s0", 3.0)),
        mtan_snr_beta=float(ckpt_args.get("mtan_snr_beta", 1.0)),
        mtan_snr_clip_min=float(ckpt_args.get("mtan_snr_clip_min", -8.0)),
        mtan_snr_clip_max=float(ckpt_args.get("mtan_snr_clip_max", 20.0)),
        mtan_snr_eps=float(ckpt_args.get("mtan_snr_eps", 1e-9)),
        mtan_lupt_psfflux_zp=float(ckpt_args.get("mtan_lupt_psfflux_zp", 31.4)),
        mtan_lupt_k=float(ckpt_args.get("mtan_lupt_k", 1.0)),
        mtan_lupt_m5_mag=_parse_float_sequence(
            ckpt_args.get("mtan_lupt_m5_mag", (23.9, 25.0, 24.7, 24.0, 23.3, 22.1)),
            expected_len=6,
            name="mtan_lupt_m5_mag",
        ),
        mtan_period_range_days=_parse_float_sequence(
            ckpt_args.get("mtan_period_range_days", (0.5, 100.0)),
            expected_len=2,
            name="mtan_period_range_days",
        ),
        mtan_time_scale_divisor=float(ckpt_args.get("mtan_time_scale_divisor", 100.0)),
    )

    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict = migrate_time_embed_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"  Loaded optical-only model from {checkpoint_path}")
    return model, ckpt_args


def load_multimodal_bundle(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
    *,
    test_data_path: str,
    neg_data_path: Optional[str],
    comparison_window: Tuple[float, float],
    nonkn_cls_base_field: str,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Load multimodal model and build runtime eval args for comparison."""
    args_ns = SimpleNamespace(
        checkpoint=checkpoint_path,
        config=config_path,
        test_data_path=test_data_path,
        nonkn_cls_base_field=nonkn_cls_base_field,
    )
    model, model_args, saved_args = load_model(args_ns, device)
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    saved_args = dict(saved_args or {})

    original_ref_start = float(model_args.get("ref_start", comparison_window[0]))
    original_ref_end = float(model_args.get("ref_end", comparison_window[1]))
    dataset_window_start, dataset_window_end = _read_root_time_window_attrs(test_data_path)

    runtime_model_args = dict(model_args)
    runtime_model_args["original_ref_start"] = original_ref_start
    runtime_model_args["original_ref_end"] = original_ref_end
    runtime_model_args["ref_start"] = float(comparison_window[0])
    runtime_model_args["ref_end"] = float(comparison_window[1])
    runtime_model_args["nonkn_cls_base_field"] = str(nonkn_cls_base_field)
    runtime_model_args["dataset_window_metadata"] = {
        "positive": _read_root_time_window_attrs(test_data_path),
        "negative": _read_root_time_window_attrs(neg_data_path),
    }
    runtime_model_args["effective_input_window_metadata"] = build_effective_input_window_metadata(
        original_ref_start,
        original_ref_end,
        runtime_input_window_start=float(comparison_window[0]),
        runtime_input_window_end=float(comparison_window[1]),
        dataset_window_start=dataset_window_start,
        dataset_window_end=dataset_window_end,
    )
    return model, runtime_model_args, saved_args



def _multimodal_has_cls_head(saved_args: Dict[str, Any]) -> bool:
    return (float(saved_args.get("cls_weight", 0.0) or 0.0) > 0.0
            or float(saved_args.get("gallery_loss_weight", 0.0) or 0.0) > 0.0)


def _resolve_multimodal_scoring(model_spec: Dict[str, Any], saved_args: Dict[str, Any]) -> str:
    requested = str(model_spec.get("scoring", "auto")).strip().lower()
    has_cls_head = _multimodal_has_cls_head(saved_args)
    if not has_cls_head:
        return "contrastive"
    if requested in {"", "auto"}:
        return "logits"
    return requested


def candidate_time_delta_scoring_kwargs(
    scoring_mode: str,
    candidate_time_delta_mode: str,
) -> Dict[str, str]:
    """Forward the ablation only to the fusion-logit retrieval scorer."""
    mode = normalize_candidate_time_delta_mode(candidate_time_delta_mode)
    return {} if str(scoring_mode).strip().lower() == "contrastive" else {
        "candidate_time_delta_mode": mode
    }


def _build_test_loader(
    test_data_path: str,
    neg_data_path: Optional[str],
    neg_group: str,
    ref_start: float,
    ref_end: float,
    *,
    batch_size: int,
    test_steps: Optional[int],
    num_workers: int,
    target_samples: Optional[int] = None,
    nonkn_cls_base_field: str = "zero_time_mjd_base",
    return_zero_time_mjd: bool = False,
):
    """Build comparison dataloader using shared runtime crop."""
    dataset = RelationalHDF5Dataset(
        test_data_path,
        negative_h5_path=neg_data_path,
        negative_group=neg_group,
        return_zero_time_mjd=bool(return_zero_time_mjd),
        nonkn_cls_base_field=str(nonkn_cls_base_field),
        opt_input_window_start=ref_start,
        opt_input_window_end=ref_end,
    )
    gw_to_lc = build_gw_to_lc_mapping(test_data_path)
    n_gw = len(gw_to_lc)
    bs = min(int(batch_size), n_gw)
    if bs <= 0:
        raise ValueError("No GW events found in test dataset")

    if test_steps is None:
        if target_samples is None:
            steps = 5
        else:
            steps = max(1, (int(target_samples) + bs - 1) // bs)
    else:
        steps = int(test_steps)

    sampler = BalancedGWBatchedSampler(gw_to_lc, batch_size=bs, steps_per_epoch=steps)
    loader = _build_dataloader(
        dataset,
        sampler,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
        prefetch_factor=2 if int(num_workers) > 0 else None,
    )
    print(f"  Test sampling: {steps} steps x {bs} batch_size = ~{steps * bs} samples")
    return loader


def cache_test_batches(loader):
    """Iterate loader once, cache all batch raw tensors on CPU."""
    cached = []
    for batch_data in tqdm(loader, desc="  Caching test batches"):
        cached.append(tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in batch_data))
    return cached


def build_cached_batches(
    *,
    test_data_path: str,
    neg_data_path: Optional[str],
    neg_group: str,
    comparison_window: Tuple[float, float],
    batch_size: int,
    test_steps: Optional[int],
    num_workers: int,
    target_samples: Optional[int],
    nonkn_cls_base_field: str,
) -> Tuple[List[Tuple[Any, ...]], int]:
    """Build and cache retrieval batches, retrying with single-worker if needed."""
    try_num_workers = int(num_workers)
    while True:
        try:
            _seed_all(42)
            loader = _build_test_loader(
                test_data_path,
                neg_data_path,
                neg_group,
                comparison_window[0],
                comparison_window[1],
                batch_size=batch_size,
                test_steps=test_steps,
                num_workers=try_num_workers,
                target_samples=target_samples,
                nonkn_cls_base_field=nonkn_cls_base_field,
                return_zero_time_mjd=True,
            )
            return cache_test_batches(loader), try_num_workers
        except PermissionError:
            if try_num_workers <= 0:
                raise
            print("WARNING: DataLoader multiprocessing failed. Retrying retrieval cache with num_workers=0.")
            try_num_workers = 0


def load_test_positive_index_map(test_data_path: str) -> Tuple[Dict[int, np.ndarray], List[int]]:
    with h5py.File(test_data_path, "r") as f:
        if "events/optical_data/parent_gw_idx" not in f:
            raise KeyError(f"Missing required field 'events/optical_data/parent_gw_idx' in {test_data_path}")
        parent_gw_idx = np.asarray(f["events/optical_data/parent_gw_idx"][:], dtype=np.int64)

    gw_positive_indices: Dict[int, List[int]] = {}
    for opt_idx, gw_id in enumerate(parent_gw_idx.tolist()):
        gw_positive_indices.setdefault(int(gw_id), []).append(int(opt_idx))

    final_map = {
        int(gw_id): np.asarray(indices, dtype=np.int64)
        for gw_id, indices in gw_positive_indices.items()
    }
    return final_map, sorted(final_map.keys())


def load_selected_positive_bank(
    test_data_path: str,
    selected_optical_indices: np.ndarray,
    *,
    runtime_input_window_start: Optional[float] = None,
    runtime_input_window_end: Optional[float] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[int, int]]:
    selected = np.unique(np.asarray(selected_optical_indices, dtype=np.int64).reshape(-1))
    if selected.size == 0:
        raise ValueError("selected_optical_indices must be non-empty to build a positive query bank.")

    with h5py.File(test_data_path, "r") as f:
        if "events/optical_data" not in f:
            raise KeyError(f"Missing required group 'events/optical_data' in {test_data_path}")
        opt = f["events/optical_data"]
        required_fields = ["times", "values", "masks", "errors", "coordinates", "parent_gw_idx"]
        for field in required_fields:
            if field not in opt:
                raise KeyError(f"Missing required field 'events/optical_data/{field}' in {test_data_path}")

        opt_t = torch.from_numpy(np.asarray(opt["times"][selected], dtype=np.float32))
        opt_v = torch.from_numpy(np.asarray(opt["values"][selected], dtype=np.float32))
        opt_mask = torch.from_numpy(np.asarray(opt["masks"][selected], dtype=np.float32))
        opt_err = torch.from_numpy(np.asarray(opt["errors"][selected], dtype=np.float32))
        opt_coords = torch.from_numpy(np.asarray(opt["coordinates"][selected], dtype=np.float32))
        gw_indices = torch.from_numpy(np.asarray(opt["parent_gw_idx"][selected], dtype=np.int64))

        if "first_detection_mjd" in opt:
            first_detection_mjd = torch.from_numpy(np.asarray(opt["first_detection_mjd"][selected], dtype=np.float32))
        else:
            first_detection_mjd = torch.from_numpy(np.asarray(opt["zero_time_mjd_base"][selected], dtype=np.float32))

    apply_window = (runtime_input_window_start is not None) and (runtime_input_window_end is not None)
    if apply_window:
        opt_t, opt_v, opt_mask, opt_err, _ = apply_runtime_input_window_torch(
            opt_t, opt_v, opt_mask, opt_err,
            window_start=runtime_input_window_start,
            window_end=runtime_input_window_end,
        )

    bank: Dict[str, torch.Tensor] = {
        "times": opt_t,
        "values": opt_v,
        "masks": opt_mask,
        "errors": opt_err,
        "coordinates": opt_coords,
        "gw_indices": gw_indices,
        "source_optical_indices": torch.from_numpy(selected.astype(np.int64, copy=False)),
        "opt_t_raw": opt_t,
        "opt_v_raw": opt_v,
        "opt_mask_raw": opt_mask,
        "opt_err_raw": opt_err,
        "opt_coords": opt_coords,
        "first_detection_mjd": first_detection_mjd,
    }
    remap = {int(source_idx): int(compact_idx) for compact_idx, source_idx in enumerate(selected.tolist())}
    return bank, remap


def build_batch_positive_bank_and_galleries(
    cached_batches,
    candidate_sequences,
    gallery_sizes,
    n_trials,
    include_undersized,
    seed: int = 42,
):
    """Build positive bank from Phase 1 batch samples (batch-based mode).

    The bank preserves every sampled row in the cached batches. Galleries then
    draw positives from those sampled rows, matching the original batch-sampled
    retrieval evaluation instead of the full-test positive pool.

    Returns:
        bank: Dict[str, torch.Tensor] — same format as load_selected_positive_bank
        galleries: Dict keyed by (gallery_size, trial, gw_id)
        unique_gw: List of GW ids present in the bank
    """
    # --- Keep all sampled positive rows from cached batches ---
    opt_t_parts: List[torch.Tensor] = []
    opt_v_parts: List[torch.Tensor] = []
    opt_mask_parts: List[torch.Tensor] = []
    opt_err_parts: List[torch.Tensor] = []
    opt_coords_parts: List[torch.Tensor] = []
    gw_idx_parts: List[torch.Tensor] = []
    first_det_mjd_parts: List[torch.Tensor] = []
    for batch_data in cached_batches:
        opt_t_parts.append(batch_data[2].float().cpu())
        opt_v_parts.append(batch_data[3].float().cpu())
        opt_mask_parts.append(batch_data[4].float().cpu())
        opt_err_parts.append(batch_data[5].float().cpu())
        opt_coords_parts.append(batch_data[6].float().cpu())
        gw_idx_parts.append(batch_data[7].long().cpu())
        if len(batch_data) >= 10 and batch_data[9] is not None:
            first_det_mjd_parts.append(batch_data[9].float().cpu())

    if not gw_idx_parts:
        raise ValueError("cached_batches is empty; cannot build batch-sampled positive bank.")

    bank: Dict[str, torch.Tensor] = {
        "times": torch.cat(opt_t_parts, dim=0),
        "values": torch.cat(opt_v_parts, dim=0),
        "masks": torch.cat(opt_mask_parts, dim=0),
        "errors": torch.cat(opt_err_parts, dim=0),
        "coordinates": torch.cat(opt_coords_parts, dim=0),
        "gw_indices": torch.cat(gw_idx_parts, dim=0),
    }
    n_pos = int(bank["gw_indices"].shape[0])
    bank["source_optical_indices"] = torch.arange(n_pos, dtype=torch.long)
    bank["opt_t_raw"] = bank["times"]
    bank["opt_v_raw"] = bank["values"]
    bank["opt_mask_raw"] = bank["masks"]
    bank["opt_err_raw"] = bank["errors"]
    bank["opt_coords"] = bank["coordinates"]

    if first_det_mjd_parts:
        bank["first_detection_mjd"] = torch.cat(first_det_mjd_parts, dim=0)

    gw_to_bank_indices: Dict[int, List[int]] = {}
    for bank_idx, gw_id in enumerate(bank["gw_indices"].tolist()):
        gw_to_bank_indices.setdefault(int(gw_id), []).append(int(bank_idx))
    gw_to_bank_arrays = {
        gw_id: np.asarray(indices, dtype=np.int64)
        for gw_id, indices in gw_to_bank_indices.items()
    }

    # --- Build galleries referencing compact bank indices ---
    galleries: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    used_gw = sorted(gw_to_bank_arrays.keys())

    for trial in range(int(n_trials)):
        trial_rng = np.random.default_rng(int(seed) + 15485863 * int(trial))
        for gw_id in used_gw:
            gw_bank_indices = gw_to_bank_arrays[int(gw_id)]
            positive_index = int(gw_bank_indices[trial_rng.integers(gw_bank_indices.size)])
            seq = candidate_sequences.get((int(trial), int(gw_id)), {})
            neg_idx_full = np.asarray(seq.get("candidate_indices", []), dtype=np.int64).reshape(-1)
            neg_cred_full = np.asarray(seq.get("credible_levels", []), dtype=np.float32).reshape(-1)
            neg_dt_full = np.asarray(seq.get("abs_dt_days", []), dtype=np.float32).reshape(-1)
            neg_coords_full = None
            if "synthetic_coordinates" in seq:
                neg_coords_full = np.asarray(seq["synthetic_coordinates"], dtype=np.float32).reshape(-1, 2)
            neg_time_full = None
            if "synthetic_zero_time_mjd_cls_base" in seq:
                neg_time_full = np.asarray(seq["synthetic_zero_time_mjd_cls_base"], dtype=np.float64).reshape(-1)

            for gallery_size in gallery_sizes:
                requested = int(gallery_size)
                target_neg = max(requested - 1, 0)
                available_neg = int(neg_idx_full.shape[0])
                if target_neg > available_neg and not bool(include_undersized):
                    continue
                take_neg = min(target_neg, available_neg)
                actual_gallery_size = 1 + int(take_neg)
                gallery_spec = {
                    "positive_index": positive_index,
                    "negative_indices": neg_idx_full[:take_neg].astype(np.int64, copy=False),
                    "negative_credible_levels": neg_cred_full[:take_neg].astype(np.float32, copy=False),
                    "negative_abs_dt_days": neg_dt_full[:take_neg].astype(np.float32, copy=False),
                    "requested_gallery_size": requested,
                    "actual_gallery_size": actual_gallery_size,
                    "coverage_met": bool(actual_gallery_size >= requested),
                    "is_undersized": bool(actual_gallery_size < requested),
                }
                if neg_coords_full is not None:
                    gallery_spec["negative_synthetic_coordinates"] = neg_coords_full[:take_neg].astype(np.float32, copy=False)
                if neg_time_full is not None:
                    gallery_spec["negative_synthetic_zero_time_mjd_cls_base"] = neg_time_full[:take_neg].astype(np.float64, copy=False)
                galleries[(requested, int(trial), int(gw_id))] = gallery_spec

    return bank, galleries, used_gw


def remap_gallery_positive_indices(
    galleries: Mapping[Tuple[int, int, int], Mapping[str, Any]],
    positive_index_remap: Mapping[int, int],
) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    remapped: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for key, gallery_spec in galleries.items():
        source_positive_index = int(gallery_spec["positive_index"])
        if source_positive_index not in positive_index_remap:
            raise KeyError(f"Gallery positive index {source_positive_index} missing from compact positive-bank remap.")
        updated_spec = dict(gallery_spec)
        updated_spec["source_positive_index"] = source_positive_index
        updated_spec["positive_index"] = int(positive_index_remap[source_positive_index])
        remapped[key] = updated_spec
    return remapped


@torch.no_grad()
def extract_gallery_embeddings(
    model,
    cached_batches,
    device,
    *,
    n_ref: int,
    ref_start: float,
    ref_end: float,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
):
    """Extract optical-side embeddings and raw tensors for gallery scoring."""
    all_h_l = []
    all_z_l = []
    all_opt_coords = []
    all_gw_indices = []
    all_opt_t = []
    all_opt_v = []
    all_opt_mask = []
    all_opt_err = []
    ref_cache = None

    for batch_data in tqdm(cached_batches, desc="  Extracting embeddings"):
        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data[:8]

        gw_s = gw_s.to(device)
        gw_m = gw_m.to(device)
        opt_t = opt_t.to(device)
        opt_v = opt_v.to(device)
        opt_mask = opt_mask.to(device)
        opt_err = opt_err.to(device)
        opt_coords = opt_coords.to(device)

        bs = gw_s.size(0)
        if ref_cache is None or ref_cache.shape[0] != bs or ref_cache.dtype != opt_t.dtype:
            ref_cache = build_ref_time(bs, n_ref, ref_start, ref_end, device, opt_t.dtype)

        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            _g, z_l, h_l, _H_gw = model.encode(
                gw_s, gw_m, opt_coords, opt_t, opt_v, ref_cache, opt_mask, opt_err
            )

        all_h_l.append(h_l.float().cpu())
        all_z_l.append(z_l.float().cpu())
        all_opt_coords.append(opt_coords.float().cpu())
        all_gw_indices.append(gw_indices.long().cpu())
        all_opt_t.append(opt_t.float().cpu())
        all_opt_v.append(opt_v.float().cpu())
        all_opt_mask.append(opt_mask.float().cpu())
        all_opt_err.append(opt_err.float().cpu())

    return {
        "h_l_cls": torch.cat(all_h_l),
        "z_l_cls": torch.cat(all_z_l),
        "opt_coords": torch.cat(all_opt_coords),
        "gw_indices": torch.cat(all_gw_indices),
        "opt_t_raw": torch.cat(all_opt_t),
        "opt_v_raw": torch.cat(all_opt_v),
        "opt_mask_raw": torch.cat(all_opt_mask),
        "opt_err_raw": torch.cat(all_opt_err),
        "dual_fusion": _is_dual_fusion_model(model),
    }


@torch.no_grad()
def extract_optical_candidate_embeddings(
    model,
    optical_data,
    device,
    *,
    n_ref: int,
    ref_start: float,
    ref_end: float,
    chunk_size: int = 1024,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
    desc: str = "  Extracting optical candidate embeddings",
):
    """Extract optical embeddings for a preloaded optical candidate bank."""
    core_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    all_h_l = []
    all_z_l = []
    all_opt_coords = []
    all_opt_t = []
    all_opt_v = []
    all_opt_mask = []
    all_opt_err = []

    total = int(optical_data["times"].shape[0])
    for start in tqdm(range(0, total, int(chunk_size)), desc=desc):
        end = min(start + int(chunk_size), total)
        chunk_idx = torch.arange(start, end, dtype=torch.long)
        opt_coords = optical_data["coordinates"].index_select(0, chunk_idx).to(device)
        opt_t = optical_data["times"].index_select(0, chunk_idx).to(device)
        opt_v = optical_data["values"].index_select(0, chunk_idx).to(device)
        opt_mask = optical_data["masks"].index_select(0, chunk_idx).to(device)
        opt_err = optical_data["errors"].index_select(0, chunk_idx).to(device)
        ref_time = build_ref_time(opt_t.size(0), n_ref, ref_start, ref_end, device, opt_t.dtype)
        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            z_l, h_l = core_model.encode_optical(
                opt_coords, opt_t, opt_v, ref_time, opt_mask, opt_err
            )

        all_h_l.append(h_l.float().cpu())
        all_z_l.append(z_l.float().cpu())
        all_opt_coords.append(opt_coords.float().cpu())
        all_opt_t.append(opt_t.float().cpu())
        all_opt_v.append(opt_v.float().cpu())
        all_opt_mask.append(opt_mask.float().cpu())
        all_opt_err.append(opt_err.float().cpu())

    result = {
        "h_l_cls": torch.cat(all_h_l),
        "z_l_cls": torch.cat(all_z_l),
        "opt_coords": torch.cat(all_opt_coords),
        "opt_t_raw": torch.cat(all_opt_t),
        "opt_v_raw": torch.cat(all_opt_v),
        "opt_mask_raw": torch.cat(all_opt_mask),
        "opt_err_raw": torch.cat(all_opt_err),
    }
    if "gw_indices" in optical_data:
        gw_indices = optical_data["gw_indices"]
        result["gw_indices"] = (
            gw_indices.detach().cpu().to(torch.long)
            if isinstance(gw_indices, torch.Tensor)
            else torch.as_tensor(gw_indices, dtype=torch.long)
        )
    if "source_optical_indices" in optical_data:
        source_indices = optical_data["source_optical_indices"]
        result["source_optical_indices"] = (
            source_indices.detach().cpu().to(torch.long)
            if isinstance(source_indices, torch.Tensor)
            else torch.as_tensor(source_indices, dtype=torch.long)
        )
    if "first_detection_mjd" in optical_data:
        result["first_detection_mjd"] = optical_data["first_detection_mjd"]
    return result


def precompute_tutorial_galleries(
    gw_positive_indices: Mapping[int, Any],
    n_tutorial_negatives: int,
    gallery_sizes,
    n_trials,
    seed,
):
    """Build galleries with one matched KN and tutorial non-KN distractors.

    For each (trial, gw_id) the positive is sampled once and reused across
    all gallery sizes.  Negative distractors are generated as one random
    permutation per (trial, gw_id); larger galleries use the prefix of
    that permutation so that a small gallery is always a subset of a
    larger gallery for the same (trial, gw_id).
    """
    base_rng = np.random.default_rng(int(seed))
    unique_gw = sorted(int(gw_id) for gw_id in gw_positive_indices.keys())
    max_neg_pool = int(n_tutorial_negatives)
    max_neg_needed = max(int(s) - 1 for s in gallery_sizes)
    max_neg_needed = max(0, min(max_neg_needed, max_neg_pool))
    galleries: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    gallery_sizes = [int(s) for s in gallery_sizes]

    for trial in range(int(n_trials)):
        for gw_id in unique_gw:
            gw_idxs = np.asarray(gw_positive_indices.get(int(gw_id), []), dtype=np.int64).reshape(-1)
            if gw_idxs.size == 0:
                raise ValueError(
                    f"GW event {gw_id} has no positive optical samples available for gallery construction."
                )
            correct_idx = int(gw_idxs[base_rng.integers(gw_idxs.size)])
            # One negative permutation per (trial, gw_id)
            if max_neg_needed > 0:
                trial_neg_order = base_rng.choice(max_neg_pool, size=max_neg_needed, replace=False)
            else:
                trial_neg_order = np.empty((0,), dtype=np.int64)

            for gallery_size in gallery_sizes:
                requested = int(gallery_size)
                if requested <= 1:
                    galleries[(requested, int(trial), int(gw_id))] = {
                        "positive_index": correct_idx,
                        "negative_indices": np.empty((0,), dtype=np.int64),
                        "negative_credible_levels": np.empty((0,), dtype=np.float32),
                        "negative_abs_dt_days": np.empty((0,), dtype=np.float32),
                        "requested_gallery_size": requested,
                        "actual_gallery_size": 1,
                        "coverage_met": True,
                        "is_undersized": False,
                    }
                    continue

                take_neg = min(requested - 1, max_neg_needed)
                if take_neg <= 0:
                    continue
                actual_size = 1 + int(take_neg)
                galleries[(requested, int(trial), int(gw_id))] = {
                    "positive_index": correct_idx,
                    "negative_indices": trial_neg_order[:take_neg].astype(np.int64, copy=False),
                    "negative_credible_levels": np.full((take_neg,), np.nan, dtype=np.float32),
                    "negative_abs_dt_days": np.full((take_neg,), np.nan, dtype=np.float32),
                    "requested_gallery_size": requested,
                    "actual_gallery_size": actual_size,
                    "coverage_met": bool(actual_size >= requested),
                    "is_undersized": bool(actual_size < requested),
                }

    return galleries, unique_gw


def enrich_gallery_outcomes(ranks: Mapping[Tuple[int, int, int], int], galleries: Mapping[Tuple[int, int, int], Mapping[str, Any]]) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    outcomes: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for key, rank in ranks.items():
        gallery_spec = galleries[key]
        outcomes[key] = {
            "rank": float(rank),
            "requested_gallery_size": int(gallery_spec.get("requested_gallery_size", key[0])),
            "actual_gallery_size": int(gallery_spec.get("actual_gallery_size", key[0])),
            "coverage_met": bool(gallery_spec.get("coverage_met", True)),
            "is_undersized": bool(gallery_spec.get("is_undersized", False)),
        }
    return outcomes


@torch.no_grad()
def score_gallery_optical_only(
    opt_model: OpticalKNClassifier,
    candidate_indices: np.ndarray,
    opt_t_all: torch.Tensor,
    opt_v_all: torch.Tensor,
    opt_mask_all: torch.Tensor,
    opt_err_all: torch.Tensor,
    device: torch.device,
    *,
    n_ref: int,
    ref_start: float,
    ref_end: float,
    chunk_size: int = 1024,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
) -> np.ndarray:
    """Score gallery candidates using optical-only model p(KN|optical)."""
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.float32)

    scores = []
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    for start in range(0, len(candidate_indices), int(chunk_size)):
        chunk_np = candidate_indices[start:start + int(chunk_size)]
        chunk_idx = torch.from_numpy(chunk_np).long()
        batch_size = len(chunk_np)

        opt_t = opt_t_all.index_select(0, chunk_idx).to(device)
        opt_v = opt_v_all.index_select(0, chunk_idx).to(device)
        opt_mask = opt_mask_all.index_select(0, chunk_idx).to(device)
        opt_err = opt_err_all.index_select(0, chunk_idx).to(device)
        ref_time = build_ref_time(batch_size, n_ref, ref_start, ref_end, device, opt_t.dtype)

        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            logits = opt_model(opt_t, opt_v, ref_time, opt_mask, opt_err).squeeze(-1)

        scores.append(torch.sigmoid(logits.float()).cpu())

    return torch.cat(scores).numpy()



@torch.no_grad()
def extract_negative_gallery_embeddings(
    model,
    neg_optical_data,
    device,
    *,
    n_ref: int,
    ref_start: float,
    ref_end: float,
    chunk_size: int = 1024,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
):
    """Extract optical embeddings for tutorial negative distractors."""
    if neg_optical_data is None:
        raise ValueError("Tutorial negative optical data is required for gallery distractors.")

    core_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    all_h_l = []
    all_z_l = []
    all_opt_coords = []
    all_opt_t = []
    all_opt_v = []
    all_opt_mask = []
    all_opt_err = []
    all_z_curve = []

    optical_encoder = getattr(core_model, "optical_encoder", None)
    supports_curve_coord_cache = (
        optical_encoder is not None
        and hasattr(optical_encoder, "encode_components")
        and hasattr(optical_encoder, "contrastive_head")
    )
    total = int(neg_optical_data["times"].shape[0])
    for start in tqdm(range(0, total, int(chunk_size)), desc="  Extracting tutorial distractor embeddings"):
        end = min(start + int(chunk_size), total)
        chunk_idx = torch.arange(start, end, dtype=torch.long)
        opt_coords = neg_optical_data["coordinates"].index_select(0, chunk_idx).to(device)
        opt_t = neg_optical_data["times"].index_select(0, chunk_idx).to(device)
        opt_v = neg_optical_data["values"].index_select(0, chunk_idx).to(device)
        opt_mask = neg_optical_data["masks"].index_select(0, chunk_idx).to(device)
        opt_err = neg_optical_data["errors"].index_select(0, chunk_idx).to(device)
        ref_time = build_ref_time(opt_t.size(0), n_ref, ref_start, ref_end, device, opt_t.dtype)
        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            if supports_curve_coord_cache:
                z_curve, coord_feat, h_l = optical_encoder.encode_components(
                    opt_coords, opt_t, opt_v, ref_time, opt_mask, opt_err=opt_err
                )
                z_l = optical_encoder.contrastive_head(z_curve, coord_feat)
            else:
                z_curve = None
                z_l, h_l = core_model.encode_optical(
                    opt_coords, opt_t, opt_v, ref_time, opt_mask, opt_err
                )

        all_h_l.append(h_l.float().cpu())
        all_z_l.append(z_l.float().cpu())
        if z_curve is not None:
            all_z_curve.append(z_curve.float().cpu())
        all_opt_coords.append(opt_coords.float().cpu())
        all_opt_t.append(opt_t.float().cpu())
        all_opt_v.append(opt_v.float().cpu())
        all_opt_mask.append(opt_mask.float().cpu())
        all_opt_err.append(opt_err.float().cpu())

    result = {
        "h_l_cls": torch.cat(all_h_l),
        "z_l_cls": torch.cat(all_z_l),
        "opt_coords": torch.cat(all_opt_coords),
        "opt_t_raw": torch.cat(all_opt_t),
        "opt_v_raw": torch.cat(all_opt_v),
        "opt_mask_raw": torch.cat(all_opt_mask),
        "opt_err_raw": torch.cat(all_opt_err),
    }
    if all_z_curve:
        result["z_curve_cls"] = torch.cat(all_z_curve)
    return result


def _candidate_coords_for_chunk(
    candidate_bank: Dict[str, torch.Tensor],
    chunk_idx: torch.Tensor,
    candidate_coords: Optional[np.ndarray],
    *,
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    if candidate_coords is None:
        return candidate_bank["opt_coords"].index_select(0, chunk_idx).to(device)
    coords_np = np.asarray(candidate_coords, dtype=np.float32)
    if coords_np.ndim != 2 or coords_np.shape[1] != 2:
        raise ValueError(f"candidate_coords must have shape [N, 2], got {coords_np.shape}")
    return torch.as_tensor(coords_np[int(start):int(end)], dtype=torch.float32, device=device)


def _candidate_z_l_for_chunk(
    model,
    candidate_bank: Dict[str, torch.Tensor],
    chunk_idx: torch.Tensor,
    opt_coords_chunk: torch.Tensor,
    device: torch.device,
    *,
    use_synthetic_coords: bool,
) -> torch.Tensor:
    if use_synthetic_coords and "z_curve_cls" in candidate_bank:
        core_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        optical_encoder = getattr(core_model, "optical_encoder", None)
        if (
            optical_encoder is not None
            and hasattr(optical_encoder, "encode_coord_only")
            and hasattr(optical_encoder, "contrastive_head")
        ):
            z_curve = candidate_bank["z_curve_cls"].index_select(0, chunk_idx).to(device)
            coord_feat = optical_encoder.encode_coord_only(opt_coords_chunk)
            return optical_encoder.contrastive_head(z_curve, coord_feat)
    return candidate_bank["z_l_cls"].index_select(0, chunk_idx).to(device)


@torch.no_grad()
def _score_candidate_bank_with_logits(
    model,
    query_cache,
    candidate_indices: np.ndarray,
    candidate_bank: Dict[str, torch.Tensor],
    device: torch.device,
    dual: bool,
    *,
    candidate_coords: Optional[np.ndarray] = None,
    candidate_abs_dt_days: Optional[np.ndarray] = None,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
) -> np.ndarray:
    """Score a query GW against an optical candidate bank with fusion logits."""
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.float32)

    need_time_delta = _model_requires_time_delta(model)
    if need_time_delta and candidate_abs_dt_days is None:
        raise ValueError(
            "Model requires time_delta_cls_feature but candidate_abs_dt_days is None. "
            "The negative pool must provide time fields (zero_time_mjd_cls_base or "
            "negative_synthetic_zero_time_mjd_cls_base) for this model."
        )

    need_cred = _model_requires_cred_level(model)
    g_query = query_cache["g"].to(device)
    gw_s_query = query_cache["gw_s"].to(device)
    scores = []
    if dual:
        H_query = query_cache["H_gw"].to(device)
        gw_m_query = query_cache["gw_m"].to(device)

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_coords is not None and np.asarray(candidate_coords).shape[0] != candidate_indices.shape[0]:
        raise ValueError("candidate_coords must have the same first dimension as candidate_indices.")
    if candidate_abs_dt_days is not None and np.asarray(candidate_abs_dt_days).shape[0] != candidate_indices.shape[0]:
        raise ValueError("candidate_abs_dt_days must have the same first dimension as candidate_indices.")
    use_synthetic_coords = candidate_coords is not None
    for start in range(0, len(candidate_indices), 1024):
        chunk_np = candidate_indices[start:start + 1024]
        chunk_idx = torch.from_numpy(chunk_np).long()
        h_chunk = candidate_bank["h_l_cls"].index_select(0, chunk_idx).to(device)
        opt_coords_chunk = _candidate_coords_for_chunk(
            candidate_bank,
            chunk_idx,
            candidate_coords,
            start=start,
            end=start + len(chunk_np),
            device=device,
        )

        batch_size = h_chunk.size(0)
        g_chunk = g_query.unsqueeze(0).expand(batch_size, -1)
        if dual:
            H_chunk = H_query.unsqueeze(0).expand(batch_size, -1, -1)
            gw_s_chunk = gw_s_query.unsqueeze(0).expand(batch_size, -1)
            gw_m_chunk = gw_m_query.unsqueeze(0).expand(batch_size, -1, -1) if need_cred else None
            cred_chunk = _compute_credible_level_single_gw(gw_m_query, opt_coords_chunk) if need_cred else None
        else:
            H_chunk = None
            gw_s_chunk = None
            gw_m_chunk = None
            cred_chunk = None

        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            if candidate_abs_dt_days is None:
                dt_chunk = None
            else:
                dt_chunk = torch.from_numpy(
                    np.asarray(candidate_abs_dt_days[start:start + len(chunk_np)], dtype=np.float32)
                ).to(device=device)
            z_chunk = _candidate_z_l_for_chunk(
                model,
                candidate_bank,
                chunk_idx,
                opt_coords_chunk,
                device,
                use_synthetic_coords=use_synthetic_coords,
            )
            logits = model.fusion_logits(
                g_chunk,
                h_chunk,
                z_l=z_chunk,
                H_gw=H_chunk,
                cred_level=cred_chunk,
                gw_s=gw_s_chunk,
                gw_m=gw_m_chunk,
                opt_coords=opt_coords_chunk,
                dt_days=dt_chunk,
            )
            logit_margin = logits[:, 1] - logits[:, 0]
        scores.append(logit_margin.float().cpu())

    return torch.cat(scores).numpy()


@torch.no_grad()
def _score_candidate_bank_contrastive(
    model,
    query_cache,
    candidate_indices: np.ndarray,
    candidate_bank: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    candidate_coords: Optional[np.ndarray] = None,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
) -> np.ndarray:
    """Score a query GW against an optical candidate bank with contrastive similarity."""
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.float32)

    core_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    g_query = query_cache["g"].to(device).unsqueeze(0)
    with _autocast_context(device, amp_dtype, enabled=amp_enabled):
        feat_g = core_model.project_gw_features(g_query)

    scores = []
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_coords is not None and np.asarray(candidate_coords).shape[0] != candidate_indices.shape[0]:
        raise ValueError("candidate_coords must have the same first dimension as candidate_indices.")
    use_synthetic_coords = candidate_coords is not None
    for start in range(0, len(candidate_indices), 1024):
        chunk_np = candidate_indices[start:start + 1024]
        chunk_idx = torch.from_numpy(chunk_np).long()
        opt_coords_chunk = _candidate_coords_for_chunk(
            candidate_bank,
            chunk_idx,
            candidate_coords,
            start=start,
            end=start + len(chunk_np),
            device=device,
        )
        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            z_chunk = _candidate_z_l_for_chunk(
                model,
                candidate_bank,
                chunk_idx,
                opt_coords_chunk,
                device,
                use_synthetic_coords=use_synthetic_coords,
            )
            feat_o = core_model.project_optical_features(z_chunk)
        sims = torch.matmul(feat_g, feat_o.T).squeeze(0)
        scores.append(sims.float().cpu())

    return torch.cat(scores).numpy()


@torch.no_grad()
def score_all_galleries_multimodal(
    model,
    positive_bank,
    negative_bank,
    galleries,
    unique_gw,
    *,
    test_data_path: str,
    device: torch.device,
    model_args: Dict[str, Any],
    gw_event_time_mjd_table=None,
    candidate_time_delta_mode: str = "native",
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
):
    """Score galleries with one matched KN and tutorial distractors using fusion logits."""
    candidate_time_delta_mode = normalize_candidate_time_delta_mode(
        candidate_time_delta_mode
    )
    model_requires_time_delta = _model_requires_time_delta(model)
    if candidate_time_delta_mode == "zero" and not model_requires_time_delta:
        raise ValueError(
            "candidate_time_delta_mode='zero' requested, but the runtime model "
            "does not consume a time-delta feature."
        )
    print(
        "  Runtime candidate time-delta mode: "
        f"{candidate_time_delta_mode} "
        f"(model_requires_time_delta={model_requires_time_delta})"
    )
    dual = bool(positive_bank.get("dual_fusion", _is_dual_fusion_model(model)))
    print("  Building GW query cache...")
    query_cache = _build_gallery_query_cache(
        model,
        unique_gw,
        test_data_path,
        device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )

    ranks: Dict[Tuple[int, int, int], int] = {}
    observed_dt_count = 0
    observed_nonzero_dt_count = 0
    for key, gallery_spec in tqdm(galleries.items(), desc="  Scoring galleries"):
        _gallery_size, _trial, gw_id = key
        pos_index = int(gallery_spec["positive_index"])
        neg_indices = np.asarray(gallery_spec["negative_indices"], dtype=np.int64)
        neg_candidate_coords = gallery_spec.get("negative_synthetic_coordinates")
        gw_event_time_mjd = _lookup_event_time_mjd(gw_event_time_mjd_table, int(gw_id))
        neg_abs_dt_days = extract_gallery_negative_abs_dt_days(
            gallery_spec,
            gw_event_time_mjd=gw_event_time_mjd,
            n_negative=int(neg_indices.shape[0]),
        )
        # Compute real abs_dt for the positive candidate
        if "first_detection_mjd" in positive_bank and gw_event_time_mjd is not None and np.isfinite(gw_event_time_mjd):
            pos_first_det = float(positive_bank["first_detection_mjd"][pos_index].item())
            pos_dt = abs(pos_first_det - gw_event_time_mjd)
        else:
            pos_dt = 0.0
        pos_dt_values = apply_candidate_time_delta_mode(
            np.asarray([pos_dt], dtype=np.float32),
            mode=candidate_time_delta_mode,
            n_candidates=1,
        )
        neg_dt_values = apply_candidate_time_delta_mode(
            neg_abs_dt_days,
            mode=candidate_time_delta_mode,
            n_candidates=int(neg_indices.shape[0]),
        )
        for label, values in (("positive", pos_dt_values), ("negative", neg_dt_values)):
            if values is None:
                continue
            values_np = np.asarray(values)
            observed_dt_count += int(values_np.size)
            nonzero_count = int(np.count_nonzero(values_np))
            observed_nonzero_dt_count += nonzero_count
            if candidate_time_delta_mode == "zero" and nonzero_count:
                raise AssertionError(
                    f"Runtime {label} dt audit failed: found {nonzero_count} non-zero values."
                )
        pos_scores = _score_candidate_bank_with_logits(
            model,
            query_cache[int(gw_id)],
            np.asarray([pos_index], dtype=np.int64),
            positive_bank,
            device,
            dual,
            candidate_abs_dt_days=pos_dt_values,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        neg_scores = _score_candidate_bank_with_logits(
            model,
            query_cache[int(gw_id)],
            neg_indices,
            negative_bank,
            device,
            dual,
            candidate_coords=neg_candidate_coords,
            candidate_abs_dt_days=neg_dt_values,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        scores = np.concatenate([pos_scores, neg_scores], axis=0)
        ranked = np.argsort(-scores)
        ranks[key] = int(np.where(ranked == 0)[0][0])

    print(
        "  Runtime candidate time-delta audit: "
        f"count={observed_dt_count}, nonzero={observed_nonzero_dt_count}"
    )
    return ranks


@torch.no_grad()
def score_all_galleries_contrastive(
    model,
    positive_bank,
    negative_bank,
    galleries,
    unique_gw,
    *,
    test_data_path: str,
    device: torch.device,
    model_args: Dict[str, Any],
    gw_event_time_mjd_table=None,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
):
    """Score galleries with one matched KN and tutorial distractors using contrastive similarity."""
    print("  Building GW query cache...")
    query_cache = _build_gallery_query_cache(
        model,
        unique_gw,
        test_data_path,
        device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )

    ranks: Dict[Tuple[int, int, int], int] = {}
    for key, gallery_spec in tqdm(galleries.items(), desc="  Scoring galleries (contrastive)"):
        _gallery_size, _trial, gw_id = key
        pos_index = int(gallery_spec["positive_index"])
        neg_indices = np.asarray(gallery_spec["negative_indices"], dtype=np.int64)
        neg_candidate_coords = gallery_spec.get("negative_synthetic_coordinates")
        pos_sims = _score_candidate_bank_contrastive(
            model,
            query_cache[int(gw_id)],
            np.asarray([pos_index], dtype=np.int64),
            positive_bank,
            device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        neg_sims = _score_candidate_bank_contrastive(
            model,
            query_cache[int(gw_id)],
            neg_indices,
            negative_bank,
            device,
            candidate_coords=neg_candidate_coords,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        sims = np.concatenate([pos_sims, neg_sims], axis=0)
        ranked = np.argsort(-sims)
        ranks[key] = int(np.where(ranked == 0)[0][0])

    return ranks


@torch.no_grad()
def score_all_galleries_optical(
    opt_model,
    positive_raw_opt,
    negative_raw_opt,
    galleries,
    *,
    device: torch.device,
    n_ref: int,
    ref_start: float,
    ref_end: float,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
):
    """Score galleries with one matched KN and tutorial distractors using optical-only logits."""
    ranks: Dict[Tuple[int, int, int], int] = {}
    for key, gallery_spec in tqdm(galleries.items(), desc="  Scoring galleries"):
        pos_index = int(gallery_spec["positive_index"])
        neg_indices = np.asarray(gallery_spec["negative_indices"], dtype=np.int64)
        pos_probs = score_gallery_optical_only(
            opt_model,
            np.asarray([pos_index], dtype=np.int64),
            positive_raw_opt["opt_t_raw"],
            positive_raw_opt["opt_v_raw"],
            positive_raw_opt["opt_mask_raw"],
            positive_raw_opt["opt_err_raw"],
            device,
            n_ref=n_ref,
            ref_start=ref_start,
            ref_end=ref_end,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        neg_probs = score_gallery_optical_only(
            opt_model,
            neg_indices,
            negative_raw_opt["times"],
            negative_raw_opt["values"],
            negative_raw_opt["masks"],
            negative_raw_opt["errors"],
            device,
            n_ref=n_ref,
            ref_start=ref_start,
            ref_end=ref_end,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        probs = np.concatenate([pos_probs, neg_probs], axis=0)
        ranked = np.argsort(-probs)
        ranks[key] = int(np.where(ranked == 0)[0][0])
    return ranks



def _read_h5_attrs(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with h5py.File(path, "r") as f:
        out: Dict[str, Any] = {}
        for key in f.attrs.keys():
            value = f.attrs[key]
            if isinstance(value, np.ndarray):
                out[str(key)] = value.tolist()
            elif isinstance(value, np.generic):
                out[str(key)] = value.item()
            elif isinstance(value, bytes):
                out[str(key)] = value.decode("utf-8", errors="replace")
            else:
                out[str(key)] = value
    return out


def load_fink_rf_bundle(checkpoint_path: str) -> Dict[str, Any]:
    if load_fink_rf_artifact is None or fink_extract_feature_matrix_from_batch is None:
        raise ImportError(
            "Fink RF support requires the standalone package at "
            f"{FINK_RF_SRC_DIR}. Run from the bgao_kn workspace or install fink-rf-reproduction."
        )
    artifact = load_fink_rf_artifact(checkpoint_path)
    missing = [key for key in ("pipeline", "basis", "config") if key not in artifact]
    if missing:
        raise KeyError(f"Fink RF artifact is missing required key(s): {missing}")
    print(f"  Loaded Fink RF artifact from {checkpoint_path}")
    return artifact


def _bank_tensor_to_numpy(bank: Mapping[str, Any], key_candidates: Sequence[str], indices: np.ndarray) -> np.ndarray:
    for key in key_candidates:
        if key in bank:
            value = bank[key]
            idx = np.asarray(indices, dtype=np.int64)
            if isinstance(value, torch.Tensor):
                idx_tensor = torch.from_numpy(idx).long()
                return value.index_select(0, idx_tensor).detach().cpu().numpy()
            return np.asarray(value)[idx]
    raise KeyError(f"Candidate bank is missing all of these fields: {list(key_candidates)}")


def _fink_rf_batch_from_bank(bank: Mapping[str, Any], indices: np.ndarray) -> Dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    return {
        "times": _bank_tensor_to_numpy(bank, ("opt_t_raw", "times"), idx).astype(np.float32, copy=False),
        "values": _bank_tensor_to_numpy(bank, ("opt_v_raw", "values"), idx).astype(np.float32, copy=False),
        "masks": _bank_tensor_to_numpy(bank, ("opt_mask_raw", "masks"), idx).astype(np.float32, copy=False),
        "errors": _bank_tensor_to_numpy(bank, ("opt_err_raw", "errors"), idx).astype(np.float32, copy=False),
    }


def score_fink_rf_candidate_bank(
    artifact: Mapping[str, Any],
    candidate_bank: Mapping[str, Any],
    candidate_indices: np.ndarray,
    attrs: Mapping[str, Any],
    *,
    chunk_size: Optional[int] = None,
) -> Tuple[Dict[int, float], Dict[int, bool], Dict[str, Any]]:
    indices = np.unique(np.asarray(candidate_indices, dtype=np.int64).reshape(-1))
    if indices.size == 0:
        return {}, {}, {"n_rows": 0, "n_valid": 0, "valid_fraction": 0.0}
    cfg = dict(artifact.get("config", {}))
    basis = artifact["basis"]
    pipeline = artifact["pipeline"]
    bs = int(chunk_size or cfg.get("batch_size", 2048))
    fit_window = tuple(cfg.get("fit_window_days", (-50.0, 50.0)))
    min_band_points = int(cfg.get("min_band_points", 2))
    min_fmax_psfflux = None if cfg.get("min_fmax_psfflux") is None else float(cfg.get("min_fmax_psfflux"))
    fmax_threshold_source = str(cfg.get("fmax_threshold_source", "lsst_m5"))
    fmax_threshold_scale = float(cfg.get("fmax_threshold_scale", 1.0))
    default_time_scale = float(cfg.get("default_time_scale_divisor_days", 100.0))
    min_valid_bands = int(cfg.get("min_valid_bands", 1))
    coefficient_bound = float(cfg.get("coefficient_bound", 2.0))

    score_map: Dict[int, float] = {}
    valid_map: Dict[int, bool] = {}
    n_valid = 0
    band_counts: Dict[str, int] = {str(band): 0 for band in getattr(basis, "bands", [])}
    for start in tqdm(range(0, indices.size, bs), desc="  Scoring Fink RF candidates", leave=False):
        chunk = indices[start:start + bs]
        batch = _fink_rf_batch_from_bank(candidate_bank, chunk)
        X, valid, counts = fink_extract_feature_matrix_from_batch(
            batch,
            attrs,
            basis=basis,
            fit_window_days=fit_window,
            min_band_points=min_band_points,
            min_fmax_psfflux=min_fmax_psfflux,
            fmax_threshold_source=fmax_threshold_source,
            fmax_threshold_scale=fmax_threshold_scale,
            default_time_scale_divisor_days=default_time_scale,
            min_valid_bands=min_valid_bands,
            coefficient_bound=coefficient_bound,
        )
        probs = pipeline.predict_proba(X)[:, 1].astype(np.float64, copy=False)
        probs = np.where(valid, probs, -np.inf)
        for row_idx, prob, ok in zip(chunk.tolist(), probs.tolist(), valid.tolist()):
            score_map[int(row_idx)] = float(prob)
            valid_map[int(row_idx)] = bool(ok)
        n_valid += int(np.asarray(valid, dtype=bool).sum())
        for band, count in counts["band_valid_counts"].items():
            band_counts[str(band)] = int(band_counts.get(str(band), 0) + int(count))
    diagnostics = {
        "n_rows": int(indices.size),
        "n_valid": int(n_valid),
        "valid_fraction": float(n_valid / max(1, int(indices.size))),
        "band_valid_counts": band_counts,
        "band_valid_fraction": {band: float(count / max(1, int(indices.size))) for band, count in band_counts.items()},
    }
    return score_map, valid_map, diagnostics


def _collect_gallery_negative_indices(galleries: Mapping[Tuple[int, int, int], Mapping[str, Any]]) -> np.ndarray:
    parts = []
    for spec in galleries.values():
        neg = np.asarray(spec.get("negative_indices", []), dtype=np.int64).reshape(-1)
        if neg.size:
            parts.append(neg)
    if not parts:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(parts, axis=0).astype(np.int64, copy=False))


def score_all_galleries_fink_rf(
    artifact: Mapping[str, Any],
    positive_bank: Mapping[str, Any],
    negative_bank: Mapping[str, Any],
    galleries: Mapping[Tuple[int, int, int], Mapping[str, Any]],
    *,
    positive_attrs: Mapping[str, Any],
    negative_attrs: Mapping[str, Any],
    chunk_size: Optional[int] = None,
) -> Tuple[Dict[Tuple[int, int, int], Dict[str, Any]], Dict[str, Any]]:
    positive_indices = np.unique(
        np.asarray([int(spec["positive_index"]) for spec in galleries.values()], dtype=np.int64)
    )
    negative_indices = _collect_gallery_negative_indices(galleries)
    pos_scores, pos_valid, pos_diag = score_fink_rf_candidate_bank(
        artifact,
        positive_bank,
        positive_indices,
        positive_attrs,
        chunk_size=chunk_size,
    )
    neg_scores, neg_valid, neg_diag = score_fink_rf_candidate_bank(
        artifact,
        negative_bank,
        negative_indices,
        negative_attrs,
        chunk_size=chunk_size,
    )

    outcomes: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    max_gallery_size = max(
        (int(spec.get("actual_gallery_size", 1)) for spec in galleries.values()),
        default=1,
    )
    harmonic_numbers = np.concatenate(
        [np.asarray([0.0]), np.cumsum(1.0 / np.arange(1, max_gallery_size + 1, dtype=np.float64))]
    )
    positive_invalid = 0
    negative_invalid = 0
    tie_block_sizes: List[int] = []
    n_tied_galleries = 0
    for key, gallery_spec in tqdm(galleries.items(), desc="  Ranking Fink RF galleries"):
        pos_index = int(gallery_spec["positive_index"])
        neg_indices = np.asarray(gallery_spec.get("negative_indices", []), dtype=np.int64).reshape(-1)
        pos_score = float(pos_scores.get(pos_index, -np.inf))
        pos_is_valid = bool(pos_valid.get(pos_index, False))
        if not pos_is_valid:
            positive_invalid += 1
        neg_values = []
        for neg_index in neg_indices.tolist():
            neg_values.append(float(neg_scores.get(int(neg_index), -np.inf)))
            if not bool(neg_valid.get(int(neg_index), False)):
                negative_invalid += 1
        neg_array = np.asarray(neg_values, dtype=np.float64)
        if pos_is_valid:
            n_strictly_better = int(np.sum(neg_array > pos_score))
            n_tied_negatives = int(np.sum(neg_array == pos_score))
            tie_block_size = n_tied_negatives + 1
            tie_block_sizes.append(tie_block_size)
            if n_tied_negatives > 0:
                n_tied_galleries += 1
        else:
            n_strictly_better = int(neg_indices.size)
            n_tied_negatives = 0
            tie_block_size = 1

        metric_contributions = uniform_random_tie_metric_contributions(
            n_strictly_better,
            n_tied_negatives,
            harmonic_numbers=harmonic_numbers,
        )
        outcomes[key] = {
            "rank": float(n_strictly_better + 0.5 * n_tied_negatives),
            "metric_contributions": metric_contributions,
            "n_strictly_better": n_strictly_better,
            "n_tied_negatives": n_tied_negatives,
            "tie_block_size": tie_block_size,
            "tie_policy": "uniform_random_expected",
            "positive_valid": pos_is_valid,
            "requested_gallery_size": int(gallery_spec.get("requested_gallery_size", key[0])),
            "actual_gallery_size": int(gallery_spec.get("actual_gallery_size", neg_indices.size + 1)),
            "coverage_met": bool(gallery_spec.get("coverage_met", True)),
            "is_undersized": bool(gallery_spec.get("is_undersized", False)),
        }

    n_valid_galleries = len(tie_block_sizes)
    diagnostics = {
        "positive": pos_diag,
        "negative": neg_diag,
        "positive_invalid_gallery_count": int(positive_invalid),
        "negative_invalid_candidate_count": int(negative_invalid),
        "tie_policy": "uniform_random_expected",
        "tie_score_equality": "exact_float64",
        "n_galleries": int(len(galleries)),
        "n_valid_positive_galleries": int(n_valid_galleries),
        "n_tied_valid_galleries": int(n_tied_galleries),
        "tied_valid_gallery_fraction": float(n_tied_galleries / max(1, n_valid_galleries)),
        "tie_block_size_mean": float(np.mean(tie_block_sizes)) if tie_block_sizes else 0.0,
        "tie_block_size_median": float(np.median(tie_block_sizes)) if tie_block_sizes else 0.0,
        "tie_block_size_max": int(max(tie_block_sizes)) if tie_block_sizes else 0,
    }
    return outcomes, diagnostics


def aggregate_metrics(ranks, gallery_sizes, n_trials, unique_gw, *, gw_source_map=None):
    """Convert per-gallery ranks to Recall@K and MRR metrics."""
    metrics: Dict[str, float] = {}
    by_source: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for gallery_size in gallery_sizes:
        recalls = {1: [], 5: [], 10: []}
        mrrs: List[float] = []
        for trial in range(int(n_trials)):
            for gw_id in unique_gw:
                key = (int(gallery_size), int(trial), int(gw_id))
                if key not in ranks:
                    continue
                rank = float(ranks[key])
                for k in recalls:
                    recalls[k].append(1.0 if rank < k else 0.0)
                mrrs.append(1.0 / float(rank + 1))

                if gw_source_map:
                    source_label = gw_source_map.get(int(gw_id), "unknown")
                    if source_label not in by_source:
                        by_source[source_label] = {}
                    if int(gallery_size) not in by_source[source_label]:
                        by_source[source_label][int(gallery_size)] = {
                            "recalls": {1: [], 5: [], 10: []},
                            "mrrs": [],
                        }
                    source_bucket = by_source[source_label][int(gallery_size)]
                    for k in source_bucket["recalls"]:
                        source_bucket["recalls"][k].append(1.0 if rank < k else 0.0)
                    source_bucket["mrrs"].append(1.0 / float(rank + 1))

        for k, values in recalls.items():
            metrics[f"gallery_{gallery_size}_recall_at_{k}"] = float(np.mean(values)) if values else 0.0
        metrics[f"gallery_{gallery_size}_mrr"] = float(np.mean(mrrs)) if mrrs else 0.0

    source_metrics: Dict[str, Dict[str, float]] = {}
    for source_label, size_acc in by_source.items():
        src_out: Dict[str, float] = {}
        for gallery_size, acc in size_acc.items():
            for k, values in acc["recalls"].items():
                src_out[f"gallery_{gallery_size}_recall_at_{k}"] = float(np.mean(values)) if values else 0.0
            src_out[f"gallery_{gallery_size}_mrr"] = float(np.mean(acc["mrrs"])) if acc["mrrs"] else 0.0
        source_metrics[source_label] = src_out

    return metrics, source_metrics


def print_unified_table(all_results, gallery_sizes, model_names):
    """Print comparison table matching the paper format."""
    col_w = 7
    name_w = max(max(len(name) for name in model_names), len("Method")) + 2
    group_w = col_w * len(TABLE_METRIC_LABELS) + (len(TABLE_METRIC_LABELS) - 1)

    header_1 = f"{'Method':<{name_w}}"
    header_2 = f"{'':<{name_w}}"
    for gallery_size in gallery_sizes:
        header_1 += f" | {'gallery=' + str(gallery_size):^{group_w}}"
        for metric_label in TABLE_METRIC_LABELS:
            header_2 += f" | {metric_label:>{col_w - 2}}"

    sep = "-" * len(header_2)
    print(f"\n{sep}")
    print(header_1)
    print(header_2)
    print(sep)

    for name in model_names:
        row_metrics = all_results.get(name, {})
        row = f"{name:<{name_w}}"
        for gallery_size in gallery_sizes:
            for metric_key in TABLE_METRIC_KEYS:
                full_key = f"gallery_{gallery_size}_{metric_key}"
                value = row_metrics.get(full_key)
                if value is None:
                    row += f" | {'--':>{col_w - 2}}"
                else:
                    row += f" | {value:>{col_w - 2}.4f}"
        print(row)

    print(sep)


def print_latex_table(all_results, gallery_sizes, model_names):
    """Print LaTeX-formatted table for the paper."""
    n_metrics = len(TABLE_METRIC_LABELS)
    print("\n% --- LaTeX table ---")
    print("\\begin{table}[htbp]")
    print("\\centering")
    col_spec = "l" + "|".join(["" + "c" * n_metrics for _ in gallery_sizes])
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    header_parts = ["\\multirow{2}{*}{Method}"]
    for gallery_size in gallery_sizes:
        header_parts.append(f"\\multicolumn{{{n_metrics}}}{{c}}{{gallery={gallery_size}}}")
    print(" & ".join(header_parts) + " \\")

    sub_parts = [""]
    for _gallery_size in gallery_sizes:
        sub_parts.extend(TABLE_METRIC_LABELS)
    print(" & ".join(sub_parts) + " \\")
    print("\\midrule")

    best_vals: Dict[str, float] = {}
    for gallery_size in gallery_sizes:
        for metric_key in TABLE_METRIC_KEYS:
            full_key = f"gallery_{gallery_size}_{metric_key}"
            values = [all_results[name].get(full_key, -1.0) for name in model_names if all_results.get(name)]
            if values:
                best_vals[full_key] = max(values)

    for name in model_names:
        row_metrics = all_results.get(name, {})
        parts = [name.replace("_", "\\_")]
        for gallery_size in gallery_sizes:
            for metric_key in TABLE_METRIC_KEYS:
                full_key = f"gallery_{gallery_size}_{metric_key}"
                if not row_metrics:
                    parts.append("--")
                    continue
                value = row_metrics.get(full_key)
                if value is None:
                    parts.append("--")
                    continue
                cell = f"{value:.4f}"
                if abs(value - best_vals.get(full_key, -1.0)) < 1e-6:
                    cell = f"\\textbf{{{cell}}}"
                parts.append(cell)
        print(" & ".join(parts) + " \\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Retrieval comparison across model variants.}")
    print("\\label{tab:retrieval_comparison}")
    print("\\end{table}")
    print("% --- end LaTeX table ---")


def build_table_rows(all_results, gallery_sizes, model_names):
    rows = []
    for name in model_names:
        row = {"method": name}
        row_metrics = all_results.get(name, {})
        for gallery_size in gallery_sizes:
            gallery_block = {}
            for metric_label, metric_key in zip(TABLE_METRIC_LABELS, TABLE_METRIC_KEYS):
                gallery_block[metric_label] = row_metrics.get(f"gallery_{gallery_size}_{metric_key}")
            row[f"gallery_{gallery_size}"] = gallery_block
        rows.append(row)
    return rows


def build_gw_source_map(gw_source_types, gw_ids):
    if gw_source_types is None:
        return {}
    if isinstance(gw_ids, torch.Tensor):
        unique_gw = sorted(torch.unique(gw_ids.detach().cpu().to(torch.long)).tolist())
    else:
        unique_gw = sorted({int(gw_id) for gw_id in gw_ids})
    return {int(gw_id): _lookup_source_type(int(gw_id), gw_source_types) for gw_id in unique_gw}




def build_simple_inbatch_negative_indices(gw_indices: torch.Tensor) -> torch.Tensor:
    """Deterministic roll-and-skip negative pairing for ablation comparison."""
    gw_indices = gw_indices.to(torch.long)
    batch_size = int(gw_indices.numel())
    device = gw_indices.device
    base = torch.arange(batch_size, device=device, dtype=torch.long)
    if batch_size <= 1:
        return base

    result = torch.roll(base, shifts=1, dims=0)
    unresolved = gw_indices[result] == gw_indices
    shift = 2
    while unresolved.any() and shift < batch_size:
        candidate = torch.roll(base, shifts=shift, dims=0)
        valid = gw_indices[candidate] != gw_indices
        update_mask = unresolved & valid
        result = torch.where(update_mask, candidate, result)
        unresolved = unresolved & (~valid)
        shift += 1

    if unresolved.any():
        unresolved_rows = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
        for row in unresolved_rows.tolist():
            candidates = torch.nonzero(gw_indices != gw_indices[row], as_tuple=False).squeeze(-1)
            if candidates.numel() > 0:
                result[row] = candidates[0]
            else:
                result[row] = row
    return result


def sample_simple_inbatch_hard_negatives_with_time(
    sim_g2o,
    gw_indices,
    batch_event_time_mjd,
    window_days,
    min_candidates,
    semi_hard,
    semi_hard_margin,
    fallback_mode,
    candidate_time_mjd=None,
    active_rows=None,
):
    del sim_g2o, batch_event_time_mjd, window_days, min_candidates, semi_hard, semi_hard_margin, fallback_mode, candidate_time_mjd
    full_idx = build_simple_inbatch_negative_indices(gw_indices)
    if active_rows is not None:
        active_rows = active_rows.to(device=gw_indices.device, dtype=torch.long)
        return full_idx[active_rows], [0], 0
    return full_idx, [0], 0


@contextmanager
def temporary_simple_hard_negative_sampling(model):
    from scripts.train import train as train_module

    patch_targets = [model]
    core_model = getattr(model, "_orig_mod", None)
    if core_model is not None:
        patch_targets.append(core_model)

    missing_time_sampler = object()
    original_time_sampler = getattr(
        train_module, "sample_inbatch_hard_negatives_with_time", missing_time_sampler
    )
    original_methods = []

    def _simple_sampler(sim_g2o, gw_indices=None, margin=0.2):
        del sim_g2o, margin
        if gw_indices is None:
            raise ValueError("simple hard-negative sampler requires gw_indices")
        return build_simple_inbatch_negative_indices(gw_indices)

    train_module.sample_inbatch_hard_negatives_with_time = sample_simple_inbatch_hard_negatives_with_time
    for target in patch_targets:
        original_methods.append((target, getattr(target, "sample_semi_hard_negatives", None), getattr(target, "sample_hard_negatives", None)))
        if hasattr(target, "sample_semi_hard_negatives"):
            target.sample_semi_hard_negatives = _simple_sampler
        if hasattr(target, "sample_hard_negatives"):
            target.sample_hard_negatives = _simple_sampler
    try:
        yield
    finally:
        current_time_sampler = getattr(
            train_module, "sample_inbatch_hard_negatives_with_time", None
        )
        if original_time_sampler is missing_time_sampler:
            if current_time_sampler is sample_simple_inbatch_hard_negatives_with_time:
                delattr(train_module, "sample_inbatch_hard_negatives_with_time")
        else:
            train_module.sample_inbatch_hard_negatives_with_time = original_time_sampler
        for target, original_semi, original_hard in original_methods:
            if original_semi is not None:
                target.sample_semi_hard_negatives = original_semi
            if original_hard is not None:
                target.sample_hard_negatives = original_hard


def evaluate_multimodal_classification(
    model,
    runtime_model_args: Dict[str, Any],
    saved_args: Dict[str, Any],
    *,
    seed: int,
    test_data_path: str,
    neg_data_path: Optional[str],
    neg_group: str,
    batch_size: int,
    test_steps: Optional[int],
    num_workers: int,
    n_neg_samples: int,
    comparison_window: Tuple[float, float],
    nonkn_cls_base_field: str,
    report_dt_bins: bool,
    dt_bin_edges: Optional[List[float]],
    report_dt_macro: bool,
    neg_optical_data,
    neg_gw_indices,
    gw_source_types,
    gw_event_time_mjd_table,
    shared_cfg: Dict[str, Any],
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    device: torch.device,
    return_logits: bool = False,
):
    """Run triplet classification for a multimodal model."""
    hardneg_windows_days = parse_day_windows(str(saved_args.get("hardneg_time_window_days", "30,60,120")))
    hardneg_min_candidates = int(saved_args.get("hardneg_min_candidates", 4))
    hardneg_semi_hard = _as_bool(saved_args.get("semi_hard", True), default=True)
    hardneg_semi_hard_margin = float(saved_args.get("semi_hard_margin", 0.2))
    hardneg_fallback_mode = str(saved_args.get("hardneg_fallback_mode", "inbatch_semihard"))
    hard_negative_strategy = str(shared_cfg.get("hard_negative_strategy", "simple")).strip().lower()

    try_num_workers = int(num_workers)
    while True:
        try:
            _seed_all(seed)
            loader = _build_test_loader(
                test_data_path,
                neg_data_path,
                neg_group,
                comparison_window[0],
                comparison_window[1],
                batch_size=batch_size,
                test_steps=test_steps,
                num_workers=try_num_workers,
                target_samples=n_neg_samples,
                nonkn_cls_base_field=nonkn_cls_base_field,
                return_zero_time_mjd=True,
            )
            if hard_negative_strategy == "simple":
                with temporary_simple_hard_negative_sampling(model):
                    triplet_logits = extract_triplet_logits(
                        model,
                        loader,
                        device,
                        runtime_model_args,
                        neg_optical_data,
                        neg_gw_indices,
                        gw_source_types=gw_source_types,
                        shuffle_gw=False,
                        shuffle_seed=int(seed),
                        amp_dtype=amp_dtype,
                        amp_enabled=amp_enabled,
                        gw_event_time_mjd_table=gw_event_time_mjd_table,
                        hardneg_windows_days=hardneg_windows_days,
                        hardneg_min_candidates=hardneg_min_candidates,
                        hardneg_semi_hard=hardneg_semi_hard,
                        hardneg_semi_hard_margin=hardneg_semi_hard_margin,
                        hardneg_fallback_mode=hardneg_fallback_mode,
                    )
            else:
                triplet_logits = extract_triplet_logits(
                    model,
                    loader,
                    device,
                    runtime_model_args,
                    neg_optical_data,
                    neg_gw_indices,
                    gw_source_types=gw_source_types,
                    shuffle_gw=False,
                    shuffle_seed=int(seed),
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    hardneg_windows_days=hardneg_windows_days,
                    hardneg_min_candidates=hardneg_min_candidates,
                    hardneg_semi_hard=hardneg_semi_hard,
                    hardneg_semi_hard_margin=hardneg_semi_hard_margin,
                    hardneg_fallback_mode=hardneg_fallback_mode,
                )
            cls_metrics = evaluate_classification_triplet(
                triplet_logits,
                report_dt_bins=bool(report_dt_bins),
                dt_bin_edges=dt_bin_edges,
                report_dt_macro=bool(report_dt_macro),
            )
            cls_by_source = evaluate_classification_triplet_by_source(triplet_logits)
            if return_logits:
                return cls_metrics, cls_by_source, triplet_logits
            return cls_metrics, cls_by_source
        except PermissionError:
            if try_num_workers <= 0:
                raise
            print("  WARNING: DataLoader multiprocessing failed. Retrying triplet classification with num_workers=0.")
            try_num_workers = 0


def normalize_shared_config(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    cfg_dir = cfg_path.parent
    # Resolve comparison window: opt_ref_start/end > comparison_window_start/end > default
    opt_ref_start = cfg.get("opt_ref_start")
    opt_ref_end = cfg.get("opt_ref_end")
    if opt_ref_start is not None and opt_ref_end is not None:
        comparison_window = (float(opt_ref_start), float(opt_ref_end))
        comparison_window_source = "opt_ref"
    elif cfg.get("comparison_window_start") is not None and cfg.get("comparison_window_end") is not None:
        comparison_window = (
            float(cfg["comparison_window_start"]),
            float(cfg["comparison_window_end"]),
        )
        comparison_window_source = "comparison_window_start_end"
    else:
        comparison_window = DEFAULT_COMPARISON_WINDOW
        comparison_window_source = "default"
    requested_neg_data_path = _resolve_path(cfg_dir, cfg.get("neg_data_path"))
    requested_neg_group = str(cfg.get("neg_group", DEFAULT_TUTORIAL_NEG_GROUP))
    tutorial_neg_data_path = _resolve_path(cfg_dir, cfg.get("tutorial_neg_data_path")) or DEFAULT_TUTORIAL_NEG_DATA_PATH
    tutorial_neg_group = str(cfg.get("tutorial_neg_group", DEFAULT_TUTORIAL_NEG_GROUP))
    force_tutorial_distractors = _as_bool(cfg.get("force_tutorial_distractors", True), default=True)
    positive_selection = str(cfg.get("positive_selection", "random")).strip().lower()
    if positive_selection not in {"random", "batch"}:
        raise ValueError("positive_selection must be one of {'random', 'batch'}.")
    # --- Redshift analysis ---
    redshift_analysis_enable = _as_bool(cfg.get("redshift_analysis_enable", False), default=False)
    redshift_catalogs = _normalize_redshift_catalog_paths(
        cfg.get("redshift_catalogs"), cfg_dir
    )
    redshift_bin_edges_raw = cfg.get("redshift_bin_edges")
    redshift_bin_labels_raw = cfg.get("redshift_bin_labels")
    redshift_bin_edges = None
    redshift_bin_labels = None
    if redshift_analysis_enable:
        redshift_bin_edges, redshift_bin_labels = _normalize_redshift_bin_config(
            redshift_bin_edges_raw, redshift_bin_labels_raw
        )

    normalized = {
        "input_config": str(cfg_path),
        "test_data_path": _resolve_path(cfg_dir, cfg.get("test_data_path")),
        "neg_data_path": tutorial_neg_data_path if force_tutorial_distractors else requested_neg_data_path,
        "neg_group": tutorial_neg_group if force_tutorial_distractors else requested_neg_group,
        "requested_neg_data_path": requested_neg_data_path,
        "requested_neg_group": requested_neg_group,
        "tutorial_neg_data_path": tutorial_neg_data_path,
        "tutorial_neg_group": tutorial_neg_group,
        "force_tutorial_distractors": bool(force_tutorial_distractors),
        "output_dir": _resolve_path(cfg_dir, cfg.get("output_dir")) or str((MODEL_DIR / "eval_results" / "ablation_comparison").resolve()),
        "device": str(cfg.get("device", "cuda")),
        "seed": int(cfg.get("seed", 42)),
        "batch_size": int(cfg.get("batch_size", 512)),
        "test_steps": None if cfg.get("test_steps") is None else int(cfg.get("test_steps")),
        "num_workers": int(cfg.get("num_workers", 2)),
        "gallery_sizes": _parse_gallery_sizes(cfg.get("gallery_sizes", "10,100,500,1000,2000,5000")),
        "gallery_trials": int(cfg.get("gallery_trials", 1)),
        "gallery_candidate_mode": str(cfg.get("gallery_candidate_mode", "time_sky_hard")).strip().lower(),
        "gallery_candidate_time_window_days": float(cfg.get("gallery_candidate_time_window_days", 30.0)),
        "gallery_candidate_credible_level_max": float(cfg.get("gallery_candidate_credible_level_max", 0.9)),
        "gallery_include_undersized": _as_bool(cfg.get("gallery_include_undersized", True), default=True),
        "positive_selection": positive_selection,
        "n_neg_samples": _parse_n_neg_samples(cfg.get("n_neg_samples", -1), default=-1),
        "negative_sample_strategy": str(cfg.get("negative_sample_strategy", "block_random")),
        "negative_sample_block_rows": (
            None
            if cfg.get("negative_sample_block_rows") in (None, "", "null")
            else int(cfg.get("negative_sample_block_rows"))
        ),
        "negative_sample_shuffle": _as_bool(cfg.get("negative_sample_shuffle", True), default=True),
        "max_gw_events": int(cfg.get("max_gw_events", 0) or 0),
        "amp_dtype": str(cfg.get("amp_dtype", "auto")),
        "no_latex": _as_bool(cfg.get("no_latex", False)),
        "compute_classification_metrics": _as_bool(cfg.get("compute_classification_metrics", True), default=True),
        "save_outcomes": _as_bool(cfg.get("save_outcomes", False)),
        "strict_output_safety": _as_bool(cfg.get("strict_output_safety", False)),
        "resume": _as_bool(cfg.get("resume", False)),
        "experiment_id": str(cfg.get("experiment_id", "")).strip(),
        "candidate_time_delta_mode": normalize_candidate_time_delta_mode(
            cfg.get("candidate_time_delta_mode", "native")
        ),
        "baseline_results_path": _resolve_path(cfg_dir, cfg.get("baseline_results_path")),
        "comparison_window": [float(comparison_window[0]), float(comparison_window[1])],
        "comparison_window_start": float(comparison_window[0]),
        "comparison_window_end": float(comparison_window[1]),
        "comparison_window_source": comparison_window_source,
        "nonkn_cls_base_field": str(cfg.get("nonkn_cls_base_field", "zero_time_mjd_cls_base")),
        "report_dt_bins": _as_bool(cfg.get("report_dt_bins", False)),
        "dt_bin_edges": str(cfg.get("dt_bin_edges", DEFAULT_DT_BIN_EDGES)),
        "report_dt_macro": _as_bool(cfg.get("report_dt_macro", False)),
        "hard_negative_strategy": str(cfg.get("hard_negative_strategy", "simple")),
        # Redshift analysis
        "redshift_analysis_enable": bool(redshift_analysis_enable),
        "redshift_catalogs": dict(redshift_catalogs),
        "redshift_bin_edges": redshift_bin_edges,
        "redshift_bin_labels": redshift_bin_labels,
    }
    if normalized["test_data_path"] is None:
        raise ValueError("Comparison config is missing test_data_path")
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Unified retrieval and triplet ablation comparison.")
    parser.add_argument("--config", type=str, required=True, help="Path to comparison JSON config")
    parser.add_argument("--refresh-model-type", help="Evaluate only this model type and merge it into an existing artifact.")
    parser.add_argument("--merge-existing", help="Existing ablation_comparison.json used as the partial-refresh base.")
    args = parser.parse_args()
    if bool(args.refresh_model_type) != bool(args.merge_existing):
        parser.error("--refresh-model-type and --merge-existing must be provided together.")


    config_path = Path(args.config).expanduser().resolve()
    raw_cfg = _load_json(config_path)
    cfg = normalize_shared_config(raw_cfg, config_path)
    if cfg["save_outcomes"] and not cfg["strict_output_safety"]:
        raise ValueError("save_outcomes=true requires strict_output_safety=true.")
    if cfg["save_outcomes"] and args.refresh_model_type:
        raise ValueError("Outcome-saving runs do not support partial refresh/merge.")
    output_dir = Path(cfg["output_dir"])
    run_manifest = None
    retrieval_writer = None
    classification_writer = None
    if cfg["strict_output_safety"]:
        experiment_config = dict(raw_cfg)
        for volatile_key in ("seed", "output_dir", "resume"):
            experiment_config.pop(volatile_key, None)
        run_manifest = prepare_output_directory(
            output_dir,
            manifest={
                "experiment_id": cfg["experiment_id"],
                "experiment_digest": stable_digest(experiment_config),
                "code_digest": source_tree_digest(MODEL_DIR),
                "seed": int(cfg["seed"]),
                "input_config": str(config_path),
                "test_data_path": cfg["test_data_path"],
                "neg_data_path": cfg["neg_data_path"],
            },
            resume=cfg["resume"],
        )
    if cfg["save_outcomes"]:
        retrieval_writer = AtomicGzipCsvWriter(output_dir / "retrieval_outcomes.csv.gz", RETRIEVAL_FIELDS)
        classification_writer = AtomicGzipCsvWriter(
            output_dir / "classification_predictions.csv.gz", CLASSIFICATION_FIELDS
        )
    configured_model_specs = _build_model_specs(raw_cfg, config_path.parent)
    base_output = None
    refresh_model_type = str(args.refresh_model_type or "").strip()
    if refresh_model_type:
        merge_path = Path(args.merge_existing).expanduser().resolve()
        base_output = _load_json(merge_path)
        validate_partial_refresh_compatibility(
            base_output=base_output,
            cfg=cfg,
            configured_model_specs=configured_model_specs,
            refresh_model_type=refresh_model_type,
        )
        model_specs = [
            spec for spec in configured_model_specs if str(spec["type"]) == refresh_model_type
        ]
        print(f"Partial refresh: type={refresh_model_type}, base={merge_path}")
    else:
        merge_path = None
        model_specs = list(configured_model_specs)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    amp_dtype, amp_enabled = _resolve_eval_amp(cfg["amp_dtype"], device)
    comparison_window = (float(cfg["comparison_window_start"]), float(cfg["comparison_window_end"]))
    gallery_sizes = list(cfg["gallery_sizes"])
    n_trials = int(cfg["gallery_trials"])
    seed = int(cfg["seed"])

    print("=" * 60)
    print("Unified Ablation Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Input optical window: {comparison_window}  (source: {cfg.get('comparison_window_source', 'N/A')})")
    print(f"Gallery sizes: {gallery_sizes}")
    print(f"Models: {[spec['name'] for spec in model_specs]}")
    print(f"Gallery distractor pool: {cfg['neg_data_path']} [{cfg['neg_group']}]")
    if cfg["force_tutorial_distractors"]:
        print("Tutorial distractors forced on for retrieval/classification negatives.")

    _seed_all(seed)
    gw_source_types = load_gw_source_types(cfg["test_data_path"])
    gw_event_time_mjd_table = load_gw_event_time_mjd_table(cfg["test_data_path"], device, required=False)
    neg_gw_indices = load_negative_gw_indices(cfg["test_data_path"])

    neg_target_samples = _resolve_target_samples(cfg["n_neg_samples"])

    neg_optical_data = None
    if cfg["neg_data_path"] and os.path.exists(cfg["neg_data_path"]):
        neg_optical_data = load_negative_optical_samples(
            cfg["neg_data_path"],
            cfg["neg_group"],
            n_samples=neg_target_samples,
            seed=seed,
            require_zero_time_mjd_base=False,
            require_zero_time_mjd_cls_base=False,
            nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
            runtime_input_window_start=comparison_window[0],
            runtime_input_window_end=comparison_window[1],
            negative_sample_strategy=cfg.get("negative_sample_strategy", "block_random"),
            negative_sample_block_rows=cfg.get("negative_sample_block_rows", None),
            negative_sample_shuffle=cfg.get("negative_sample_shuffle", True),
        )
    if neg_optical_data is None:
        raise FileNotFoundError(
            f"Tutorial distractor dataset is required but unavailable: {cfg['neg_data_path']}"
        )

    positive_selection = str(cfg["positive_selection"]).strip().lower()

    if positive_selection == "batch":
        print(f"\n{'=' * 60}")
        print("Phase 1: Load and cache shared retrieval batches")
        print("=" * 60)
        cached_batches, used_num_workers = build_cached_batches(
            test_data_path=cfg["test_data_path"],
            neg_data_path=None,
            neg_group=cfg["neg_group"],
            comparison_window=comparison_window,
            batch_size=cfg["batch_size"],
            test_steps=cfg["test_steps"],
            num_workers=cfg["num_workers"],
            target_samples=neg_target_samples,
            nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
        )
        cfg["used_num_workers"] = int(used_num_workers)

        gw_indices_parts = []
        for batch_data in cached_batches:
            gw_indices_parts.append(batch_data[7].long().cpu())
        shared_gw_indices = torch.cat(gw_indices_parts)
        unique_sampled_gw = sorted(torch.unique(shared_gw_indices).cpu().tolist())
        sampled_positive_counts = np.asarray(
            [int((shared_gw_indices == int(gw_id)).sum().item()) for gw_id in unique_sampled_gw],
            dtype=np.int64,
        )
        sampled_positive_summary = {
            "n_unique_gw": int(len(unique_sampled_gw)),
            "n_sampled_positives": int(shared_gw_indices.numel()),
            "sampled_positives_per_gw": {
                "min": int(sampled_positive_counts.min()) if sampled_positive_counts.size else 0,
                "median": float(np.median(sampled_positive_counts)) if sampled_positive_counts.size else 0.0,
                "mean": float(np.mean(sampled_positive_counts)) if sampled_positive_counts.size else 0.0,
                "max": int(sampled_positive_counts.max()) if sampled_positive_counts.size else 0,
            },
        }
    else:
        cached_batches = []
        cfg["used_num_workers"] = cfg["num_workers"]
        sampled_positive_summary = None

    gw_positive_indices, all_test_gw_ids = load_test_positive_index_map(cfg["test_data_path"])
    if int(cfg["max_gw_events"]) > 0 and len(all_test_gw_ids) > int(cfg["max_gw_events"]):
        rng = np.random.default_rng(seed)
        selected_gw = sorted(
            int(v)
            for v in rng.choice(
                np.asarray(all_test_gw_ids, dtype=np.int64),
                size=int(cfg["max_gw_events"]),
                replace=False,
            ).tolist()
        )
        gw_positive_indices = {
            int(gw_id): gw_positive_indices[int(gw_id)]
            for gw_id in selected_gw
        }
        all_test_gw_ids = selected_gw
        print(f"Limited evaluation to {len(all_test_gw_ids)} GW events (max_gw_events={cfg['max_gw_events']}).")
    full_test_positive_counts = np.asarray(
        [int(np.asarray(gw_positive_indices[int(gw_id)]).size) for gw_id in all_test_gw_ids],
        dtype=np.int64,
    )
    gallery_positive_summary = {
        "n_unique_gw": int(len(all_test_gw_ids)),
        "n_total_positives": int(sum(int(count) for count in full_test_positive_counts.tolist())),
        "positives_per_gw": {
            "min": int(full_test_positive_counts.min()) if full_test_positive_counts.size else 0,
            "median": float(np.median(full_test_positive_counts)) if full_test_positive_counts.size else 0.0,
            "mean": float(np.mean(full_test_positive_counts)) if full_test_positive_counts.size else 0.0,
            "max": int(full_test_positive_counts.max()) if full_test_positive_counts.size else 0,
        },
    }
    gw_source_map = build_gw_source_map(gw_source_types, all_test_gw_ids)

    # --- Redshift metadata recovery from source catalogs ---
    redshift_metadata = None
    if cfg.get("redshift_analysis_enable"):
        redshift_metadata = _build_redshift_metadata_from_catalogs(
            cfg["test_data_path"],
            cfg["redshift_catalogs"],
        )
        print(f"Loaded redshift metadata for {len(redshift_metadata)} GW events from source catalogs")

    print(f"\n{'=' * 60}")
    print("Phase 2: Pre-generate shared galleries")
    print("=" * 60)
    gallery_candidate_mode = str(cfg["gallery_candidate_mode"]).strip().lower()
    gallery_gw_skymaps: Dict[int, torch.Tensor] = {}

    if gallery_candidate_mode in ("time_sky_hard", "synthetic_time_sky_hard", "tutorial_random"):
        # Build candidate sequences (shared across both modes for time_sky_hard)
        if gallery_candidate_mode == "time_sky_hard":
            candidate_sequences, gallery_gw_skymaps, _gallery_gw_times = build_time_sky_candidate_sequences(
                test_data_path=cfg["test_data_path"],
                unique_gw_ids=all_test_gw_ids,
                neg_optical_data=neg_optical_data,
                n_trials=n_trials,
                seed=seed,
                time_window_days=cfg["gallery_candidate_time_window_days"],
                credible_level_max=cfg["gallery_candidate_credible_level_max"],
                zero_time_field=cfg["nonkn_cls_base_field"],
            )
        elif gallery_candidate_mode == "synthetic_time_sky_hard":
            candidate_sequences, gallery_gw_skymaps, _gallery_gw_times = build_synthetic_time_sky_candidate_sequences(
                test_data_path=cfg["test_data_path"],
                unique_gw_ids=all_test_gw_ids,
                neg_optical_data=neg_optical_data,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                seed=seed,
                time_window_days=cfg["gallery_candidate_time_window_days"],
                credible_level_max=cfg["gallery_candidate_credible_level_max"],
            )
        else:
            candidate_sequences = {}

        if positive_selection == "batch":
            # --- Batch-based: one positive per GW from Phase 1 samples ---
            positive_query_bank, galleries, unique_gw = build_batch_positive_bank_and_galleries(
                cached_batches,
                candidate_sequences,
                gallery_sizes,
                n_trials,
                include_undersized=bool(cfg["gallery_include_undersized"]),
                seed=seed,
            )
            print(
                "  Built galleries with batch-sampled positives, "
                f"time_window=+0..{cfg['gallery_candidate_time_window_days']:.1f} days, "
                f"credible<= {cfg['gallery_candidate_credible_level_max']:.3f}, "
                f"include_undersized={bool(cfg['gallery_include_undersized'])}"
            )
        else:
            # --- Random: each trial randomly selects a positive per GW ---
            if gallery_candidate_mode in ("time_sky_hard", "synthetic_time_sky_hard"):
                galleries, unique_gw = build_prefixed_gallery_specs(
                    gw_positive_indices=gw_positive_indices,
                    candidate_sequences=candidate_sequences,
                    gallery_sizes=gallery_sizes,
                    n_trials=n_trials,
                    seed=seed,
                    include_undersized=bool(cfg["gallery_include_undersized"]),
                )
            else:
                galleries, unique_gw = precompute_tutorial_galleries(
                    gw_positive_indices,
                    int(neg_optical_data["times"].shape[0]),
                    gallery_sizes,
                    n_trials,
                    seed,
                )
            print(
                f"  Built {gallery_candidate_mode} galleries with "
                f"time_window=+0..{cfg['gallery_candidate_time_window_days']:.1f} days, "
                f"credible<= {cfg['gallery_candidate_credible_level_max']:.3f}, "
                f"include_undersized={bool(cfg['gallery_include_undersized'])}"
            )
            if not galleries:
                raise ValueError("Gallery construction produced no instances.")
            selected_positive_indices = np.unique(
                np.asarray([int(spec["positive_index"]) for spec in galleries.values()], dtype=np.int64)
            )
            positive_query_bank, positive_index_remap = load_selected_positive_bank(
                cfg["test_data_path"],
                selected_positive_indices,
                runtime_input_window_start=comparison_window[0],
                runtime_input_window_end=comparison_window[1],
            )
            galleries = remap_gallery_positive_indices(galleries, positive_index_remap)
    else:
        raise ValueError(
            f"Unsupported gallery_candidate_mode='{gallery_candidate_mode}'. "
            "Expected one of {'time_sky_hard', 'synthetic_time_sky_hard', 'tutorial_random'}."
        )

    if not galleries:
        raise ValueError("Gallery construction produced no instances.")

    positive_source_label = "cached test batches" if positive_selection == "batch" else "the full test set"
    print(
        "  Selected "
        f"{int(positive_query_bank['opt_t_raw'].shape[0])} positive optical queries "
        f"from {positive_source_label}"
    )
    print(f"  Built {len(galleries)} gallery instances for {len(unique_gw)} GW events")
    gallery_identity_summary = summarize_gallery_identity(galleries)
    print(f"  Gallery identity SHA256: {gallery_identity_summary['sha256']}")

    selected_gw_indices_np = positive_query_bank["gw_indices"].detach().cpu().numpy().astype(np.int64, copy=False)
    selected_unique_gw, selected_counts = np.unique(selected_gw_indices_np, return_counts=True)
    selected_positive_summary = {
        "source": positive_selection,
        "n_unique_gw": int(selected_unique_gw.size),
        "n_selected_positives": int(selected_gw_indices_np.size),
        "selected_positives_per_gw": {
            "min": int(selected_counts.min()) if selected_counts.size else 0,
            "median": float(np.median(selected_counts)) if selected_counts.size else 0.0,
            "mean": float(np.mean(selected_counts)) if selected_counts.size else 0.0,
            "max": int(selected_counts.max()) if selected_counts.size else 0,
        },
    }

    dt_bin_edges = parse_dt_bin_edges(cfg["dt_bin_edges"]) if cfg["report_dt_bins"] else None
    model_results: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    retrieval_rows: OrderedDict[str, Dict[str, float]] = OrderedDict()
    curve_rows: List[Dict[str, Any]] = []
    redshift_rows: List[Dict[str, Any]] = []
    redshift_macro_rows: List[Dict[str, Any]] = []
    model_names: List[str] = []

    for idx, model_spec in enumerate(model_specs, start=1):
        name = model_spec["name"]
        model_type = model_spec["type"]
        model_names.append(name)
        triplet_logits_for_output = None
        print(f"\n{'=' * 60}")
        print(f"Phase 3.{idx}: {name} ({model_type})")
        print("=" * 60)
        print(f"  Resolved checkpoint: {model_spec['resolved_checkpoint'] or 'N/A'}")
        if model_spec.get("resolved_config"):
            print(f"  Config: {model_spec['resolved_config']}")

        if model_type == "optical":
            model, ckpt_args = load_optical_model(model_spec["resolved_checkpoint"], device)
            optical_n_ref = int(ckpt_args.get("n_ref", 64))
            original_window = [
                float(ckpt_args.get("ref_start", comparison_window[0])),
                float(ckpt_args.get("ref_end", comparison_window[1])),
            ]
            ranks = score_all_galleries_optical(
                model,
                positive_query_bank,
                neg_optical_data,
                galleries,
                device=device,
                n_ref=optical_n_ref,
                ref_start=comparison_window[0],
                ref_end=comparison_window[1],
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            outcomes = enrich_gallery_outcomes(ranks, galleries)
            retrieval_metrics, retrieval_by_source, coverage_stats = aggregate_gallery_outcomes(
                outcomes=outcomes,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                unique_gw=unique_gw,
                gw_source_map=gw_source_map if gw_source_map else None,
            )
            model_results[name] = {
                "type": model_type,
                "scoring": "optical",
                "resolved_checkpoint": model_spec["resolved_checkpoint"],
                "resolved_config": model_spec.get("resolved_config"),
                "original_model_window": original_window,
                "comparison_window": list(comparison_window),
                "retrieval": retrieval_metrics,
                "retrieval_by_source": retrieval_by_source,
                "coverage": {key: float(val.get("coverage", 0.0)) for key, val in coverage_stats.items()},
                "effective_gallery_size_stats": coverage_stats,
                "classification_supported": False,
                "classification": None,
                "classification_by_source": {},
            }
            del model
        elif model_type == "multimodal":
            model, runtime_model_args, saved_args = load_multimodal_bundle(
                model_spec["resolved_checkpoint"],
                model_spec["resolved_config"],
                device,
                test_data_path=cfg["test_data_path"],
                neg_data_path=cfg["neg_data_path"],
                comparison_window=comparison_window,
                nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
            )
            embeddings = extract_optical_candidate_embeddings(
                model,
                positive_query_bank,
                device,
                n_ref=int(runtime_model_args.get("n_ref", 64)),
                ref_start=float(runtime_model_args["ref_start"]),
                ref_end=float(runtime_model_args["ref_end"]),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
                desc="  Extracting positive query embeddings",
            )
            negative_embeddings = extract_negative_gallery_embeddings(
                model,
                neg_optical_data,
                device,
                n_ref=int(runtime_model_args.get("n_ref", 64)),
                ref_start=float(runtime_model_args["ref_start"]),
                ref_end=float(runtime_model_args["ref_end"]),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            scoring_mode = _resolve_multimodal_scoring(model_spec, saved_args)
            time_delta_scoring_kwargs = candidate_time_delta_scoring_kwargs(
                scoring_mode,
                cfg["candidate_time_delta_mode"],
            )

            if scoring_mode == "contrastive":
                ranks = score_all_galleries_contrastive(
                    model,
                    embeddings,
                    negative_embeddings,
                    galleries,
                    unique_gw,
                    test_data_path=cfg["test_data_path"],
                    device=device,
                    model_args=runtime_model_args,
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                    **time_delta_scoring_kwargs,
                )
            else:
                ranks = score_all_galleries_multimodal(
                    model,
                    embeddings,
                    negative_embeddings,
                    galleries,
                    unique_gw,
                    test_data_path=cfg["test_data_path"],
                    device=device,
                    model_args=runtime_model_args,
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                    **time_delta_scoring_kwargs,
                )
            outcomes = enrich_gallery_outcomes(ranks, galleries)
            retrieval_metrics, retrieval_by_source, coverage_stats = aggregate_gallery_outcomes(
                outcomes=outcomes,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                unique_gw=unique_gw,
                gw_source_map=gw_source_map if gw_source_map else None,
            )

            if scoring_mode == "contrastive":
                print("  Skipping triplet classification (fusion classifier untrained for contrastive-only model)")
                classification_metrics = None
                classification_by_source = {}
                classification_supported = False
            elif not bool(cfg["compute_classification_metrics"]):
                print("  Skipping triplet classification (compute_classification_metrics=false)")
                classification_metrics = None
                classification_by_source = {}
                classification_supported = False
            else:
                print("  Computing triplet classification metrics...")
                classification_metrics, classification_by_source, triplet_logits_for_output = evaluate_multimodal_classification(
                    model,
                    runtime_model_args,
                    saved_args,
                    seed=seed,
                    test_data_path=cfg["test_data_path"],
                    neg_data_path=cfg["neg_data_path"],
                    neg_group=cfg["neg_group"],
                    batch_size=cfg["batch_size"],
                    test_steps=cfg["test_steps"],
                    num_workers=cfg["num_workers"],
                    n_neg_samples=neg_target_samples,
                    comparison_window=comparison_window,
                    nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
                    report_dt_bins=cfg["report_dt_bins"],
                    dt_bin_edges=dt_bin_edges,
                    report_dt_macro=cfg["report_dt_macro"],
                    neg_optical_data=neg_optical_data,
                    neg_gw_indices=neg_gw_indices,
                    gw_source_types=gw_source_types,
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    shared_cfg=cfg,
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                    device=device,
                    return_logits=True,
                )
                classification_supported = True
            model_results[name] = {
                "type": model_type,
                "scoring": scoring_mode,
                "resolved_checkpoint": model_spec["resolved_checkpoint"],
                "resolved_config": model_spec.get("resolved_config"),
                "original_model_window": [
                    float(runtime_model_args.get("original_ref_start", comparison_window[0])),
                    float(runtime_model_args.get("original_ref_end", comparison_window[1])),
                ],
                "comparison_window": list(comparison_window),
                "retrieval": retrieval_metrics,
                "retrieval_by_source": retrieval_by_source,
                "coverage": {key: float(val.get("coverage", 0.0)) for key, val in coverage_stats.items()},
                "effective_gallery_size_stats": coverage_stats,
                "classification_supported": classification_supported,
                "classification": classification_metrics,
                "classification_by_source": classification_by_source,
                "candidate_time_delta_mode": cfg["candidate_time_delta_mode"],
                "effective_input_window_metadata": runtime_model_args.get("effective_input_window_metadata"),
            }
            del negative_embeddings, embeddings, model
        elif model_type == "fink_rf":
            artifact = load_fink_rf_bundle(model_spec["resolved_checkpoint"])
            artifact_cfg = dict(artifact.get("config", {}))
            if (
                os.path.realpath(str(artifact_cfg.get("neg_train_path", ""))) == os.path.realpath(str(cfg["neg_data_path"]))
                and str(artifact_cfg.get("neg_train_group", "")) == str(cfg["neg_group"])
            ):
                raise ValueError(
                    "Fink RF leakage guard: the negative training source is also the retrieval gallery source."
                )
            fink_chunk_size = int(model_spec.get("chunk_size") or artifact.get("config", {}).get("batch_size", 2048))
            outcomes, fink_diagnostics = score_all_galleries_fink_rf(
                artifact,
                positive_query_bank,
                neg_optical_data,
                galleries,
                positive_attrs=_read_h5_attrs(cfg["test_data_path"]),
                negative_attrs=_read_h5_attrs(cfg["neg_data_path"]),
                chunk_size=fink_chunk_size,
            )
            retrieval_metrics, retrieval_by_source, coverage_stats = aggregate_gallery_outcomes(
                outcomes=outcomes,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                unique_gw=unique_gw,
                gw_source_map=gw_source_map if gw_source_map else None,
            )
            model_results[name] = {
                "type": model_type,
                "scoring": "fink_rf",
                "resolved_checkpoint": model_spec["resolved_checkpoint"],
                "resolved_config": model_spec.get("resolved_config"),
                "original_model_window": artifact.get("config", {}).get("fit_window_days"),
                "comparison_window": list(comparison_window),
                "retrieval": retrieval_metrics,
                "retrieval_by_source": retrieval_by_source,
                "coverage": {key: float(val.get("coverage", 0.0)) for key, val in coverage_stats.items()},
                "effective_gallery_size_stats": coverage_stats,
                "classification_supported": False,
                "classification": None,
                "classification_by_source": {},
                "fink_rf_diagnostics": fink_diagnostics,
            }
        elif model_type == "skymap":
            if not _gallery_mode_supports_skymap_only(gallery_candidate_mode):
                raise ValueError(
                    "skymap-only model requires gallery_candidate_mode to provide "
                    "time/sky credible-level galleries."
                )
            outcomes = score_all_galleries_skymap(
                positive_bank={"opt_coords": positive_query_bank["opt_coords"]},
                galleries=galleries,
                gw_skymaps=gallery_gw_skymaps,
            )
            retrieval_metrics, retrieval_by_source, coverage_stats = aggregate_gallery_outcomes(
                outcomes=outcomes,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                unique_gw=unique_gw,
                gw_source_map=gw_source_map if gw_source_map else None,
            )
            model_results[name] = {
                "type": model_type,
                "scoring": "skymap",
                "resolved_checkpoint": None,
                "resolved_config": None,
                "original_model_window": None,
                "comparison_window": list(comparison_window),
                "retrieval": retrieval_metrics,
                "retrieval_by_source": retrieval_by_source,
                "coverage": {key: float(val.get("coverage", 0.0)) for key, val in coverage_stats.items()},
                "effective_gallery_size_stats": coverage_stats,
                "classification_supported": False,
                "classification": None,
                "classification_by_source": {},
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if retrieval_writer is not None:
            retrieval_writer.writerows(retrieval_outcome_rows(
                seed=seed,
                model=name,
                outcomes=outcomes,
                gw_source_map=gw_source_map,
                redshift_metadata=redshift_metadata,
            ))
        if classification_writer is not None and triplet_logits_for_output is not None:
            classification_writer.writerows(classification_prediction_rows(
                seed=seed,
                model=name,
                triplet_logits=triplet_logits_for_output,
            ))

        retrieval_by_source = model_results[name].get("retrieval_by_source", {})
        if {"bns", "nsbh"}.issubset(
            {str(source).strip().lower() for source in retrieval_by_source}
        ):
            source_macro, source_gap = compute_source_macro_and_gap(
                retrieval_by_source
            )
        else:
            source_macro, source_gap = {}, {}
        model_results[name]["retrieval_source_macro"] = source_macro
        model_results[name]["retrieval_source_gap"] = source_gap

        retrieval_rows[name] = model_results[name]["retrieval"]
        model_curve_rows = build_curve_rows(
            method_name=name,
            gallery_sizes=gallery_sizes,
            retrieval_metrics=model_results[name]["retrieval"],
            coverage_stats=model_results[name]["effective_gallery_size_stats"],
        )
        model_results[name]["curve_rows"] = model_curve_rows
        curve_rows.extend(model_curve_rows)

        if redshift_metadata is not None:
            model_redshift_rows = _aggregate_redshift_metrics(
                outcomes=outcomes,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                unique_gw=unique_gw,
                redshift_metadata=redshift_metadata,
                bin_edges=cfg["redshift_bin_edges"],
                bin_labels=cfg["redshift_bin_labels"],
                method_name=name,
            )
            redshift_rows.extend(model_redshift_rows)

        for gallery_size in gallery_sizes:
            coverage_info = model_results[name]["effective_gallery_size_stats"].get(f"gallery_{gallery_size}", {})
            r1 = model_results[name]["retrieval"].get(f"gallery_{gallery_size}_recall_at_1", 0.0)
            r5 = model_results[name]["retrieval"].get(f"gallery_{gallery_size}_recall_at_5", 0.0)
            r10 = model_results[name]["retrieval"].get(f"gallery_{gallery_size}_recall_at_10", 0.0)
            mrr = model_results[name]["retrieval"].get(f"gallery_{gallery_size}_mrr", 0.0)
            print(
                f"  gallery={gallery_size}  R@1={r1:.4f}  R@5={r5:.4f}  "
                f"R@10={r10:.4f}  MRR={mrr:.4f}  "
                f"coverage={coverage_info.get('coverage', 0.0):.3f}  "
                f"effN={coverage_info.get('effective_gallery_size_mean', 0.0):.1f}"
            )
        if model_results[name]["classification_supported"] and model_results[name]["classification"]:
            cls = model_results[name]["classification"]
            print(
                "  classification "
                f"AUROC={cls.get('auroc', 0.0):.4f} "
                f"AUPRC={cls.get('auprc', 0.0):.4f} "
                f"F1={cls.get('f1_optimal', 0.0):.4f}"
            )
        if source_macro:
            print(
                "  source-macro retrieval available for "
                f"{len(source_macro)} metric(s); source gaps recorded in JSON"
            )
        torch.cuda.empty_cache()
        gc.collect()

    if base_output is not None:
        for summary_key, fresh_summary in (
            ("gallery_positive_summary", gallery_positive_summary),
            ("selected_positive_summary", selected_positive_summary),
        ):
            base_summary = base_output.get(summary_key)
            if base_summary is not None and base_summary != fresh_summary:
                raise ValueError(
                    f"Partial-refresh {summary_key} mismatch; refusing to merge different gallery populations."
                )
        (
            model_results,
            retrieval_rows,
            curve_rows,
            redshift_rows,
            model_names,
        ) = merge_partial_model_results(
            base_output=base_output,
            fresh_model_results=model_results,
            fresh_curve_rows=curve_rows,
            fresh_redshift_rows=redshift_rows,
            configured_model_specs=configured_model_specs,
            refresh_model_type=refresh_model_type,
        )

    print(f"\n{'=' * 60}")
    print("Results: Overall Retrieval")
    print("=" * 60)
    print_unified_table(retrieval_rows, gallery_sizes, model_names)
    if not cfg["no_latex"]:
        print_latex_table(retrieval_rows, gallery_sizes, model_names)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_retrieval_curves(curve_rows, output_dir)
    redshift_macro_rows = []
    if redshift_rows:
        write_redshift_csv(redshift_rows, output_dir / "redshift_metrics.csv")
        redshift_macro_rows = aggregate_redshift_macro_metrics(redshift_rows)
        write_redshift_macro_csv(redshift_macro_rows, output_dir / "redshift_macro_metrics_log10_weighted.csv")
        plot_redshift_metrics(redshift_rows, output_dir)
        plot_redshift_macro_metrics(redshift_macro_rows, output_dir)
        print(f"Wrote redshift-binned outputs to {output_dir}")

    baseline_comparison = None
    if cfg["baseline_results_path"]:
        baseline_output = _load_json(Path(cfg["baseline_results_path"]))
        validate_time_delta_baseline_compatibility(baseline_output, cfg)
        if baseline_output.get("gallery_positive_summary") != gallery_positive_summary:
            raise ValueError("Baseline gallery-positive population does not match this run.")
        if baseline_output.get("selected_positive_summary") != selected_positive_summary:
            raise ValueError("Baseline selected-positive population does not match this run.")
        if len(model_names) != 1:
            raise ValueError("Time-delta baseline comparison requires exactly one evaluated model.")
        model_name = str(model_names[0])
        baseline_model = baseline_output.get("models", {}).get(model_name)
        ablated_model = model_results.get(model_name)
        if not isinstance(baseline_model, Mapping) or not isinstance(ablated_model, Mapping):
            raise ValueError(f"Baseline/current results must both contain model {model_name!r}.")
        metric_delta_rows = build_time_delta_metric_delta_rows(baseline_model, ablated_model)
        redshift_delta_rows = build_time_delta_redshift_delta_rows(
            baseline_output.get("redshift_rows", []),
            redshift_rows,
            method_name=model_name,
        )
        write_rows_csv(
            metric_delta_rows,
            output_dir / "time_delta_ablation_metric_deltas.csv",
        )
        write_rows_csv(
            redshift_delta_rows,
            output_dir / "time_delta_ablation_redshift_deltas.csv",
        )
        plot_time_delta_ablation(metric_delta_rows, output_dir)
        plot_time_delta_redshift_ablation(redshift_delta_rows, output_dir)
        baseline_comparison = {
            "baseline_results_path": cfg["baseline_results_path"],
            "model": model_name,
            "baseline_candidate_time_delta_mode": baseline_output.get("config", {}).get(
                "candidate_time_delta_mode", "native"
            ),
            "current_candidate_time_delta_mode": cfg["candidate_time_delta_mode"],
            "gallery_compatibility": {
                "config_fields": list(_BASELINE_COMPARISON_FIELDS),
                "positive_summaries_match": True,
                "identity_basis": "deterministic replay from identical seed and gallery config",
                "current_gallery_identity": gallery_identity_summary,
            },
            "metric_delta_rows": metric_delta_rows,
            "redshift_delta_rows": redshift_delta_rows,
        }

    output = {
        "table": {
            "gallery_sizes": gallery_sizes,
            "metric_labels": TABLE_METRIC_LABELS,
            "rows": build_table_rows(retrieval_rows, gallery_sizes, model_names),
        },
        "curve_rows": curve_rows,
        "redshift_rows": redshift_rows,
        "redshift_macro_rows": redshift_macro_rows,
        "redshift_config": {
            "enabled": bool(cfg.get("redshift_analysis_enable", False)),
            "catalogs": cfg.get("redshift_catalogs", {}),
            "bin_edges": cfg.get("redshift_bin_edges"),
            "bin_labels": cfg.get("redshift_bin_labels"),
            "macro_gallery_weight_scheme": "log10(gallery_size)",
        } if cfg.get("redshift_analysis_enable") else None,
        "models": dict(model_results),
        "sampled_positive_summary": sampled_positive_summary,
        "gallery_positive_summary": gallery_positive_summary,
        "selected_positive_summary": selected_positive_summary,
        "gallery_identity": gallery_identity_summary,
        "baseline_comparison": baseline_comparison,
        "partial_refresh": {
            "base_results": str(merge_path),
            "refreshed_model_type": refresh_model_type,
            "tie_policy": "uniform_random_expected" if refresh_model_type == "fink_rf" else None,
        } if base_output is not None else None,
        "config": {
            "input_config": cfg["input_config"],
            "test_data_path": cfg["test_data_path"],
            "neg_data_path": cfg["neg_data_path"],
            "neg_group": cfg["neg_group"],
            "requested_neg_data_path": cfg["requested_neg_data_path"],
            "requested_neg_group": cfg["requested_neg_group"],
            "tutorial_neg_data_path": cfg["tutorial_neg_data_path"],
            "tutorial_neg_group": cfg["tutorial_neg_group"],
            "force_tutorial_distractors": cfg["force_tutorial_distractors"],
            "output_dir": str(output_dir),
            "device": str(device),
            "amp_dtype": cfg["amp_dtype"],
            "seed": seed,
            "batch_size": cfg["batch_size"],
            "test_steps": cfg["test_steps"],
            "num_workers": cfg["num_workers"],
            "used_num_workers": cfg.get("used_num_workers", cfg["num_workers"]),
            "gallery_sizes": gallery_sizes,
            "gallery_trials": n_trials,
            "gallery_candidate_mode": cfg["gallery_candidate_mode"],
            "gallery_candidate_time_window_days": cfg["gallery_candidate_time_window_days"],
            "gallery_candidate_credible_level_max": cfg["gallery_candidate_credible_level_max"],
            "gallery_include_undersized": cfg["gallery_include_undersized"],
            "positive_selection": cfg["positive_selection"],
            "max_gw_events": cfg["max_gw_events"],
            "comparison_window": list(comparison_window),
            "n_neg_samples": cfg["n_neg_samples"],
            "negative_sample_strategy": cfg["negative_sample_strategy"],
            "negative_sample_block_rows": cfg["negative_sample_block_rows"],
            "negative_sample_shuffle": cfg["negative_sample_shuffle"],
            "compute_classification_metrics": cfg["compute_classification_metrics"],
            "save_outcomes": cfg["save_outcomes"],
            "strict_output_safety": cfg["strict_output_safety"],
            "experiment_id": cfg["experiment_id"],
            "candidate_time_delta_mode": cfg["candidate_time_delta_mode"],
            "baseline_results_path": cfg["baseline_results_path"],
            "nonkn_cls_base_field": cfg["nonkn_cls_base_field"],
            "report_dt_bins": cfg["report_dt_bins"],
            "dt_bin_edges": dt_bin_edges if dt_bin_edges is not None else None,
            "report_dt_macro": cfg["report_dt_macro"],
            "hard_negative_strategy": cfg["hard_negative_strategy"],
            "n_unique_gw": gallery_positive_summary["n_unique_gw"],
            "n_gallery_positives": gallery_positive_summary["n_total_positives"],
            "gallery_positives_per_gw": gallery_positive_summary["positives_per_gw"],
            "n_sampled_positives": sampled_positive_summary["n_sampled_positives"] if sampled_positive_summary else None,
            "n_sampled_positive_gw": sampled_positive_summary["n_unique_gw"] if sampled_positive_summary else None,
            "sampled_positives_per_gw": sampled_positive_summary["sampled_positives_per_gw"] if sampled_positive_summary else None,
            "n_selected_positives": selected_positive_summary["n_selected_positives"],
            "n_selected_positive_gw": selected_positive_summary["n_unique_gw"],
            "selected_positives_per_gw": selected_positive_summary["selected_positives_per_gw"],
            "models": [
                {
                    "name": spec["name"],
                    "type": spec["type"],
                    "scoring": spec.get("scoring"),
                    "checkpoint": spec["checkpoint"],
                    "config": spec.get("config"),
                    "resolved_checkpoint": spec["resolved_checkpoint"],
                    "resolved_config": spec.get("resolved_config"),
                }
                for spec in configured_model_specs
            ],
        },
    }
    output_path = output_dir / "ablation_comparison.json"
    write_json_atomic(output_path, output)
    completed_artifacts = [output_path.name]
    if retrieval_writer is not None:
        retrieval_writer.commit()
        completed_artifacts.append("retrieval_outcomes.csv.gz")
    if classification_writer is not None:
        classification_writer.commit()
        completed_artifacts.append("classification_predictions.csv.gz")
    if run_manifest is not None:
        mark_run_success(output_dir, run_manifest, completed_artifacts)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
