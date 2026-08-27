from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval.eval_gw_kn_pairing_sensitivity import (
    aggregate_pair_metrics,
    build_candidate_layout,
    build_event_pairs,
    build_single_parameter_event_pairs,
    crossed_interaction,
    derive_physical_features,
    derive_single_parameter_features,
    holm_adjust,
    optical_sampling_features,
    pair_curve_realizations,
    robust_standardize,
    single_parameter_dose_response,
    summarize_pair_metrics,
    summarize_single_parameter_metrics,
)


def test_derive_physical_features_uses_model_visible_distance_and_intrinsics() -> None:
    scalars = np.asarray(
        [
            [2.0, 1.0, 0.4, -0.2, -0.5, 0.2, 0.03],
            [1.5, 1.5, 0.1, 0.1, 0.25, 0.4, 0.04],
        ]
    )
    features = derive_physical_features(scalars)

    expected_chirp = (2.0 * 1.0) ** (3.0 / 5.0) / 3.0 ** (1.0 / 5.0)
    assert features.shape == (2, 5)
    assert features[0, 0] == pytest.approx(expected_chirp)
    assert features[0, 1] == pytest.approx(0.5)
    assert features[0, 2] == pytest.approx(0.2)
    assert features[0, 3] == pytest.approx(0.5)
    assert features[0, 4] == pytest.approx(np.log10(0.2))


def test_sampling_features_depend_only_on_times_and_masks() -> None:
    times = np.asarray([[0.0, 0.1, 0.3], [0.0, 0.05, 0.1]], dtype=float)
    masks = np.zeros((2, 3, 2), dtype=float)
    masks[0, 0, 0] = 1
    masks[0, 1, 1] = 1
    masks[0, 2, :] = 1  # outside the comparison window
    masks[1, :, 0] = 1

    features = optical_sampling_features(
        times, masks, window_start=-0.1, window_end=0.2
    )

    assert features[0, 0] == pytest.approx(np.log1p(2))
    assert features[0, 1] == pytest.approx(1.0)
    assert features[0, 2] == pytest.approx(np.log1p(0.1))
    assert features[1, 0] == pytest.approx(np.log1p(3))
    assert features[1, 1] == pytest.approx(0.5)


def test_robust_standardize_rejects_nonfinite_and_handles_constant_columns() -> None:
    values = np.asarray([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]])
    standardized, center, scale = robust_standardize(values)
    assert np.all(np.isfinite(standardized))
    assert center.tolist() == [2.0, 4.0]
    assert scale[1] == 1.0
    with pytest.raises(ValueError, match="non-finite"):
        robust_standardize(np.asarray([[1.0, np.nan]]))


def test_event_pairing_is_same_source_disjoint_and_deterministic() -> None:
    gw_ids = np.arange(12, dtype=np.int64)
    sources = np.asarray(["bns"] * 6 + ["nsbh"] * 6)
    physical = np.column_stack(
        [np.arange(12, dtype=float), np.zeros((12, 4), dtype=float)]
    )
    nuisance = np.zeros((12, 4), dtype=float)
    kwargs = {
        "gw_ids": gw_ids,
        "source_types": sources,
        "physical_features": physical,
        "nuisance_features": nuisance,
        "physical_distance_quantile": 0.5,
        "nuisance_l1_max": 3.0,
        "nuisance_feature_max_abs": 1.0,
        "min_pairs_per_source": 2,
    }

    pairs, audit = build_event_pairs(**kwargs)
    repeated, repeated_audit = build_event_pairs(**kwargs)

    assert pairs == repeated
    assert audit == repeated_audit
    assert audit["pair_counts"] == {"bns": 2, "nsbh": 2}
    used = []
    for pair in pairs:
        assert sources[pair["gw_a"]] == sources[pair["gw_b"]]
        assert pair["gw_a"] != pair["gw_b"]
        assert pair["physical_distance"] >= pair["physical_distance_threshold"]
        used.extend([pair["gw_a"], pair["gw_b"]])
    assert len(used) == len(set(used))


def test_curve_pairing_uses_sampling_features_without_reuse() -> None:
    event_pairs = [
        {
            "pair_id": "bns_0000",
            "source_type": "bns",
            "gw_a": 0,
            "gw_b": 1,
        }
    ]
    parent_to_curves = {
        0: np.asarray([0, 1, 2]),
        1: np.asarray([3, 4, 5]),
    }
    curve_features = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [2.1, 0.0, 0.0],
        ]
    )
    records, audit = pair_curve_realizations(
        event_pairs=event_pairs,
        parent_to_curves=parent_to_curves,
        curve_features=curve_features,
        curve_source_types=["bns"] * 6,
        max_curve_pairs=2,
    )

    assert len(records) == 2
    assert len({row["optical_index_a"] for row in records}) == 2
    assert len({row["optical_index_b"] for row in records}) == 2
    assert audit["curve_pairs_per_event_pair"]["max"] == 2


def test_crossed_interaction_cancels_additive_gw_and_optical_scores() -> None:
    gw_a, gw_b = 10.0, -3.0
    optical_a, optical_b = 7.0, 2.0
    additive = crossed_interaction(
        gw_a + optical_a,
        gw_a + optical_b,
        gw_b + optical_a,
        gw_b + optical_b,
    )
    compatible = crossed_interaction(
        gw_a + optical_a + 6.0,
        gw_a + optical_b,
        gw_b + optical_a,
        gw_b + optical_b + 6.0,
    )

    assert additive["interaction"] == pytest.approx(0.0)
    assert additive["directional_win_rate"] == pytest.approx(0.5)
    assert compatible["interaction"] == pytest.approx(6.0)
    assert compatible["directional_win_rate"] == pytest.approx(1.0)


def test_candidate_layout_shares_each_anchor_across_all_four_cells() -> None:
    curve_pairs = [
        {
            "curve_pair_id": 0,
            "gw_a": 2,
            "gw_b": 3,
            "optical_index_a": 10,
            "optical_index_b": 11,
        }
    ]
    coordinates = np.zeros((12, 2), dtype=np.float32)
    coordinates[10] = [1.0, 2.0]
    coordinates[11] = [3.0, 4.0]
    first = np.zeros(12)
    first[10] = 105.0
    first[11] = 210.0
    event = np.zeros(4)
    event[2] = 100.0
    event[3] = 200.0

    indices, coords, dt_days, slots = build_candidate_layout(
        curve_pairs,
        compact_index={10: 0, 11: 1},
        coordinates=coordinates,
        first_detection_mjd=first,
        event_time_mjd=event,
    )

    assert indices.tolist() == [0, 1, 0, 1]
    assert np.array_equal(coords[:2], [[1.0, 2.0], [1.0, 2.0]])
    assert np.array_equal(coords[2:], [[3.0, 4.0], [3.0, 4.0]])
    assert dt_days.tolist() == [5.0, 5.0, 10.0, 10.0]
    assert slots == [(0, "a"), (0, "b")]


def test_pair_aggregation_and_model_comparison_share_identical_pairs() -> None:
    event_pairs = [
        {
            "pair_id": "bns_0000",
            "source_type": "bns",
            "gw_a": 0,
            "gw_b": 1,
            "physical_distance": 2.0,
        },
        {
            "pair_id": "nsbh_0000",
            "source_type": "nsbh",
            "gw_a": 2,
            "gw_b": 3,
            "physical_distance": 3.0,
        },
    ]
    interaction_rows = []
    for model, interaction in (
        ("Mixed Gallery v1", 2.0),
        ("Default MAGIKS", 1.0),
        ("Optical-only", 0.0),
    ):
        for pair in event_pairs:
            for anchor in ("a", "b"):
                interaction_rows.append(
                    {
                        "model": model,
                        "model_type": (
                            "optical" if model == "Optical-only" else "multimodal"
                        ),
                        "pair_id": pair["pair_id"],
                        "curve_pair_id": 0,
                        "source_type": pair["source_type"],
                        "anchor": anchor,
                        "physical_distance": pair["physical_distance"],
                        "interaction": interaction,
                        "directional_win_rate": 0.5 if interaction == 0 else 1.0,
                        "margin_a": interaction,
                        "margin_b": interaction,
                    }
                )
    pair_rows = aggregate_pair_metrics(interaction_rows, event_pairs)
    summary = summarize_pair_metrics(
        pair_rows,
        new_model_name="Mixed Gallery v1",
        baseline_model_name="Default MAGIKS",
        bootstrap_samples=100,
        permutation_samples=100,
        seed=42,
    )

    comparison = next(row for row in summary if row["endpoint"] == "new_minus_baseline")
    optical = next(
        row
        for row in summary
        if row["endpoint"] == "absolute_sensitivity"
        and row["model"] == "Optical-only"
        and row["source"] == "source_macro"
    )
    assert comparison["mean_interaction"] == pytest.approx(1.0)
    assert optical["mean_interaction"] == pytest.approx(0.0)


def test_single_parameter_features_preserve_component_spin_association() -> None:
    scalars = np.asarray(
        [
            [2.0, 1.0, 0.4, -0.2, 0.5, 0.2, 0.03],
            [1.0, 2.0, -0.2, 0.4, -0.5, 0.2, 0.03],
        ]
    )
    features = derive_single_parameter_features(scalars)

    np.testing.assert_allclose(features["chi_eff"], [0.2, 0.2])
    np.testing.assert_allclose(features["primary_spin_z"], [0.4, 0.4])
    np.testing.assert_allclose(features["abs_costheta"], [0.5, 0.5])


def _hypercube_single_parameter_fixture():
    cube = np.asarray(
        [[2.0 * float((row >> column) & 1) for column in range(5)] for row in range(32)]
    )
    gw_ids = np.arange(64, dtype=np.int64)
    sources = np.asarray(["bns"] * 32 + ["nsbh"] * 32)
    stacked = np.vstack([cube, cube])
    features = {
        "chirp_mass_detector": stacked[:, 0],
        "mass_ratio": stacked[:, 1],
        "chi_eff": np.concatenate([cube[:, 2], np.zeros(32)]),
        "primary_spin_z": np.concatenate([np.zeros(32), cube[:, 2]]),
        "abs_costheta": stacked[:, 3],
        "log10_distance_gpc": stacked[:, 4],
    }
    return gw_ids, sources, features


def test_single_parameter_pairing_is_deterministic_disjoint_and_balanced() -> None:
    gw_ids, sources, features = _hypercube_single_parameter_fixture()
    kwargs = {
        "gw_ids": gw_ids,
        "source_types": sources,
        "feature_values": features,
        "nuisance_features": np.zeros((64, 4)),
        "target_min_abs_iqr": 1.0,
        "other_calipers_iqr": [0.25, 0.5, 0.75],
        "primary_caliper_iqr": 0.5,
        "min_pairs_by_caliper": {0.25: 1, 0.5: 1, 0.75: 1},
        "nuisance_l1_max": 3.0,
        "nuisance_feature_max_abs": 1.0,
    }
    pairs, audit = build_single_parameter_event_pairs(**kwargs)
    repeated, repeated_audit = build_single_parameter_event_pairs(**kwargs)

    assert pairs == repeated
    assert audit == repeated_audit
    assert len(audit["conditions"]) == 30
    assert {
        row["target_parameter"] for row in pairs if row["source_type"] == "bns"
    } == {
        "chirp_mass_detector",
        "mass_ratio",
        "chi_eff",
        "abs_costheta",
        "log10_distance_gpc",
    }
    assert {
        row["target_parameter"] for row in pairs if row["source_type"] == "nsbh"
    } == {
        "chirp_mass_detector",
        "mass_ratio",
        "primary_spin_z",
        "abs_costheta",
        "log10_distance_gpc",
    }
    by_condition: dict[str, list[dict]] = {}
    for row in pairs:
        by_condition.setdefault(row["condition_id"], []).append(row)
        assert row["target_delta_iqr"] >= 1.0
        assert row["other_parameter_max_abs_difference"] <= row["caliper_iqr"]
        assert row["nuisance_max_abs_difference"] <= 1.0
    for condition_pairs in by_condition.values():
        used = [
            int(value)
            for row in condition_pairs
            for value in (row["gw_a"], row["gw_b"])
        ]
        assert len(used) == len(set(used))


def test_holm_adjust_matches_step_down_definition() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def _single_parameter_metric_fixture() -> list[dict]:
    target_sources = {
        "chirp_mass_detector": ("bns", "nsbh"),
        "mass_ratio": ("bns", "nsbh"),
        "abs_costheta": ("bns", "nsbh"),
        "log10_distance_gpc": ("bns", "nsbh"),
        "chi_eff": ("bns",),
        "primary_spin_z": ("nsbh",),
    }
    rows = []
    for target, target_source_types in target_sources.items():
        for source in target_source_types:
            pair_id = f"{target}_{source}"
            for model, interaction, win_rate in (
                ("Mixed Gallery v1", 2.0, 1.0),
                ("Default MAGIKS", 1.0, 0.75),
                ("Optical-only", 0.0, 0.5),
            ):
                rows.append(
                    {
                        "model": model,
                        "pair_id": pair_id,
                        "condition_id": pair_id,
                        "target_parameter": target,
                        "caliper_iqr": 0.5,
                        "source_type": source,
                        "interaction": interaction,
                        "directional_win_rate": win_rate,
                    }
                )
    return rows


def test_single_parameter_summary_builds_six_endpoint_holm_families() -> None:
    summary = summarize_single_parameter_metrics(
        _single_parameter_metric_fixture(),
        new_model_name="Mixed Gallery v1",
        baseline_model_name="Default MAGIKS",
        primary_caliper_iqr=0.5,
        bootstrap_samples=40,
        permutation_samples=40,
        seed=42,
    )
    primary = [row for row in summary if row["family"]]

    assert len({row["family"] for row in primary}) == 4
    assert all(
        sum(candidate["family"] == row["family"] for candidate in primary) == 6
        for row in primary
    )
    assert all(row["holm_adjusted_p"] != "" for row in primary)
    directional_delta = next(
        row
        for row in summary
        if row["endpoint"] == "new_minus_baseline"
        and row["metric"] == "directional_win_rate_delta"
        and row["target_parameter"] == "chi_eff"
    )
    assert directional_delta["estimate"] == pytest.approx(0.25)


def test_dose_response_uses_equal_count_bins_even_with_ties() -> None:
    rows = []
    for source in ("bns", "nsbh"):
        for pair_index, target_delta in enumerate([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]):
            for model, interaction in (
                ("Mixed Gallery v1", 1.0),
                ("Default MAGIKS", 0.5),
                ("Optical-only", 0.0),
            ):
                rows.append(
                    {
                        "model": model,
                        "pair_id": f"{source}_{pair_index}",
                        "target_parameter": "chirp_mass_detector",
                        "source_type": source,
                        "caliper_iqr": 0.5,
                        "target_delta_iqr": target_delta,
                        "interaction": interaction,
                        "directional_win_rate": 0.5 if interaction == 0 else 1.0,
                    }
                )
    dose, trends = single_parameter_dose_response(
        rows, primary_caliper_iqr=0.5, n_bins=3
    )

    source_rows = [
        row
        for row in dose
        if row["source"] in {"bns", "nsbh"} and row["model"] == "Mixed Gallery v1"
    ]
    assert len(source_rows) == 6
    assert {row["n_pairs"] for row in source_rows} == {2}
    assert any(row["source"] == "source_macro" for row in dose)
    assert all(np.isfinite(row["spearman_target_delta_interaction"]) for row in trends)
