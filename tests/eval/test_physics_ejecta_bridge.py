from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from Model.scripts.eval.eval_gw_kn_directional_bridge import (
    BRIDGE_NAME,
    DOSE_BIN_LABELS,
    GW_BLIND_NAME,
    PLOT_AXIS_SCALE,
    PLOT_FONT_SCALE,
    PLOT_LATEX_PREAMBLE,
    PLOT_LEGEND_SCALE,
    PLOT_MODEL_LABELS,
    PLOT_MODEL_MARKERS,
    PLOT_PARAMETER_LABELS,
    aggregate_pair_dwr,
    decision_row,
    normalise_v2_config,
    summarize_dwr,
)
from Model.scripts.eval.fit_physics_ejecta_bridge import (
    _prepare_neighbor_samples,
    _read_selected_rows,
    _select_curves_fast,
    fit_bridge,
)
from Model.scripts.eval.fit_physics_ejecta_bridge import (
    normalise_config as normalise_fit_config,
)
from Model.scripts.eval.physics_ejecta_bridge import (
    LIGHTCURVE_FEATURE_NAMES,
    PhysicsEjectaBridge,
    bhattacharyya_score,
    cadence_candidate_mask,
    derive_gw_feature_matrix,
    deterministic_event_split,
    extract_lightcurve_features,
    luptitude_to_flux_sigma,
    masked_feature_distances,
    select_curves_per_event,
    select_unique_neighbors,
    weighted_gaussian,
)


def test_v2_config_rejects_interaction_comparison(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    with pytest.raises(ValueError, match="does not support interaction"):
        normalise_v2_config(
            {
                "primary_metric": "directional_win_rate",
                "compare_interaction": True,
            },
            config,
        )


def test_v2_config_allows_one_neural_model_with_bridge_and_zero_baseline(
    tmp_path: Path,
) -> None:
    full_name = "Full"
    (tmp_path / "full.pth").touch()
    (tmp_path / "optical.pth").touch()
    (tmp_path / "full.json").write_text("{}")
    (tmp_path / "optical.json").write_text("{}")
    (tmp_path / "bridge").mkdir()
    raw = {
        "primary_metric": "directional_win_rate",
        "compare_interaction": False,
        "pairing_mode": "single_parameter",
        "single_parameter_profile": "source_physical_v1",
        "models": [
            {
                "name": full_name,
                "type": "multimodal",
                "scoring": "logits",
                "checkpoint": "full.pth",
                "config": "full.json",
            },
            {
                "name": BRIDGE_NAME,
                "type": "physics_ejecta_bridge",
                "artifact_path": "bridge",
            },
            {
                "name": GW_BLIND_NAME,
                "type": "optical",
                "checkpoint": "optical.pth",
                "config": "optical.json",
            },
        ],
        "new_model_name": full_name,
        "baseline_model_name": full_name,
        "test_data_path": "test.h5",
        "output_dir": "out",
        "pairwise_comparisons": [
            {
                "model": full_name,
                "baseline_model": BRIDGE_NAME,
                "family": "full_minus_bridge_directional_win_rate",
            }
        ],
        "plot_model_order": [full_name, BRIDGE_NAME, GW_BLIND_NAME],
        "plot_pairwise_families": ["full_minus_bridge_directional_win_rate"],
        "robustness_model_order": [full_name, BRIDGE_NAME, GW_BLIND_NAME],
    }

    cfg, specs, bridge = normalise_v2_config(raw, tmp_path / "config.json")

    assert cfg["new_model_name"] == cfg["baseline_model_name"] == full_name
    assert {spec["name"] for spec in specs} == {full_name, GW_BLIND_NAME}
    assert bridge["name"] == BRIDGE_NAME


def test_plot_labels_use_physical_symbols_and_readable_dose_names() -> None:
    assert set(PLOT_PARAMETER_LABELS) == {
        "chirp_mass_detector",
        "mass_ratio",
        "chi_eff",
        "primary_spin_z",
        "abs_costheta",
        "log10_distance_gpc",
    }
    assert all(label.startswith("$") for label in PLOT_PARAMETER_LABELS.values())
    assert "NSBH" in PLOT_PARAMETER_LABELS["primary_spin_z"]
    assert r"\iota" in PLOT_PARAMETER_LABELS["abs_costheta"]
    assert r"\theta" not in PLOT_PARAMETER_LABELS["abs_costheta"]
    assert PLOT_MODEL_LABELS == {
        "Full": "MAGIKS",
        BRIDGE_NAME: "Physics-informed empirical bridge",
        GW_BLIND_NAME: "GW-independent optical control",
    }
    assert PLOT_MODEL_MARKERS == {
        "Full": "o",
        BRIDGE_NAME: "s",
        GW_BLIND_NAME: "D",
    }
    assert DOSE_BIN_LABELS == {
        "T1_low": "Low",
        "T2_mid": "Medium",
        "T3_high": "High",
    }
    assert all("_" not in label for label in DOSE_BIN_LABELS.values())
    assert r"\usepackage{txfonts}" in PLOT_LATEX_PREAMBLE
    assert r"\setmainfont{Times New Roman}" in PLOT_LATEX_PREAMBLE
    assert PLOT_FONT_SCALE == 2.0
    assert PLOT_AXIS_SCALE == 1.5
    assert PLOT_LEGEND_SCALE == 1.5


def test_deterministic_split_keeps_event_identity() -> None:
    source = ["bns", "bns", "nsbh"]
    simulation = [1, 1, 1]
    first = deterministic_event_split(source, simulation, seed=42)
    second = deterministic_event_split(source, simulation, seed=42)
    np.testing.assert_array_equal(first, second)
    assert first[0] == first[1]


def test_source_gw_features_preserve_primary_spin() -> None:
    scalars = np.asarray(
        [
            [4.0, 1.0, 0.7, -0.2, 0.2, 0.5, 0.1],
            [1.0, 4.0, -0.2, 0.7, 0.2, 0.5, 0.1],
        ]
    )
    nsbh = derive_gw_feature_matrix(scalars, "nsbh")
    np.testing.assert_allclose(nsbh[:, 2], [0.7, 0.7])
    bns = derive_gw_feature_matrix(scalars, "bns")
    np.testing.assert_allclose(bns[:, 2], [0.52, 0.52])


def test_luptitude_inverse_matches_known_flux() -> None:
    b = np.asarray([10.0] * 6)
    flux = np.asarray([[[0.0, 5.0, -3.0, 20.0, 2.0, 1.0]]])
    factor = 2.5 / np.log(10.0)
    values = 31.4 - factor * (np.arcsinh(flux / (2.0 * b)) + np.log(b))
    errors = np.full_like(values, 0.1)
    recovered, sigma = luptitude_to_flux_sigma(
        values, errors, psfflux_zp=31.4, lupt_b_njy=b
    )
    np.testing.assert_allclose(recovered, flux, atol=1e-10)
    assert np.all(sigma > 0)


def test_lightcurve_features_respect_band_masks() -> None:
    times = np.asarray([[-0.05, 0.01, 0.08, 0.15]])
    values = np.full((1, 4, 6), 25.0)
    errors = np.full_like(values, 0.1)
    masks = np.zeros_like(values)
    masks[0, :, 1] = 1
    features, available, cadence = extract_lightcurve_features(
        times,
        values,
        masks,
        errors,
        psfflux_zp=31.4,
        lupt_b_njy=np.asarray([200.0, 72.0, 95.0, 182.0, 347.0, 1049.0]),
    )
    assert features.shape == available.shape == (1, len(LIGHTCURVE_FEATURE_NAMES))
    assert available[0, 6:12].any()
    assert not available[0, :6].any()
    assert cadence[0, 1] == 1 / 6


def test_curve_selection_is_capped_and_reproducible() -> None:
    parent = np.asarray([0, 0, 0, 1, 1, 2])
    first = select_curves_per_event(
        parent, np.asarray([0, 1]), seed=9, max_curves_per_event=2
    )
    second = select_curves_per_event(
        parent, np.asarray([0, 1]), seed=9, max_curves_per_event=2
    )
    np.testing.assert_array_equal(first, second)
    assert len(first) == 4
    assert np.sum(parent[first] == 0) == 2


def test_fast_curve_selection_is_capped_reproducible_and_handles_unsorted() -> None:
    parent = np.asarray([2, 0, 1, 0, 2, 1, 0, 2, 1])
    first = _select_curves_fast(
        parent, np.asarray([0, 2]), seed=42, max_curves_per_event=2
    )
    second = _select_curves_fast(
        parent, np.asarray([0, 2]), seed=42, max_curves_per_event=2
    )
    np.testing.assert_array_equal(first, second)
    assert len(first) == 4
    assert set(parent[first]) == {0, 2}
    assert all(np.count_nonzero(parent[first] == event) == 2 for event in (0, 2))


def test_contiguous_block_reader_matches_hdf5_point_selection(tmp_path: Path) -> None:
    source = np.arange(360, dtype=np.float32).reshape(120, 3)
    selected = np.asarray([0, 7, 8, 31, 32, 71, 119])
    path = tmp_path / "selected_rows.h5"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("values", data=source, chunks=(8, 3))
        actual = _read_selected_rows(dataset, selected, block_rows=16)
    np.testing.assert_array_equal(actual, source[selected])


def test_batched_neighbor_search_matches_exact_scalar_distance() -> None:
    rng = np.random.default_rng(12)
    references = rng.normal(size=(24, 5))
    parents = np.arange(24)
    psi = rng.normal(size=(24, 4))
    queries = rng.normal(size=(3, 5))
    samples, distances = _prepare_neighbor_samples(
        queries,
        references,
        parents,
        psi,
        max_k=4,
        device="cpu",
        batch_size=2,
    )
    for query, actual_samples, actual_distances in zip(queries, samples, distances):
        expected = select_unique_neighbors(
            np.linalg.norm(references - query[None, :], axis=1),
            parents,
            k=4,
            alpha=1.0,
        )
        np.testing.assert_array_equal(actual_samples, psi[expected.indices])
        np.testing.assert_allclose(actual_distances, expected.distances, atol=2e-6)


def test_masked_batched_neighbor_search_matches_scalar_cadence_filter() -> None:
    rng = np.random.default_rng(4)
    references = rng.normal(size=(20, 4))
    parents = np.arange(20)
    psi = rng.normal(size=(20, 4))
    reference_mask = rng.random((20, 4)) > 0.15
    reference_cadence = rng.normal(scale=0.1, size=(20, 3))
    query = rng.normal(size=(1, 4))
    query_mask = np.asarray([[True, False, True, True]])
    query_cadence = np.zeros((1, 3))
    samples, distances = _prepare_neighbor_samples(
        query,
        references,
        parents,
        psi,
        max_k=2,
        query_mask=query_mask,
        reference_mask=reference_mask,
        query_cadence=query_cadence,
        reference_cadence=reference_cadence,
        device="cpu",
        batch_size=1,
    )
    scalar_distances = masked_feature_distances(
        query[0], query_mask[0], references, reference_mask
    )
    eligible = cadence_candidate_mask(
        query_cadence[0], reference_cadence, max_abs=1.0, l1_max=3.0
    )
    expected = select_unique_neighbors(
        scalar_distances, parents, k=2, alpha=1.0, eligible=eligible
    )
    np.testing.assert_array_equal(samples[0], psi[expected.indices])
    np.testing.assert_allclose(distances[0], expected.distances, atol=2e-6)


def test_masked_distance_uses_only_common_dimensions() -> None:
    query = np.asarray([1.0, 20.0, 3.0])
    qmask = np.asarray([True, False, True])
    refs = np.asarray([[2.0, -100.0, 5.0], [1.0, 20.0, 3.0]])
    rmask = np.asarray([[True, True, True], [True, False, False]])
    distance = masked_feature_distances(query, qmask, refs, rmask)
    np.testing.assert_allclose(distance, [np.sqrt(2.5), 0.0])


def test_cadence_caliper_and_unique_neighbors() -> None:
    references = np.asarray([[0.1, 0.1, 0.1], [2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(
        cadence_candidate_mask(np.zeros(3), references, max_abs=1.0, l1_max=3.0),
        [True, False],
    )
    posterior = select_unique_neighbors(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([10, 10, 11, 12]),
        k=3,
        alpha=1.0,
    )
    np.testing.assert_array_equal(posterior.indices, [0, 2, 3])
    assert np.isclose(posterior.weights.sum(), 1.0)


def test_weighted_gaussian_and_bhattacharyya() -> None:
    samples = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    mean, covariance = weighted_gaussian(
        samples, np.asarray([0.5, 0.5]), covariance_floor=0.1
    )
    np.testing.assert_allclose(mean, [1.0, 0.0])
    assert np.all(np.linalg.eigvalsh(covariance) > 0)
    same = bhattacharyya_score(mean, covariance, mean, covariance)
    shifted = bhattacharyya_score(mean, covariance, mean + 1.0, covariance)
    assert same == 0.0
    assert shifted < same
    almost_psd = np.diag([1.0, -5e-10])
    assert bhattacharyya_score(np.zeros(2), almost_psd, np.zeros(2), almost_psd) == 0.0


def _pair(target: str = "chi_eff") -> dict:
    return {
        "condition_id": f"{target}__bns__caliper_0p5",
        "target_parameter": target,
        "caliper_iqr": 0.5,
        "is_primary_caliper": True,
        "pair_id": "pair0",
        "source_type": "bns",
        "gw_a": 1,
        "gw_b": 2,
        "target_delta_iqr": 1.2,
        "other_parameter_l1_distance": 0.1,
        "other_parameter_max_abs_difference": 0.1,
        "nuisance_l1_distance": 0.2,
        "nuisance_max_abs_difference": 0.2,
    }


def test_directional_decision_and_control_tie() -> None:
    pair = _pair()
    covariance = np.eye(4) * 0.1
    posterior_a = np.zeros(4)
    posterior_b = np.asarray([2.0, -1.0, 0.5, 0.7])
    bridge_scores = (
        bhattacharyya_score(posterior_a, covariance, posterior_a, covariance),
        bhattacharyya_score(posterior_a, covariance, posterior_b, covariance),
        bhattacharyya_score(posterior_b, covariance, posterior_a, covariance),
        bhattacharyya_score(posterior_b, covariance, posterior_b, covariance),
    )
    aligned = decision_row(
        model=BRIDGE_NAME,
        model_type="physics_ejecta_bridge",
        pair=pair,
        curve_pair_id=0,
        anchor="a",
        scores=bridge_scores,
    )
    assert aligned["directional_win_rate"] == 1.0
    control = decision_row(
        model=GW_BLIND_NAME,
        model_type="optical",
        pair=pair,
        curve_pair_id=0,
        anchor="a",
        scores=(2.0, 1.0, 2.0, 1.0),
    )
    assert control["directional_win_rate"] == 0.5


def test_pair_aggregation_averages_anchors_then_curves() -> None:
    pair = _pair()
    decisions = []
    for curve_id, dwr_scores in enumerate(
        [((2.0, 1.0, 1.0, 2.0),), ((1.0, 2.0, 2.0, 1.0),)]
    ):
        for anchor in ("a", "b"):
            decisions.append(
                decision_row(
                    model=BRIDGE_NAME,
                    model_type="physics_ejecta_bridge",
                    pair=pair,
                    curve_pair_id=curve_id,
                    anchor=anchor,
                    scores=dwr_scores[0],
                )
            )
    rows = aggregate_pair_dwr(decisions, [pair])
    assert len(rows) == 1
    assert rows[0]["directional_win_rate"] == 0.5
    assert rows[0]["n_curve_pairs"] == 2


def test_summary_has_four_six_endpoint_holm_families() -> None:
    targets = [
        ("chirp_mass_detector", ["bns", "nsbh"]),
        ("mass_ratio", ["bns", "nsbh"]),
        ("chi_eff", ["bns"]),
        ("primary_spin_z", ["nsbh"]),
        ("abs_costheta", ["bns", "nsbh"]),
        ("log10_distance_gpc", ["bns", "nsbh"]),
    ]
    models = ["Mixed Gallery v1", "Default MAGIKS", BRIDGE_NAME, GW_BLIND_NAME]
    rows = []
    counter = 0
    for target, sources in targets:
        for source in sources:
            for index in range(6):
                pair_id = f"p{counter}"
                counter += 1
                for model_index, model in enumerate(models):
                    rows.append(
                        {
                            **_pair(target),
                            "condition_id": f"{target}__{source}__caliper_0p5",
                            "pair_id": pair_id,
                            "source_type": source,
                            "model": model,
                            "model_type": "x",
                            "directional_win_rate": (
                                0.5
                                if model == GW_BLIND_NAME
                                else 0.55 + 0.01 * model_index
                            ),
                        }
                    )
    summary = summarize_dwr(
        rows,
        new_model="Mixed Gallery v1",
        default_model="Default MAGIKS",
        primary_caliper=0.5,
        bootstrap_samples=100,
        permutation_samples=100,
        seed=42,
    )
    frame = np.asarray([row["family"] for row in summary], dtype=object)
    families = sorted(set(frame) - {""})
    assert len(families) == 4
    assert all(np.count_nonzero(frame == family) == 6 for family in families)
    assert all("interaction" not in row for row in summary)
    blind = [
        row
        for row in summary
        if row["endpoint"] == "absolute_directional_win_rate"
        and row["model"] == GW_BLIND_NAME
    ]
    assert blind and all(row["permutation_p"] == "" for row in blind)
    assert all(row["estimate"] == 0.5 for row in blind)


def test_artifact_runtime_never_requires_test_truth(tmp_path: Path) -> None:
    manifest = {
        "status": "complete",
        "artifact_version": "physics_ejecta_bridge_v1",
        "test_truth_fields_used_for_scoring": [],
        "sources": {
            source: {
                "gw_hyperparameters": {"k": 2, "alpha": 1.0, "covariance_floor": 0.1},
                "lc_hyperparameters": {"k": 2, "alpha": 1.0, "covariance_floor": 0.1},
            }
            for source in ("bns", "nsbh")
        },
    }
    (tmp_path / "bridge_fit_manifest.json").write_text(json.dumps(manifest))
    arrays = {}
    for source in ("bns", "nsbh"):
        prefix = f"{source}__"
        arrays[prefix + "gw_center"] = np.zeros(3)
        arrays[prefix + "gw_scale"] = np.ones(3)
        arrays[prefix + "gw_features"] = np.asarray([[1.0, 0.25, 0.0], [1.1, 0.3, 0.1]])
        arrays[prefix + "gw_parent_ids"] = np.asarray([1, 2])
        arrays[prefix + "gw_psi"] = np.zeros((2, 4))
        arrays[prefix + "psi_center"] = np.zeros(4)
        arrays[prefix + "psi_scale"] = np.ones(4)
        lc_features = np.arange(60, dtype=float).reshape(20, 3) / 20.0
        arrays[prefix + "lc_center"] = np.zeros(3)
        arrays[prefix + "lc_scale"] = np.ones(3)
        arrays[prefix + "cadence_center"] = np.zeros(3)
        arrays[prefix + "cadence_scale"] = np.ones(3)
        arrays[prefix + "lc_features"] = lc_features
        arrays[prefix + "lc_feature_mask"] = np.ones((20, 3), dtype=np.uint8)
        arrays[prefix + "lc_cadence"] = np.zeros((20, 3))
        arrays[prefix + "lc_parent_ids"] = np.arange(20)
        arrays[prefix + "lc_psi"] = np.column_stack(
            [lc_features, np.arange(20, dtype=float) / 20.0]
        )
    np.savez_compressed(tmp_path / "bridge_reference.npz", **arrays)
    bridge = PhysicsEjectaBridge(tmp_path)
    assert bridge.manifest["test_truth_fields_used_for_scoring"] == []
    queries = arrays["bns__lc_features"][:2]
    masks = np.ones_like(queries, dtype=bool)
    cadences = np.zeros((2, 3))
    batch_mean, batch_covariance, batch_neighbors = bridge.lc_posteriors_batch(
        queries, masks, cadences, "bns", device="cpu", batch_size=2
    )
    for index in range(2):
        scalar_mean, scalar_covariance, scalar_neighbors = bridge.lc_posterior(
            queries[index], masks[index], cadences[index], "bns"
        )
        np.testing.assert_allclose(batch_mean[index], scalar_mean, atol=1e-6)
        np.testing.assert_allclose(
            batch_covariance[index], scalar_covariance, atol=1e-6
        )
        np.testing.assert_array_equal(
            batch_neighbors[index].indices, scalar_neighbors.indices
        )
    bridge.arrays.close()


def test_optimized_fit_smoke_uses_complete_calibration_split(tmp_path: Path) -> None:
    rng = np.random.default_rng(81)
    n_per_source = 100
    n_events = 2 * n_per_source
    curves_per_event = 2
    n_curves = n_events * curves_per_event
    source = np.asarray([b"bns"] * n_per_source + [b"nsbh"] * n_per_source)
    simulation_id = np.arange(n_events, dtype=np.int64)
    scalars = np.zeros((n_events, 7), dtype=np.float32)
    scalars[:n_per_source, 0] = rng.uniform(1.3, 1.8, n_per_source)
    scalars[:n_per_source, 1] = rng.uniform(1.0, 1.3, n_per_source)
    scalars[n_per_source:, 0] = rng.uniform(5.0, 9.0, n_per_source)
    scalars[n_per_source:, 1] = rng.uniform(1.0, 1.8, n_per_source)
    scalars[:, 2:4] = rng.uniform(-0.5, 0.5, (n_events, 2))
    scalars[:, 4] = rng.uniform(-1.0, 1.0, n_events)
    scalars[:, 5] = rng.uniform(0.05, 0.8, n_events)
    scalars[:, 6] = scalars[:, 5] * 0.1
    train_path = tmp_path / "train.h5"
    times = np.tile(np.linspace(-0.08, 0.18, 8, dtype=np.float32), (n_curves, 1))
    values = rng.normal(25.0, 0.4, (n_curves, 8, 6)).astype(np.float32)
    masks = np.ones_like(values, dtype=np.float32)
    errors = np.full_like(values, 0.1)
    with h5py.File(train_path, "w") as handle:
        handle.attrs["psfflux_zp"] = 31.4
        handle.attrs["lupt_b_njy"] = np.asarray(
            [200.0, 72.0, 95.0, 182.0, 347.0, 1049.0]
        )
        gw = handle.create_group("events/gw_data")
        gw.create_dataset("scalars", data=scalars)
        gw.create_dataset("source_type", data=source)
        gw.create_dataset("simulation_id", data=simulation_id)
        gw.create_dataset(
            "event_uid",
            data=np.asarray([f"event-{i}".encode() for i in range(n_events)]),
        )
        gw.create_dataset("has_kn", data=np.ones(n_events, dtype=np.int8))
        gw.create_dataset("mej_dynamic", data=rng.uniform(1e-4, 5e-2, n_events))
        gw.create_dataset("mej_wind", data=rng.uniform(1e-3, 1e-1, n_events))
        optical = handle.create_group("events/optical_data")
        optical.create_dataset(
            "parent_gw_idx", data=np.repeat(np.arange(n_events), curves_per_event)
        )
        optical.create_dataset("times", data=times, chunks=(16, 8))
        optical.create_dataset("values", data=values, chunks=(16, 8, 6))
        optical.create_dataset("masks", data=masks, chunks=(16, 8, 6))
        optical.create_dataset("errors", data=errors, chunks=(16, 8, 6))
    artifact = tmp_path / "artifact"
    config_path = tmp_path / "fit.json"
    raw = {
        "train_data_path": str(train_path),
        "artifact_dir": str(artifact),
        "comparison_window": [-0.1, 0.2],
        "fit_fraction": 0.8,
        "max_curves_per_event": 2,
        "k_candidates": [2],
        "alpha_candidates": [1.0],
        "covariance_floor_candidates": [0.1],
        "calibration_device": "cpu",
        "calibration_batch_size": 8,
        "hdf5_read_block_rows": 32,
        "num_threads": 2,
        "seed": 42,
    }
    config_path.write_text(json.dumps(raw))
    fit_bridge(normalise_fit_config(raw, config_path), config_path)
    manifest = json.loads((artifact / "bridge_fit_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["fit_runtime"]["combined_source_optical_read"] is True
    assert manifest["fit_runtime"]["neighbor_distance_precision"] == "float64"
    for source_name in ("bns", "nsbh"):
        source_manifest = manifest["sources"][source_name]
        assert (
            source_manifest["n_fit_events"]
            + source_manifest["n_calibration_events_used"]
            == source_manifest["n_events"]
        )
