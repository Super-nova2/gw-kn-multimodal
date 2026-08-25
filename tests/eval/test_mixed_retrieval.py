from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from mixed_retrieval import (
    allocate_mixed_negative_counts,
    build_mixed_galleries,
    mixed_gallery_identity_digest,
    mixed_score_outcome,
    select_source_balanced_queries,
)
from scripts.eval import eval_mixed_retrieval_comparison as mixed_eval


def _sequences(query_ids: list[int], n_trials: int, n_candidates: int):
    result = {}
    for trial in range(n_trials):
        for gw_id in query_ids:
            result[(trial, gw_id)] = {
                "candidate_indices": np.arange(n_candidates, dtype=np.int64),
                "synthetic_coordinates": np.column_stack(
                    [np.arange(n_candidates), np.arange(n_candidates)]
                ).astype(np.float32),
                "abs_dt_days": np.arange(n_candidates, dtype=np.float32),
            }
    return result


def test_allocate_mixed_counts_uses_one_to_three_ratio() -> None:
    assert allocate_mixed_negative_counts(16, 0.25) == (4, 11)
    assert allocate_mixed_negative_counts(32, 0.25) == (8, 23)
    assert allocate_mixed_negative_counts(1000, 0.25) == (250, 749)
    with pytest.raises(ValueError):
        allocate_mixed_negative_counts(2, 0.25)


def test_kn_random_is_cross_type_and_prefix_nested() -> None:
    parent = np.repeat(np.arange(6, dtype=np.int64), 2)
    source = ["bns", "bns", "nsbh", "nsbh", "nsbh", "nsbh"]
    positive_map = {gw_id: np.flatnonzero(parent == gw_id) for gw_id in range(6)}
    galleries = build_mixed_galleries(
        gw_positive_indices=positive_map,
        query_gw_ids=[0],
        optical_parent_gw_idx=parent,
        nonkn_candidate_sequences=_sequences([0], 2, 16),
        gallery_sizes=[5, 9],
        n_trials=2,
        seed=42,
        kn_fraction=0.25,
        kn_candidate_mode="kn_random",
        gw_source_types=source,
    )
    for trial in range(2):
        small = galleries[(5, trial, 0)]
        large = galleries[(9, trial, 0)]
        assert np.array_equal(
            small["kn_negative_indices"],
            large["kn_negative_indices"][: small["n_kn_negative"]],
        )
        assert np.array_equal(
            small["nonkn_negative_indices"],
            large["nonkn_negative_indices"][: small["n_nonkn_negative"]],
        )
        assert 0 not in large["kn_negative_parent_gw_ids"]
        assert len(set(large["kn_negative_parent_gw_ids"].tolist())) == 2
        assert any(
            source[parent_id] == "nsbh"
            for parent_id in large["kn_negative_parent_gw_ids"]
        )
    repeated = build_mixed_galleries(
        gw_positive_indices=positive_map,
        query_gw_ids=[0],
        optical_parent_gw_idx=parent,
        nonkn_candidate_sequences=_sequences([0], 2, 16),
        gallery_sizes=[5, 9],
        n_trials=2,
        seed=42,
        kn_fraction=0.25,
        kn_candidate_mode="kn_random",
    )
    assert mixed_gallery_identity_digest(galleries) == mixed_gallery_identity_digest(
        repeated
    )


def test_nuisance_matching_stays_same_source() -> None:
    parent = np.repeat(np.arange(8, dtype=np.int64), 2)
    source = ["bns"] * 4 + ["nsbh"] * 4
    features = np.column_stack(
        [np.arange(parent.size, dtype=float), np.ones((parent.size, 4))]
    )
    galleries = build_mixed_galleries(
        gw_positive_indices={
            gw_id: np.flatnonzero(parent == gw_id) for gw_id in range(8)
        },
        query_gw_ids=[0],
        optical_parent_gw_idx=parent,
        nonkn_candidate_sequences=_sequences([0], 1, 16),
        gallery_sizes=[9],
        n_trials=1,
        seed=3,
        kn_fraction=0.25,
        kn_candidate_mode="kn_nuisance_matched",
        gw_source_types=source,
        nuisance_features=features,
    )
    parents = galleries[(9, 0, 0)]["kn_negative_parent_gw_ids"]
    assert all(source[parent_id] == "bns" for parent_id in parents)


def test_mixed_score_outcome_has_three_tie_aware_scopes() -> None:
    outcome = mixed_score_outcome(1.0, [1.0, 2.0], [0.0])
    assert outcome["all"]["recall_at_1"] == 0.0
    assert outcome["kn_only"]["recall_at_1"] == 0.0
    assert outcome["nonkn_only"]["recall_at_1"] == 1.0
    tied = mixed_score_outcome(1.0, [1.0], [])
    assert tied["kn_only"]["recall_at_1"] == pytest.approx(0.5)
    assert tied["kn_only"]["mrr"] == pytest.approx(0.75)


def test_source_balanced_query_selection() -> None:
    selected = select_source_balanced_queries(
        range(10), ["bns"] * 5 + ["nsbh"] * 5, queries_per_source=3, seed=7
    )
    assert len(selected) == 6
    assert sum(value < 5 for value in selected) == 3
    assert sum(value >= 5 for value in selected) == 3


def test_legacy_plotter_is_called_for_each_condition_and_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = []
    for condition in ("training_aligned", "positive_shared"):
        for scope in ("all", "kn_only", "nonkn_only"):
            for model in ("Physical Pairing v1", "Default MAGIKS"):
                for size in (10, 100):
                    rows.append(
                        {
                            "model": model,
                            "condition": condition,
                            "scope": scope,
                            "gallery_size": size,
                            "source": "all",
                            "recall_at_1": 0.2,
                            "recall_at_5": 0.4,
                            "recall_at_10": 0.5,
                            "mrr": 0.3,
                            "random_recall_at_1": 1.0 / size,
                            "random_recall_at_5": min(5, size) / size,
                            "random_recall_at_10": min(10, size) / size,
                            "random_mrr": 0.1,
                        }
                    )
    calls = []

    def fake_plotter(curve_rows, output_dir):
        calls.append(list(curve_rows))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "retrieval_curves.png").touch()
        (output_dir / "retrieval_curves.pdf").touch()

    monkeypatch.setattr(mixed_eval, "plot_retrieval_curves", fake_plotter)
    paths = mixed_eval._plot_with_legacy_retrieval_plotter(pd.DataFrame(rows), tmp_path)

    assert len(calls) == 6
    assert len(paths) == 12
    for curve_rows in calls:
        random_rows = [row for row in curve_rows if row["method"] == "Random ranking"]
        assert len(random_rows) == 8


def test_condition_feature_semantics() -> None:
    positive_coordinate = np.asarray([1.0, 2.0], dtype=np.float32)
    positive_dt = np.asarray([3.0], dtype=np.float32)
    kn_dt = np.asarray([4.0, 5.0], dtype=np.float32)
    synthetic = np.asarray([[6.0, 7.0], [8.0, 9.0]], dtype=np.float32)
    nonkn_dt = np.asarray([10.0, 11.0], dtype=np.float32)

    aligned = mixed_eval._condition_candidate_features(
        "training_aligned",
        positive_coordinate=positive_coordinate,
        positive_dt_days=positive_dt,
        kn_dt_days=kn_dt,
        nonkn_synthetic_coordinates=synthetic,
        nonkn_training_aligned_dt_days=nonkn_dt,
        n_kn=2,
        n_nonkn=2,
    )
    assert aligned["positive_coordinates"] is None
    assert np.array_equal(
        aligned["kn_coordinates"], np.asarray([[1.0, 2.0], [1.0, 2.0]])
    )
    assert np.array_equal(aligned["nonkn_coordinates"], synthetic)
    assert np.array_equal(aligned["kn_dt_days"], kn_dt)
    assert np.array_equal(aligned["nonkn_dt_days"], nonkn_dt)

    shared = mixed_eval._condition_candidate_features(
        "positive_shared",
        positive_coordinate=positive_coordinate,
        positive_dt_days=positive_dt,
        kn_dt_days=kn_dt,
        nonkn_synthetic_coordinates=synthetic,
        nonkn_training_aligned_dt_days=nonkn_dt,
        n_kn=2,
        n_nonkn=2,
    )
    assert np.array_equal(shared["positive_coordinates"], [[1.0, 2.0]])
    assert np.array_equal(shared["nonkn_coordinates"], [[1.0, 2.0], [1.0, 2.0]])
    assert np.array_equal(shared["kn_dt_days"], [3.0, 3.0])
    assert np.array_equal(shared["nonkn_dt_days"], [3.0, 3.0])


def test_source_balanced_query_selection_rejects_insufficient_source() -> None:
    with pytest.raises(ValueError, match="need 3"):
        select_source_balanced_queries(
            range(5),
            ["bns", "bns", "bns", "nsbh", "nsbh"],
            queries_per_source=3,
            seed=7,
        )


def test_redshift_query_validation_checks_coverage_bins_and_counts() -> None:
    sources = ["bns", "nsbh"] * 4
    redshifts = [0.02, 0.02, 0.05, 0.05, 0.08, 0.08, 0.12, 0.12]
    metadata = {
        index: {"redshift": redshift} for index, redshift in enumerate(redshifts)
    }
    query_rows, counts = mixed_eval._validate_redshift_query_metadata(
        range(8),
        sources,
        metadata,
        bin_edges=[0.0, 0.04, 0.065, 0.10, 0.14],
        bin_labels=["z0", "z1", "z2", "z3"],
        min_pooled=2,
        min_per_source=1,
    )
    assert len(query_rows) == 8
    assert [row["n_queries"] for row in counts if row["source"] == "pooled"] == [
        2,
        2,
        2,
        2,
    ]

    missing = dict(metadata)
    missing.pop(7)
    with pytest.raises(ValueError, match="Missing redshift"):
        mixed_eval._validate_redshift_query_metadata(
            range(8),
            sources,
            missing,
            bin_edges=[0.0, 0.04, 0.065, 0.10, 0.14],
            bin_labels=["z0", "z1", "z2", "z3"],
            min_pooled=2,
            min_per_source=1,
        )

    outside = dict(metadata)
    outside[7] = {"redshift": 0.2}
    with pytest.raises(ValueError, match="outside configured bins"):
        mixed_eval._validate_redshift_query_metadata(
            range(8),
            sources,
            outside,
            bin_edges=[0.0, 0.04, 0.065, 0.10, 0.14],
            bin_labels=["z0", "z1", "z2", "z3"],
            min_pooled=2,
            min_per_source=1,
        )


def _paired_outcome_rows() -> pd.DataFrame:
    rows = []
    gallery_deltas = {
        10: 0.9,
        100: 0.01,
        500: 0.02,
        1000: 0.03,
        2000: 0.04,
        5000: 0.05,
    }
    for model in ("Mixed Gallery v1", "Default MAGIKS"):
        for gw_id, source in enumerate(("bns", "bns", "nsbh", "nsbh")):
            for size, delta in gallery_deltas.items():
                for trial in (0, 1):
                    baseline = 0.10 + 0.02 * trial + 0.01 * gw_id
                    value = baseline + (delta if model == "Mixed Gallery v1" else 0.0)
                    rows.append(
                        {
                            "model": model,
                            "condition": "training_aligned",
                            "scope": "all",
                            "gallery_size": size,
                            "trial": trial,
                            "gw_id": gw_id,
                            "source": source,
                            "recall_at_1": value,
                            "recall_at_5": value,
                            "recall_at_10": value,
                            "mrr": value,
                        }
                    )
    return pd.DataFrame(rows)


def test_trials_are_collapsed_before_paired_bootstrap() -> None:
    collapsed = mixed_eval.analysis.collapse_trials(_paired_outcome_rows())
    assert collapsed["n_trials"].eq(2).all()
    row = collapsed[
        collapsed["model"].eq("Default MAGIKS")
        & collapsed["gw_id"].eq(0)
        & collapsed["gallery_size"].eq(100)
    ].iloc[0]
    assert row["mrr"] == pytest.approx(0.11)

    first = mixed_eval.analysis.paired_bootstrap(
        collapsed,
        new_name="Mixed Gallery v1",
        baseline_name="Default MAGIKS",
        n_bootstrap=100,
        seed=59,
    )
    second = mixed_eval.analysis.paired_bootstrap(
        collapsed,
        new_name="Mixed Gallery v1",
        baseline_name="Default MAGIKS",
        n_bootstrap=100,
        seed=59,
    )
    pd.testing.assert_frame_equal(first, second)
    assert {"pooled", "source_macro", "bns", "nsbh"}.issubset(
        set(first["source_aggregation"])
    )
    assert first["bootstrap_unit"].eq("gw_id_after_trial_mean").all()
    assert first[first["source_aggregation"].eq("pooled")]["n_unique_gw"].eq(4).all()


def test_primary_training_score_matches_training_formula_and_excludes_g10() -> None:
    collapsed = mixed_eval.analysis.collapse_trials(_paired_outcome_rows())
    weights = {100: 0.10, 500: 0.15, 1000: 0.20, 2000: 0.25, 5000: 0.30}
    summary = mixed_eval.analysis.training_effect_summary(
        collapsed,
        new_name="Mixed Gallery v1",
        baseline_name="Default MAGIKS",
        condition="training_aligned",
        scope="all",
        primary_metric={
            "name": "training_selection_score",
            "source_aggregation": "source_macro",
            "metric_weights": {"mrr": 0.8, "recall_at_1": 0.2},
            "gallery_size_weights": weights,
        },
        n_bootstrap=100,
        seed=59,
    )
    expected = sum(
        weights[size] * delta
        for size, delta in {
            100: 0.01,
            500: 0.02,
            1000: 0.03,
            2000: 0.04,
            5000: 0.05,
        }.items()
    )
    assert summary.iloc[0]["mean_delta"] == pytest.approx(expected)
    assert summary.iloc[0]["conclusion"] == "supported"


def test_interval_plots_export_nonempty_png_and_pdf(tmp_path: Path) -> None:
    interval_rows = []
    for model_index, model in enumerate(
        ("Mixed Gallery v1", "Default MAGIKS", "Optical-only")
    ):
        for size in (10, 100):
            for metric, _ in mixed_eval.analysis.PLOT_METRICS:
                estimate = 0.5 - 0.1 * model_index
                interval_rows.append(
                    {
                        "model": model,
                        "condition": "training_aligned",
                        "scope": "all",
                        "gallery_size": size,
                        "metric": metric,
                        "source_aggregation": "pooled",
                        "estimate": estimate,
                        "ci_low": estimate - 0.02,
                        "ci_high": estimate + 0.02,
                        "n_unique_gw": 4,
                    }
                )
    paths = mixed_eval.analysis.plot_retrieval_curves(
        pd.DataFrame(interval_rows),
        output_dir=tmp_path,
        model_order=["Mixed Gallery v1", "Default MAGIKS", "Optical-only"],
        kn_fraction=0.25,
    )
    assert {
        "plots/training_aligned/all/retrieval_curves.png",
        "plots/training_aligned/all/retrieval_curves.pdf",
    } == set(paths)
    assert all((tmp_path / path).stat().st_size > 0 for path in paths)


def test_postprocess_only_fails_cleanly_without_outcomes(tmp_path: Path) -> None:
    config = {
        "evaluation_mode": "mixed_kn_nonkn",
        "experiment_id": "toy",
        "models": [],
        "test_data_path": str(tmp_path / "test.h5"),
        "neg_data_path": str(tmp_path / "neg.h5"),
        "output_dir": str(tmp_path / "missing-output"),
        "gallery_sizes": [10, 100],
        "conditions": ["training_aligned"],
        "primary_condition": "training_aligned",
        "primary_scope": "all",
        "primary_metric": {
            "name": "training_selection_score",
            "source_aggregation": "source_macro",
            "metric_weights": {"mrr": 0.8, "recall_at_1": 0.2},
            "gallery_size_weights": {"100": 1.0},
        },
        "redshift_analysis": {"enabled": False},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="requires saved outcomes"):
        mixed_eval.run_postprocess_only(config_path)


def test_final_v2_config_uses_all_balanced_training_aligned_protocol() -> None:
    config_path = (
        MODEL_DIR / "args" / "eval" / "retrieval_comparison_mixed_kn_nonkn.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["experiment_id"] == "mixed_kn_nonkn_train_aligned_v2"
    assert config["queries_per_source"] == 887
    assert config["n_neg_samples"] == 500000
    assert config["kn_candidate_mode"] == "kn_random"
    assert config["primary_condition"] == "training_aligned"
    assert config["primary_scope"] == "all"
    assert config["bootstrap_samples"] == 5000
    assert config["bootstrap_seed"] == 59
    assert len(config["models"]) == 3
    assert config["redshift_analysis"]["bin_edges"] == [
        0.0,
        0.04,
        0.065,
        0.10,
        0.14,
    ]


def test_postprocess_writes_statistics_and_redshift_artifacts(tmp_path: Path) -> None:
    outcomes = _paired_outcome_rows()
    outcomes["redshift"] = outcomes["gw_id"].map({0: 0.02, 1: 0.02, 2: 0.08, 3: 0.08})
    outcomes["redshift_bin_index"] = outcomes["gw_id"].map({0: 0, 1: 0, 2: 1, 3: 1})
    outcomes["redshift_bin_label"] = outcomes["redshift_bin_index"].map(
        {0: "low", 1: "high"}
    )
    cfg = {
        "models": [
            {"name": "Mixed Gallery v1"},
            {"name": "Default MAGIKS"},
        ],
        "kn_fraction": 0.25,
        "bootstrap_new_model": "Mixed Gallery v1",
        "bootstrap_baseline_model": "Default MAGIKS",
        "bootstrap_samples": 20,
        "bootstrap_seed": 59,
        "conditions": ["training_aligned"],
        "primary_condition": "training_aligned",
        "primary_scope": "all",
        "primary_metric": {
            "name": "training_selection_score",
            "source_aggregation": "source_macro",
            "metric_weights": {"mrr": 0.8, "recall_at_1": 0.2},
            "gallery_size_weights": {
                100: 0.10,
                500: 0.15,
                1000: 0.20,
                2000: 0.25,
                5000: 0.30,
            },
        },
        "redshift_analysis": {
            "enabled": True,
            "primary_gallery_size": 1000,
            "bin_labels": ["low", "high"],
            "aggregations": ["pooled", "source_macro"],
        },
    }
    processed = mixed_eval.analysis.postprocess(outcomes, cfg, tmp_path)
    assert processed["training_effect"].iloc[0]["conclusion"] == "supported"
    for filename in (
        "gw_metric_intervals.csv",
        "paired_bootstrap.csv",
        "training_effect_summary.csv",
        "training_effect_summary.json",
        "redshift_metric_intervals.csv",
    ):
        assert (tmp_path / filename).stat().st_size > 0
    assert any(
        path.endswith("redshift_performance.png")
        for path in processed["artifacts"]["redshift_plots"]
    )
