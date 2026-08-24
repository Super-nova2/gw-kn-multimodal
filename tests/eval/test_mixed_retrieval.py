from __future__ import annotations

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
