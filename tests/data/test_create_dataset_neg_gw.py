import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.data import create_dataset_bns_nsbh as builder


def _frame(event_ids, sample_class, dynamic, wind):
    count = len(event_ids)
    return pd.DataFrame(
        {
            "simulation_id": event_ids,
            "event_uid": [
                f"bns_train_{sample_class}_{event_id}" for event_id in event_ids
            ],
            "sample_class": sample_class,
            "mass1_detector": np.full(count, 1.5),
            "mass2_detector": np.full(count, 1.3),
            "spin1z": np.zeros(count),
            "spin2z": np.zeros(count),
            "recovered_mass1_detector": np.full(count, 1.6),
            "recovered_mass2_detector": np.full(count, 1.2),
            "recovered_spin1z": np.full(count, 0.1),
            "recovered_spin2z": np.full(count, -0.1),
            "inclination": np.zeros(count),
            "distmean": np.full(count, 100.0),
            "diststd": np.full(count, 10.0),
            "mjd": np.arange(60001.0, 60001.0 + count),
            "mej_dynamic": dynamic,
            "mej_wind": wind,
        }
    )


def test_prepare_source_uses_split_catalogs_and_collision_safe_identity(
    monkeypatch, tmp_path
):
    positive = _frame(
        [2, 3, 4, 5, 6],
        "pos",
        [0.0, 0.2, 0.2, 0.1, 0.1],
        [0.3, 0.0, 0.3, 0.1, 0.1],
    )
    # Reuse simulation_id=2 deliberately: event_uid must keep this distinct from pos 2.
    negative = _frame([2], "neg", [0.0], [0.0])
    success_path = tmp_path / "success.txt"
    # Positive event 6 lacks successful coverage and must not become type-2.
    success_path.write_text("2\n3\n4\n5\n", encoding="utf-8")
    seen_positive = []

    monkeypatch.setattr(
        builder,
        "_load_gw_catalog",
        lambda path, **kwargs: (positive if path == "pos.csv" else negative).copy(),
    )
    monkeypatch.setattr(builder, "_normalize_gw_catalog_columns", lambda data, _: data)
    monkeypatch.setattr(
        builder,
        "_scan_source_pos_neg",
        lambda event_ids, **kwargs: (
            seen_positive.extend(event_ids.tolist()) or np.asarray([2, 3]),
            np.asarray([4, 5]),
            0,
        ),
    )
    monkeypatch.setattr(
        builder,
        "_scan_ids_with_skymap_only",
        lambda event_ids, **kwargs: (np.asarray(event_ids), 0),
    )
    config = builder.SourceConfig(
        tag="bns",
        full_catalog_path="pos.csv",
        negative_catalog_path="neg.csv",
        skymap_dir=str(tmp_path / "pos"),
        negative_skymap_dir=str(tmp_path / "neg"),
        sim_root=str(tmp_path),
        sim_name="TEST",
        success_ids_path=str(success_path),
    )

    prepared = builder._prepare_source(config, "train", np.random.default_rng(42))

    assert seen_positive == [2, 3, 4, 5]
    assert prepared.pos_event_ids.tolist() == [2, 3]
    assert prepared.neg_event_ids.tolist() == [-3, 4, 5]
    assert prepared.neg_type_by_event == {-3: 1, 4: 2, 5: 2}
    assert prepared.simulation_id_by_event[-3] == 2
    assert prepared.event_uid_by_event[-3] == "bns_train_neg_2"
    assert prepared.event_uid_by_event[2] == "bns_train_pos_2"
    assert prepared.n_filtered_non_success_mej_pos == 1
    np.testing.assert_allclose(
        prepared.gw_params[prepared.event_to_row[2], :4],
        [1.6, 1.2, 0.1, -0.1],
    )


def test_scan_source_pos_neg_uses_verified_aggregate_status(tmp_path):
    skymap_dir = tmp_path / "skymaps"
    skymap_dir.mkdir()
    for event_id in (2, 3):
        (skymap_dir / f"{event_id}.fits").touch()
    artifact = tmp_path / "simulation_intermediates.h5"
    with h5py.File(artifact, "w") as handle:
        handle.attrs["schema_version"] = "kn-simulation-intermediates-v2"
        handle.attrs["artifact_kind"] = "aggregate"
        events = handle.create_group("events")
        events.create_dataset("simulation_id", data=[2, 3])
        events.create_dataset("optical_realization_count", data=[1, 0])
        events.create_dataset(
            "status", data=["success", "success"], dtype=h5py.string_dtype()
        )

    positive, negative, missing = builder._scan_source_pos_neg(
        np.asarray([2, 3]),
        str(skymap_dir),
        None,
        "TEST",
        "scan",
        sim_artifact=str(artifact),
    )

    assert positive.tolist() == [2]
    assert negative.tolist() == [3]
    assert missing == 0

    with h5py.File(artifact, "r+") as handle:
        handle["events/status"][0] = "failed"
    with pytest.raises(ValueError, match="listed as successful"):
        builder._scan_source_pos_neg(
            np.asarray([2]),
            str(skymap_dir),
            None,
            "TEST",
            "scan",
            sim_artifact=str(artifact),
        )
