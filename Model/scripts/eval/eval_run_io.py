#!/usr/bin/env python3
"""Atomic evaluation outcomes and immutable run-manifest helpers."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


RETRIEVAL_FIELDS = (
    "seed",
    "trial",
    "gw_id",
    "source_type",
    "redshift_bin",
    "redshift",
    "gallery_size",
    "actual_gallery_size",
    "model",
    "rank",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "coverage",
)
CLASSIFICATION_FIELDS = (
    "seed",
    "sample_index",
    "gw_id",
    "source_type",
    "pair_type",
    "label",
    "probability",
    "model",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_tree_digest(root: Path) -> str:
    """Hash Python sources so resumes and aggregation reject code drift."""
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write a CSV through a sibling temporary file and publish it atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=tuple(fieldnames), extrasaction="raise"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def prepare_output_directory(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    resume: bool = False,
) -> Dict[str, Any]:
    """Create an isolated output directory or validate an explicit resume."""
    output_dir = Path(output_dir)
    payload = dict(manifest)
    payload["manifest_digest"] = stable_digest(payload)
    manifest_path = output_dir / "run_manifest.json"
    success_path = output_dir / "_SUCCESS.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use a new directory "
                "or set resume=true after verifying the run."
            )
        if success_path.exists():
            raise FileExistsError(f"Run is already complete and immutable: {output_dir}")
        if not manifest_path.is_file():
            raise ValueError(f"Cannot resume without {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("manifest_digest") != payload["manifest_digest"]:
            raise ValueError(
                "Resume manifest mismatch; refusing to mix incompatible results."
            )
        return payload
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(manifest_path, payload)
    return payload


def mark_run_success(
    output_dir: Path,
    manifest: Mapping[str, Any],
    artifacts: Sequence[str],
) -> None:
    output_dir = Path(output_dir)
    missing = [name for name in artifacts if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot mark run complete; missing artifacts: {missing}"
        )
    write_json_atomic(
        output_dir / "_SUCCESS.json",
        {
            "manifest_digest": manifest["manifest_digest"],
            "artifacts": list(artifacts),
            "status": "complete",
        },
    )


class AtomicGzipCsvWriter:
    """Stream a gzip CSV and expose it atomically only after ``commit``."""

    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite outcome file: {self.path}")
        self.fieldnames = tuple(fieldnames)
        self.tmp_path = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp"
        )
        self._handle = gzip.open(
            self.tmp_path, "wt", encoding="utf-8", newline=""
        )
        self._writer = csv.DictWriter(
            self._handle, fieldnames=self.fieldnames, extrasaction="raise"
        )
        self._writer.writeheader()
        self._committed = False

    def writerows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self._writer.writerows(rows)

    def commit(self) -> None:
        if self._committed:
            return
        self._handle.flush()
        self._handle.close()
        os.replace(self.tmp_path, self.path)
        self._committed = True

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        if not self._committed:
            try:
                self.tmp_path.unlink()
            except FileNotFoundError:
                pass


def _redshift_values(
    metadata: Optional[Mapping[int, Mapping[str, Any]]], gw_id: int
) -> tuple[str, Any]:
    if not metadata or int(gw_id) not in metadata:
        return "", ""
    row = metadata[int(gw_id)]
    return (
        str(row.get("redshift_bin_label", row.get("redshift_bin", ""))),
        row.get("redshift", ""),
    )


def retrieval_outcome_rows(
    *,
    seed: int,
    model: str,
    outcomes: Mapping[tuple[int, int, int], Mapping[str, Any]],
    gw_source_map: Optional[Mapping[int, str]] = None,
    redshift_metadata: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Iterable[Dict[str, Any]]:
    for (gallery_size, trial, gw_id), outcome in sorted(outcomes.items()):
        contributions = outcome.get("metric_contributions") or {}
        rank = float(outcome.get("rank", float("nan")))
        redshift_bin, redshift = _redshift_values(redshift_metadata, int(gw_id))
        actual_size = int(outcome.get("actual_gallery_size", gallery_size))
        requested_size = max(
            1, int(outcome.get("requested_gallery_size", gallery_size))
        )

        def metric(name: str, fallback: float) -> float:
            return float(contributions.get(name, fallback))

        yield {
            "seed": int(seed),
            "trial": int(trial),
            "gw_id": int(gw_id),
            "source_type": str(
                (gw_source_map or {}).get(int(gw_id), "unknown")
            ),
            "redshift_bin": redshift_bin,
            "redshift": redshift,
            "gallery_size": int(gallery_size),
            "actual_gallery_size": actual_size,
            "model": str(model),
            "rank": rank,
            "recall_at_1": metric("recall_at_1", float(rank < 1)),
            "recall_at_5": metric("recall_at_5", float(rank < 5)),
            "recall_at_10": metric("recall_at_10", float(rank < 10)),
            "mrr": metric("mrr", 1.0 / (rank + 1.0)),
            "coverage": min(
                1.0, max(0.0, actual_size / requested_size)
            ),
        }


def classification_prediction_rows(
    *,
    seed: int,
    model: str,
    triplet_logits: Mapping[str, Any],
) -> Iterable[Dict[str, Any]]:
    import torch

    specs = (
        ("positive", "logits_positive", "gw_id_positive", "source_positive", 1),
        (
            "optical_negative",
            "logits_optical_neg",
            "gw_id_optical_neg",
            "source_optical_neg",
            0,
        ),
        (
            "gw_negative",
            "logits_gw_neg",
            "gw_id_gw_neg",
            "source_gw_neg",
            0,
        ),
        (
            "mismatched_negative",
            "logits_hard_neg",
            "gw_id_hard_neg",
            "source_hard_neg",
            0,
        ),
    )
    for pair_type, logits_key, ids_key, sources_key, label in specs:
        logits = triplet_logits.get(logits_key)
        if logits is None:
            continue
        probs = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
        gw_ids = list(triplet_logits.get(ids_key, []))
        sources = list(triplet_logits.get(sources_key, []))
        if len(gw_ids) != len(probs):
            raise ValueError(
                f"{ids_key} has {len(gw_ids)} entries for {len(probs)} predictions"
            )
        if sources and len(sources) != len(probs):
            raise ValueError(f"{sources_key} length does not match {logits_key}")
        for sample_index, probability in enumerate(probs.tolist()):
            yield {
                "seed": int(seed),
                "sample_index": int(sample_index),
                "gw_id": int(gw_ids[sample_index]),
                "source_type": (
                    str(sources[sample_index]) if sources else "unknown"
                ),
                "pair_type": pair_type,
                "label": int(label),
                "probability": float(probability),
                "model": str(model),
            }
