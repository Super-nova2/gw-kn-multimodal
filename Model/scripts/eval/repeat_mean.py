#!/usr/bin/env python3
"""Prepare, aggregate, and plot exactly three fixed-checkpoint evaluation seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.eval.eval_run_io import (
        stable_digest,
        write_csv_atomic,
        write_json_atomic,
    )
except ModuleNotFoundError:  # Direct execution from Model/scripts/eval.
    from eval_run_io import stable_digest, write_csv_atomic, write_json_atomic


EVAL_SEEDS = (42, 123, 456)
RETRIEVAL_METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "coverage",
)
CLASSIFICATION_METRICS = (
    "auroc",
    "auprc",
    "accuracy",
    "precision",
    "recall",
    "f1",
)
PER_SEED_FIELDS = (
    "task",
    "scope",
    "seed",
    "model",
    "gallery_size",
    "metric",
    "value",
    "n_gw",
    "n_rows",
)
MEAN_FIELDS = (
    "task",
    "scope",
    "model",
    "gallery_size",
    "metric",
    "mean",
    "n_seeds",
)
DELTA_FIELDS = (
    "task",
    "scope",
    "reference_model",
    "comparison_model",
    "gallery_size",
    "metric",
    "mean_delta",
    "n_seeds",
)
ALLOWED_CONFIG_KEYS = {
    "eval_seeds",
    "expected_gallery_trials",
    "reference_model",
    "input_root",
    "output_dir",
}

PAPER_GALLERY_SIZES = (100, 500, 1000, 5000)
PAPER_RETRIEVAL_METRICS = (
    ("recall_at_1", "R@1"),
    ("recall_at_10", "R@10"),
    ("mrr", "MRR"),
)
PAPER_RETRIEVAL_MODELS = (
    "Skymap-only",
    "Optical-only baseline",
    "Fink Random Forest",
    "w/o Retrieval Loss",
    "w/o Cross-Attn",
    "w/o Fusion",
    "w/o Contrastive Loss",
    "Full",
)
PAPER_GW170817_MODELS = (
    "Skymap-only",
    "Optical-only baseline",
    "Fink Random Forest",
    "w/o Retrieval Loss",
    "Full",
)
PAPER_METHOD_LABELS = {
    "Skymap-only": "Skymap-only",
    "Optical-only baseline": "Optical-only",
    "Fink Random Forest": "Fink Random Forest",
    "w/o Retrieval Loss": "w/o Retrieval Loss",
    "w/o Cross-Attn": "w/o Cross Attention",
    "w/o Fusion": "w/o Fusion Branch",
    "w/o Contrastive Loss": "w/o Contrastive Loss",
    "Full": "MAGIKS",
}
PAPER_TABLE_LABELS = {
    "Skymap-only": "Skymap-only",
    "Optical-only baseline": "Optical-only",
    "Fink Random Forest": "Fink Random Forest",
    "w/o Retrieval Loss": "w/o Retrieval loss",
    "w/o Cross-Attn": "w/o Cross-attention",
    "w/o Fusion": "w/o Fusion",
    "w/o Contrastive Loss": "w/o Contrastive loss",
    "Full": "MAGIKS",
}
PAPER_PLOT_ORDER = (
    "Fink Random Forest",
    "Optical-only baseline",
    "Skymap-only",
    "w/o Contrastive Loss",
    "w/o Cross-Attn",
    "w/o Fusion",
    "w/o Retrieval Loss",
    "Full",
)
PAPER_METHOD_STYLES = {
    "Fink Random Forest": ("#2CA02C", "o", "-"),
    "Optical-only baseline": ("#1F77B4", "o", "-"),
    "Skymap-only": ("#F2C230", "o", "-"),
    "w/o Contrastive Loss": ("#9467BD", "o", "-"),
    "w/o Cross-Attn": ("#8C564B", "o", "-"),
    "w/o Fusion": ("#E377C2", "o", "-"),
    "w/o Retrieval Loss": ("#7F7F7F", "o", "-"),
    "Full": ("#D62728", "o", "-"),
}
PAPER_PAIR_STYLES = {
    "positive": ("#2ecc71", "Positives"),
    "optical_negative": ("#3498db", "Optical Negatives"),
    "gw_negative": ("#f39c12", "GW Negatives"),
    "mismatched_negative": ("#e74c3c", "Mismatched Negatives"),
}
TABLE4_CAPTION = (
    "Summary of retrieval ablation experiments at representative gallery sizes"
)
TABLE4_COMMENT = (
    "Each gallery contains one true kilonova counterpart and time- and "
    "sky-compatible distractors. R@1 and R@10 denote the fraction of queries "
    "for which the true counterpart is ranked first or within the top ten "
    "candidates, respectively, and MRR is the mean reciprocal rank of the true "
    "counterpart. The rows compare skymap-only and optical-only baselines with "
    "the reproduced Fink random forest baseline and ablated variants of MAGIKS. "
    "For the Fink baseline, metrics are analytical expectations under uniform "
    "random ordering within exact-score ties. Best values are shown in bold, "
    "and second-best values are underlined."
)
TABLE6_CAPTION = "GW170817-like retrieval results at representative gallery sizes"
TABLE6_COMMENT = (
    "Each gallery contains one simulated GW170817-like kilonova counterpart "
    "and time- and sky-compatible distractor transients. Recall@K measures the "
    "fraction of queries for which the true counterpart is ranked within the "
    "top K candidates, and MRR measures the average inverse rank of the true "
    "counterpart. The rows compare the skymap-only baseline, optical-only "
    "baseline, Fink random forest baseline, the variant without retrieval "
    "loss, and the full MAGIKS model. For the Fink baseline, metrics are "
    "analytical expectations under uniform random ordering within exact-score "
    "ties. Best values are shown in bold, and second-best values are underlined."
)


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite generated config: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
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


def prepare_experiment(config_path: Path, *, dry_run: bool) -> list[Path]:
    """Generate three seed configs, two smoke configs, and two mean configs."""
    raw = _load_json(config_path)
    base = config_path.parent
    seeds = tuple(int(value) for value in raw.get("eval_seeds", ()))
    if seeds != EVAL_SEEDS:
        raise ValueError(
            f"eval_seeds must be exactly {list(EVAL_SEEDS)} in this order"
        )
    if int(raw.get("gallery_trials", -1)) != 10:
        raise ValueError("gallery_trials must be exactly 10")
    output_root = _resolve(base, raw["output_root"])
    generated_root = _resolve(base, raw["generated_config_root"])
    eval_specs = {
        "retrieval": _resolve(base, raw["retrieval_base_config"]),
        "gw170817": _resolve(base, raw["gw170817_base_config"]),
    }
    for source in eval_specs.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    planned: list[tuple[Path, Dict[str, Any]]] = []
    for task, source_path in eval_specs.items():
        source = _load_json(source_path)
        for seed in seeds:
            payload = dict(source)
            payload.update(
                {
                    "seed": seed,
                    "gallery_trials": 10,
                    "save_outcomes": True,
                    "strict_output_safety": True,
                    "resume": False,
                    "experiment_id": str(raw["experiment_id"]),
                    "output_dir": str(output_root / task / f"seed_{seed}"),
                }
            )
            planned.append(
                (generated_root / task / f"seed_{seed}.json", payload)
            )

        smoke = dict(source)
        smoke.update(
            {
                "seed": seeds[0],
                "gallery_trials": int(raw.get("smoke_gallery_trials", 2)),
                "gallery_sizes": str(raw.get("smoke_gallery_sizes", "10")),
                "save_outcomes": True,
                "strict_output_safety": True,
                "resume": False,
                "experiment_id": f"{raw['experiment_id']}_smoke",
                "output_dir": str(
                    output_root / "smoke" / task / f"seed_{seeds[0]}"
                ),
                "n_neg_samples": int(raw.get("smoke_n_neg_samples", 2000)),
            }
        )
        if task == "retrieval":
            smoke["max_gw_events"] = int(raw.get("smoke_max_gw_events", 4))
            smoke["test_steps"] = int(raw.get("smoke_test_steps", 1))
        else:
            smoke["max_kn_per_redshift_bin"] = int(
                raw.get("smoke_max_kn_per_redshift_bin", 1)
            )
        planned.append((generated_root / "smoke" / f"{task}.json", smoke))
        planned.append(
            (
                generated_root / task / "mean.json",
                {
                    "eval_seeds": list(seeds),
                    "expected_gallery_trials": 10,
                    "reference_model": str(raw.get("reference_model", "Full")),
                    "input_root": str(output_root / task),
                    "output_dir": str(output_root / task / "mean3_summary"),
                },
            )
        )

    generated = [path for path, _ in planned]
    conflicts = [path for path in generated if path.exists()]
    if conflicts and not dry_run:
        raise FileExistsError(
            "Refusing to overwrite generated configs: "
            + ", ".join(str(path) for path in conflicts)
        )
    if not dry_run:
        for path, payload in planned:
            _write_json_new(path, payload)
    print(f"{'Would generate' if dry_run else 'Generated'} {len(generated)} configs")
    for path in generated:
        print(path)
    return generated


def _validate_config(config: Mapping[str, Any]) -> tuple[int, ...]:
    unknown = set(config).difference(ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unsupported seed-mean config fields: {sorted(unknown)}")
    seeds = tuple(int(value) for value in config.get("eval_seeds", ()))
    if seeds != EVAL_SEEDS:
        raise ValueError(
            f"eval_seeds must be exactly {list(EVAL_SEEDS)} in this order"
        )
    if int(config.get("expected_gallery_trials", 10)) != 10:
        raise ValueError("expected_gallery_trials must be exactly 10")
    return seeds


def _validated_manifest(
    run_dir: Path, seed: int
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    success_path = run_dir / "_SUCCESS.json"
    manifest_path = run_dir / "run_manifest.json"
    if not success_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete or missing seed run: {run_dir}")
    success = _load_json(success_path)
    manifest = _load_json(manifest_path)
    if success.get("status") != "complete":
        raise ValueError(f"Run is not marked complete: {run_dir}")
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError(f"Manifest seed mismatch: {run_dir}")
    digest = manifest.get("manifest_digest")
    unsigned = dict(manifest)
    unsigned.pop("manifest_digest", None)
    if not digest or stable_digest(unsigned) != digest:
        raise ValueError(f"Invalid manifest digest: {run_dir}")
    if success.get("manifest_digest") != digest:
        raise ValueError(f"Success/manifest digest mismatch: {run_dir}")
    artifacts = success.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"Success marker has no artifact list: {run_dir}")
    for name in artifacts:
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe artifact path in {success_path}: {name}")
        if not (run_dir / relative).is_file():
            raise FileNotFoundError(
                f"Declared artifact is missing: {run_dir / relative}"
            )
    return manifest, success


def _config_signature(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    config_path = Path(str(manifest.get("input_config", ""))).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Manifest input_config is missing: {config_path}")
    config = _load_json(config_path)
    return {
        "models": config.get("models"),
        "test_data_path": str(config.get("test_data_path", "")),
        "neg_data_path": str(config.get("neg_data_path", "")),
        "gallery_trials": int(config.get("gallery_trials", -1)),
        "gallery_sizes": str(config.get("gallery_sizes", "")),
    }


def _read_runs(
    input_root: Path, seeds: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    retrieval_parts: list[pd.DataFrame] = []
    classification_parts: list[pd.DataFrame] = []
    compatibility: list[Dict[str, Any]] = []
    has_classification: list[bool] = []
    for seed in seeds:
        run_dir = input_root / f"seed_{seed}"
        manifest, success = _validated_manifest(run_dir, seed)
        artifacts = {str(name) for name in success["artifacts"]}

        retrieval_path = run_dir / "retrieval_outcomes.csv.gz"
        if "retrieval_outcomes.csv.gz" not in artifacts:
            raise FileNotFoundError(
                f"Successful run does not declare retrieval outcomes: {run_dir}"
            )
        retrieval = pd.read_csv(retrieval_path)
        if retrieval.empty:
            raise ValueError(f"Empty retrieval outcomes: {retrieval_path}")
        retrieval_parts.append(retrieval)

        classification_path = run_dir / "classification_predictions.csv.gz"
        declared_classification = "classification_predictions.csv.gz" in artifacts
        if classification_path.exists() != declared_classification:
            raise ValueError(
                f"Classification artifact declaration mismatch: {run_dir}"
            )
        has_classification.append(declared_classification)
        if declared_classification:
            classification = pd.read_csv(classification_path)
            if classification.empty:
                raise ValueError(
                    f"Empty classification predictions: {classification_path}"
                )
            classification_parts.append(classification)

        compatibility.append(
            {
                "experiment_digest": manifest.get("experiment_digest"),
                "code_digest": manifest.get("code_digest"),
                "test_data_name": Path(
                    str(manifest.get("test_data_path", ""))
                ).name,
                "neg_data_name": Path(
                    str(manifest.get("neg_data_path", ""))
                ).name,
                "config": _config_signature(manifest),
            }
        )

    if len({stable_digest(item) for item in compatibility}) != 1:
        raise ValueError(
            "Seed manifests, checkpoints, or data paths are not compatible"
        )
    first = compatibility[0]
    if not first["experiment_digest"] or not first["code_digest"]:
        raise ValueError("Seed runs require nonempty experiment and code digests")
    if len(set(has_classification)) != 1:
        raise ValueError(
            "Classification predictions must exist for all three seeds or none"
        )
    retrieval = pd.concat(retrieval_parts, ignore_index=True)
    classification = (
        pd.concat(classification_parts, ignore_index=True)
        if classification_parts
        else pd.DataFrame()
    )
    return retrieval, classification, first


def _require_columns(
    frame: pd.DataFrame, required: Iterable[str], label: str
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _validate_retrieval(
    frame: pd.DataFrame, seeds: Sequence[int], trials: int
) -> pd.DataFrame:
    identity = ["seed", "trial", "gw_id", "gallery_size"]
    invariant = ["source_type", "redshift_bin", "actual_gallery_size"]
    required = identity + invariant + ["model", *RETRIEVAL_METRICS]
    _require_columns(frame, required, "retrieval outcomes")
    work = frame.loc[:, required].copy()
    for column in (
        "seed",
        "trial",
        "gw_id",
        "gallery_size",
        "actual_gallery_size",
    ):
        work[column] = pd.to_numeric(work[column], errors="raise").astype(
            np.int64
        )
    for metric in RETRIEVAL_METRICS:
        work[metric] = pd.to_numeric(work[metric], errors="raise").astype(
            np.float64
        )
        if not np.isfinite(work[metric].to_numpy()).all():
            raise ValueError(f"Non-finite retrieval metric: {metric}")
    if set(int(value) for value in work["seed"].unique()) != set(seeds):
        raise ValueError(
            "Retrieval rows do not contain exactly the configured seeds"
        )
    if work.duplicated(identity + ["model"]).any():
        raise ValueError("Duplicate retrieval gallery/model identities")
    for column in invariant:
        counts = work.groupby(identity, dropna=False)[column].nunique(
            dropna=False
        )
        if (counts > 1).any():
            raise ValueError(f"Methods do not share the same gallery {column}")
    paired = work.pivot(
        index=identity, columns="model", values="recall_at_1"
    )
    if paired.isna().any().any():
        raise ValueError("Unpaired retrieval rows across models")
    expected = set(range(trials))
    for (seed, model, gallery_size), block in work.groupby(
        ["seed", "model", "gallery_size"], sort=False
    ):
        observed = set(int(value) for value in block["trial"].unique())
        if observed != expected:
            raise ValueError(
                f"Expected trials 0..{trials - 1} for seed={seed}, "
                f"model={model}, gallery={gallery_size}; got {sorted(observed)}"
            )
    return work


def _scope_blocks(
    frame: pd.DataFrame, *, include_redshift: bool
) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "overall", frame
    for source, block in frame.groupby(
        "source_type", dropna=False, sort=True
    ):
        label = str(source).strip()
        if label and label.lower() != "nan":
            yield f"source:{label}", block
    if include_redshift:
        for redshift, block in frame.groupby(
            "redshift_bin", dropna=False, sort=True
        ):
            label = str(redshift).strip()
            if label and label.lower() != "nan":
                yield f"redshift:{label}", block


def summarize_retrieval_by_seed(
    frame: pd.DataFrame, seeds: Sequence[int]
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for seed in seeds:
        seed_frame = frame[frame["seed"] == seed]
        for scope, scoped in _scope_blocks(
            seed_frame, include_redshift=True
        ):
            for (model, gallery_size), block in scoped.groupby(
                ["model", "gallery_size"], sort=True
            ):
                for metric in RETRIEVAL_METRICS:
                    rows.append(
                        {
                            "task": "retrieval",
                            "scope": scope,
                            "seed": int(seed),
                            "model": str(model),
                            "gallery_size": int(gallery_size),
                            "metric": metric,
                            "value": float(block[metric].mean()),
                            "n_gw": int(block["gw_id"].nunique()),
                            "n_rows": int(len(block)),
                        }
                    )
    return rows


def _classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.unique(labels).size != 2:
        raise ValueError(
            "Classification metrics require both labels in every reported scope"
        )
    predicted = probabilities >= 0.5
    tp = float(np.sum(predicted & (labels == 1)))
    fp = float(np.sum(predicted & (labels == 0)))
    tn = float(np.sum(~predicted & (labels == 0)))
    fn = float(np.sum(~predicted & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": (tp + tn) / (tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def _validate_classification(
    frame: pd.DataFrame, seeds: Sequence[int]
) -> pd.DataFrame:
    identity = [
        "seed",
        "pair_type",
        "sample_index",
        "gw_id",
        "source_type",
        "label",
    ]
    required = identity + ["probability", "model"]
    _require_columns(frame, required, "classification predictions")
    work = frame.loc[:, required].copy()
    for column in ("seed", "sample_index", "gw_id", "label"):
        work[column] = pd.to_numeric(work[column], errors="raise").astype(
            np.int64
        )
    work["probability"] = pd.to_numeric(
        work["probability"], errors="raise"
    ).astype(np.float64)
    probabilities = work["probability"].to_numpy()
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError(
            "Classification probabilities must be finite and within [0, 1]"
        )
    if set(int(value) for value in work["seed"].unique()) != set(seeds):
        raise ValueError(
            "Classification rows do not contain exactly the configured seeds"
        )
    if work.duplicated(identity + ["model"]).any():
        raise ValueError("Duplicate classification prediction identities")
    paired = work.pivot(
        index=identity, columns="model", values="probability"
    )
    if paired.isna().any().any():
        raise ValueError("Unpaired classification rows across models")
    return work


def summarize_classification_by_seed(
    frame: pd.DataFrame, seeds: Sequence[int]
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for seed in seeds:
        seed_frame = frame[frame["seed"] == seed]
        for scope, scoped in _scope_blocks(
            seed_frame, include_redshift=False
        ):
            for model, block in scoped.groupby("model", sort=True):
                metrics = _classification_metrics(
                    block["label"].to_numpy(),
                    block["probability"].to_numpy(),
                )
                for metric in CLASSIFICATION_METRICS:
                    rows.append(
                        {
                            "task": "classification",
                            "scope": scope,
                            "seed": int(seed),
                            "model": str(model),
                            "gallery_size": "",
                            "metric": metric,
                            "value": float(metrics[metric]),
                            "n_gw": int(block["gw_id"].nunique()),
                            "n_rows": int(len(block)),
                        }
                    )
    return rows


def equal_seed_means(
    per_seed_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    reference_model: str,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    frame = pd.DataFrame(per_seed_rows)
    if frame.empty:
        raise ValueError("No per-seed metrics were produced")
    mean_keys = ["task", "scope", "model", "gallery_size", "metric"]
    summaries: list[Dict[str, Any]] = []
    for identity, block in frame.groupby(
        mean_keys, dropna=False, sort=True
    ):
        observed = tuple(sorted(int(value) for value in block["seed"]))
        if observed != tuple(sorted(seeds)) or len(block) != len(seeds):
            raise ValueError(
                f"Metric is not present exactly once for all seeds: {identity}"
            )
        task, scope, model, gallery_size, metric = identity
        summaries.append(
            {
                "task": str(task),
                "scope": str(scope),
                "model": str(model),
                "gallery_size": (
                    ""
                    if pd.isna(gallery_size) or gallery_size == ""
                    else int(gallery_size)
                ),
                "metric": str(metric),
                "mean": float(block["value"].sum() / len(seeds)),
                "n_seeds": len(seeds),
            }
        )

    delta_rows: list[Dict[str, Any]] = []
    metric_keys = ["task", "scope", "gallery_size", "metric"]
    for metric_identity, metric_block in frame.groupby(
        metric_keys, dropna=False, sort=True
    ):
        pivot = metric_block.pivot(
            index="seed", columns="model", values="value"
        )
        if reference_model not in pivot.columns:
            raise ValueError(f"Reference model is missing: {reference_model}")
        if pivot.isna().any().any() or set(pivot.index) != set(seeds):
            raise ValueError(
                f"Models are not paired within every seed: {metric_identity}"
            )
        task, scope, gallery_size, metric = metric_identity
        for comparison_model in sorted(
            str(value)
            for value in pivot.columns
            if value != reference_model
        ):
            deltas = pivot[reference_model] - pivot[comparison_model]
            delta_rows.append(
                {
                    "task": str(task),
                    "scope": str(scope),
                    "reference_model": reference_model,
                    "comparison_model": comparison_model,
                    "gallery_size": (
                        ""
                        if pd.isna(gallery_size) or gallery_size == ""
                        else int(gallery_size)
                    ),
                    "metric": str(metric),
                    "mean_delta": float(deltas.sum() / len(seeds)),
                    "n_seeds": len(seeds),
                }
            )
    return summaries, delta_rows


def _plot_diagnostics(
    per_seed_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    pdf_output_path: Path,
    png_output_path: Path,
    seeds: Sequence[int],
) -> None:
    from io import BytesIO

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image

    per_seed = pd.DataFrame(per_seed_rows)
    means = pd.DataFrame(summary_rows)
    pdf_fd, pdf_tmp_name = tempfile.mkstemp(
        prefix=f".{pdf_output_path.name}.",
        suffix=".tmp.pdf",
        dir=pdf_output_path.parent,
    )
    png_fd, png_tmp_name = tempfile.mkstemp(
        prefix=f".{png_output_path.name}.",
        suffix=".tmp.png",
        dir=png_output_path.parent,
    )
    os.close(pdf_fd)
    os.close(png_fd)
    png_pages: list[Image.Image] = []

    def save_page(pdf: PdfPages, figure: Any) -> None:
        pdf.savefig(figure)
        with BytesIO() as buffer:
            figure.savefig(
                buffer,
                format="png",
                dpi=150,
                facecolor="white",
            )
            buffer.seek(0)
            with Image.open(buffer) as image:
                png_pages.append(image.convert("RGB").copy())

    try:
        with PdfPages(pdf_tmp_name) as pdf:
            retrieval = per_seed[
                (per_seed["task"] == "retrieval")
                & (per_seed["scope"] == "overall")
            ]
            retrieval_means = means[
                (means["task"] == "retrieval")
                & (means["scope"] == "overall")
            ]
            for metric in RETRIEVAL_METRICS:
                block = retrieval[retrieval["metric"] == metric]
                mean_block = retrieval_means[
                    retrieval_means["metric"] == metric
                ]
                models = sorted(block["model"].unique().tolist())
                if not models:
                    continue
                ncols = min(4, len(models))
                nrows = int(np.ceil(len(models) / ncols))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(4.2 * ncols, 3.2 * nrows),
                    squeeze=False,
                )
                for axis, model in zip(axes.flat, models):
                    model_block = block[block["model"] == model]
                    for seed in seeds:
                        seed_block = model_block[
                            model_block["seed"] == seed
                        ].sort_values("gallery_size")
                        axis.plot(
                            seed_block["gallery_size"],
                            seed_block["value"],
                            marker="o",
                            alpha=0.55,
                            label=str(seed),
                        )
                    model_mean = mean_block[
                        mean_block["model"] == model
                    ].sort_values("gallery_size")
                    axis.plot(
                        model_mean["gallery_size"],
                        model_mean["mean"],
                        color="black",
                        marker="s",
                        linewidth=2.2,
                        label="mean",
                    )
                    axis.set_xscale("log")
                    axis.set_title(model)
                    axis.set_xlabel("Gallery size")
                    axis.set_ylabel(metric.replace("_", " "))
                    axis.grid(alpha=0.2)
                for axis in axes.flat[len(models) :]:
                    axis.set_visible(False)
                handles, labels = axes.flat[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.955),
                    ncol=4,
                )
                fig.suptitle(
                    f"Three fixed evaluation seeds and equal mean: {metric}",
                    y=0.995,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.88))
                save_page(pdf, fig)
                plt.close(fig)

            classification = per_seed[
                (per_seed["task"] == "classification")
                & (per_seed["scope"] == "overall")
            ]
            classification_means = means[
                (means["task"] == "classification")
                & (means["scope"] == "overall")
            ]
            models = sorted(classification["model"].unique().tolist())
            if models:
                ncols = min(4, len(models))
                nrows = int(np.ceil(len(models) / ncols))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(4.2 * ncols, 3.2 * nrows),
                    squeeze=False,
                )
                x_values = np.arange(len(CLASSIFICATION_METRICS))
                for axis, model in zip(axes.flat, models):
                    model_block = classification[
                        classification["model"] == model
                    ]
                    for seed in seeds:
                        seed_block = model_block[
                            model_block["seed"] == seed
                        ].set_index("metric")
                        axis.plot(
                            x_values,
                            [
                                seed_block.loc[metric, "value"]
                                for metric in CLASSIFICATION_METRICS
                            ],
                            marker="o",
                            alpha=0.55,
                            label=str(seed),
                        )
                    model_mean = classification_means[
                        classification_means["model"] == model
                    ].set_index("metric")
                    axis.plot(
                        x_values,
                        [
                            model_mean.loc[metric, "mean"]
                            for metric in CLASSIFICATION_METRICS
                        ],
                        color="black",
                        marker="s",
                        linewidth=2.2,
                        label="mean",
                    )
                    axis.set_xticks(
                        x_values,
                        CLASSIFICATION_METRICS,
                        rotation=35,
                        ha="right",
                    )
                    axis.set_ylim(-0.02, 1.02)
                    axis.set_title(model)
                    axis.grid(alpha=0.2)
                for axis in axes.flat[len(models) :]:
                    axis.set_visible(False)
                handles, labels = axes.flat[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.955),
                    ncol=4,
                )
                fig.suptitle(
                    "Classification at threshold 0.5: three seeds and equal mean",
                    y=0.995,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.88))
                save_page(pdf, fig)
                plt.close(fig)
        if not png_pages:
            raise ValueError("No diagnostic figures were generated")
        gap = 24
        width = max(image.width for image in png_pages)
        height = sum(image.height for image in png_pages) + gap * (
            len(png_pages) - 1
        )
        combined = Image.new("RGB", (width, height), "white")
        y_offset = 0
        for image in png_pages:
            x_offset = (width - image.width) // 2
            combined.paste(image, (x_offset, y_offset))
            y_offset += image.height + gap
        combined.save(png_tmp_name, format="PNG", optimize=True)
        combined.close()
        for image in png_pages:
            image.close()
        os.replace(pdf_tmp_name, pdf_output_path)
        os.replace(png_tmp_name, png_output_path)
    except BaseException:
        for tmp_name in (pdf_tmp_name, png_tmp_name):
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
        for image in png_pages:
            image.close()
        raise



def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_frame_atomic(path: Path, frame: pd.DataFrame) -> None:
    write_csv_atomic(
        path,
        frame.to_dict(orient="records"),
        [str(column) for column in frame.columns],
    )


def _load_completed_mean(input_root: Path) -> tuple[Path, pd.DataFrame, list[Path]]:
    mean_dir = input_root / "mean3_summary"
    success_path = mean_dir / "_SUCCESS.json"
    summary_path = mean_dir / "mean_summary.csv"
    per_seed_path = mean_dir / "per_seed_metrics.csv"
    if not success_path.is_file():
        raise FileNotFoundError(f"Mean result is not complete: {mean_dir}")
    success = _load_json(success_path)
    if success.get("status") != "complete":
        raise ValueError(f"Mean result is not marked complete: {mean_dir}")
    if tuple(int(value) for value in success.get("eval_seeds", ())) != EVAL_SEEDS:
        raise ValueError(f"Mean result does not use exactly {list(EVAL_SEEDS)}: {mean_dir}")
    artifacts = {str(value) for value in success.get("artifacts", ())}
    for required in ("mean_summary.csv", "per_seed_metrics.csv"):
        if required not in artifacts or not (mean_dir / required).is_file():
            raise FileNotFoundError(f"Mean result is missing {required}: {mean_dir}")
    summary = pd.read_csv(summary_path)
    _require_columns(summary, MEAN_FIELDS, "mean summary")
    if summary.empty:
        raise ValueError(f"Empty mean summary: {summary_path}")
    if set(pd.to_numeric(summary["n_seeds"], errors="raise").astype(int)) != {3}:
        raise ValueError(f"Mean summary must contain exactly three seeds: {summary_path}")

    per_seed = pd.read_csv(per_seed_path)
    _require_columns(per_seed, PER_SEED_FIELDS, "per-seed metrics")
    identity = ["task", "scope", "model", "gallery_size", "metric"]
    recomputed = (
        per_seed.groupby(identity, dropna=False, sort=True)["value"]
        .agg(["sum", "count"])
        .reset_index()
    )
    if set(recomputed["count"].astype(int)) != {3}:
        raise ValueError(f"Per-seed metrics are not present exactly three times: {per_seed_path}")
    recomputed["expected_mean"] = recomputed["sum"] / 3.0
    checked = summary.merge(recomputed, on=identity, how="outer", validate="one_to_one")
    if checked[["mean", "expected_mean"]].isna().any().any() or not np.allclose(
        checked["mean"],
        checked["expected_mean"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Stored means are not equal to the arithmetic three-seed mean: {mean_dir}")
    return mean_dir, summary, [success_path, summary_path, per_seed_path]


def _load_seed_plot_inputs(
    input_root: Path,
    *,
    result_name: str,
    require_classification: bool,
) -> tuple[Dict[int, Dict[str, Any]], pd.DataFrame, list[Path]]:
    payloads: Dict[int, Dict[str, Any]] = {}
    predictions: list[pd.DataFrame] = []
    compatibility: list[Dict[str, Any]] = []
    source_paths: list[Path] = []
    for seed in EVAL_SEEDS:
        run_dir = input_root / f"seed_{seed}"
        manifest, success = _validated_manifest(run_dir, seed)
        artifacts = {str(value) for value in success["artifacts"]}
        if result_name not in artifacts:
            raise FileNotFoundError(f"Successful run does not declare {result_name}: {run_dir}")
        result_path = run_dir / result_name
        payloads[seed] = _load_json(result_path)
        source_paths.extend([run_dir / "_SUCCESS.json", run_dir / "run_manifest.json", result_path])

        classification_name = "classification_predictions.csv.gz"
        declared_classification = classification_name in artifacts
        if require_classification and not declared_classification:
            raise FileNotFoundError(f"Classification predictions are missing: {run_dir}")
        if declared_classification:
            prediction_path = run_dir / classification_name
            prediction = pd.read_csv(prediction_path)
            if prediction.empty:
                raise ValueError(f"Empty classification predictions: {prediction_path}")
            predictions.append(prediction)
            source_paths.append(prediction_path)

        compatibility.append(
            {
                "experiment_digest": manifest.get("experiment_digest"),
                "code_digest": manifest.get("code_digest"),
                "test_data_name": Path(str(manifest.get("test_data_path", ""))).name,
                "neg_data_name": Path(str(manifest.get("neg_data_path", ""))).name,
                "config": _config_signature(manifest),
            }
        )
    if len({stable_digest(value) for value in compatibility}) != 1:
        raise ValueError("Seed manifests, checkpoints, or data paths are not compatible")
    classification = (
        _validate_classification(pd.concat(predictions, ignore_index=True), EVAL_SEEDS)
        if predictions
        else pd.DataFrame()
    )
    return payloads, classification, source_paths


def _paper_table_frame(
    summary: pd.DataFrame,
    models: Sequence[str],
) -> pd.DataFrame:
    block = summary[
        (summary["task"] == "retrieval")
        & (summary["scope"] == "overall")
        & (summary["model"].isin(models))
        & (summary["metric"].isin([key for key, _ in PAPER_RETRIEVAL_METRICS]))
        & (summary["gallery_size"].isin(PAPER_GALLERY_SIZES))
    ].copy()
    expected = len(models) * len(PAPER_GALLERY_SIZES) * len(PAPER_RETRIEVAL_METRICS)
    if len(block) != expected:
        raise ValueError(
            f"Paper retrieval table expected {expected} mean values, found {len(block)}"
        )
    keyed = block.set_index(["model", "gallery_size", "metric"])
    rows: list[Dict[str, Any]] = []
    for model in models:
        row: Dict[str, Any] = {"Method": PAPER_TABLE_LABELS[model]}
        for gallery_size in PAPER_GALLERY_SIZES:
            for metric_key, metric_label in PAPER_RETRIEVAL_METRICS:
                key = (model, gallery_size, metric_key)
                if key not in keyed.index:
                    raise ValueError(f"Missing paper table value: {key}")
                row[f"G{gallery_size}_{metric_label}"] = float(keyed.loc[key, "mean"])
        rows.append(row)
    return pd.DataFrame(rows)


def _paper_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "font.size": 17,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,
            "savefig.dpi": 300,
        }
    )
    return plt


def _save_figure_atomic(figure: Any, pdf_path: Path, png_path: Path) -> None:
    if pdf_path.exists() or png_path.exists():
        raise FileExistsError(f"Refusing to overwrite figure pair: {pdf_path}, {png_path}")
    pdf_fd, pdf_tmp = tempfile.mkstemp(
        prefix=f".{pdf_path.name}.", suffix=".tmp.pdf", dir=pdf_path.parent
    )
    png_fd, png_tmp = tempfile.mkstemp(
        prefix=f".{png_path.name}.", suffix=".tmp.png", dir=png_path.parent
    )
    os.close(pdf_fd)
    os.close(png_fd)
    try:
        figure.savefig(
            pdf_tmp,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
        figure.savefig(
            png_tmp,
            format="png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
        if Path(pdf_tmp).stat().st_size == 0 or Path(png_tmp).stat().st_size == 0:
            raise RuntimeError("Matplotlib produced an empty figure")
        os.replace(pdf_tmp, pdf_path)
        os.replace(png_tmp, png_path)
    except BaseException:
        for temporary in (pdf_tmp, png_tmp):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def _latex_table_rows(frame: pd.DataFrame) -> list[str]:
    numeric = frame.iloc[:, 1:].astype(float)
    best_by_column: Dict[str, float] = {}
    second_by_column: Dict[str, Optional[float]] = {}
    for column in numeric.columns:
        unique_values = sorted(set(numeric[column].tolist()), reverse=True)
        best_by_column[str(column)] = float(unique_values[0])
        second_by_column[str(column)] = (
            float(unique_values[1]) if len(unique_values) > 1 else None
        )

    lines: list[str] = []
    for _, row in frame.iterrows():
        rendered = []
        for column in frame.columns[1:]:
            value = float(row[column])
            text = f"{value:.3f}"
            if np.isclose(
                value, best_by_column[str(column)], rtol=0.0, atol=1e-15
            ):
                text = rf"\textbf{{{text}}}"
            elif (
                second_by_column[str(column)] is not None
                and np.isclose(
                    value,
                    float(second_by_column[str(column)]),
                    rtol=0.0,
                    atol=1e-15,
                )
            ):
                text = rf"\underline{{{text}}}"
            rendered.append(text)
        lines.append(f"{row['Method']} & " + " & ".join(rendered) + r" \\")
    return lines


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(value) for value in command],
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = (result.stdout + "\n" + result.stderr)[-5000:]
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(command)}\n{details}"
        )
    return result


def _crop_pdf_with_ghostscript(source: Path, output: Path) -> None:
    bbox_result = _run_checked(
        [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=bbox",
            str(source),
        ]
    )
    bbox_output = bbox_result.stdout + "\n" + bbox_result.stderr
    bbox_lines = [
        line.strip()
        for line in bbox_output.splitlines()
        if line.strip().startswith("%%HiResBoundingBox:")
    ]
    if len(bbox_lines) != 1:
        raise RuntimeError(f"Expected one PDF bounding box, found {bbox_lines}: {bbox_output}")
    llx, lly, urx, ury = [
        float(value) for value in bbox_lines[0].split(":", 1)[1].split()
    ]
    padding = 7.0
    llx -= padding
    lly -= padding
    urx += padding
    ury += padding
    width = urx - llx
    height = ury - lly
    _run_checked(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dFIXEDMEDIA",
            f"-dDEVICEWIDTHPOINTS={width:.6f}",
            f"-dDEVICEHEIGHTPOINTS={height:.6f}",
            f"-sOutputFile={output}",
            "-c",
            f"<</PageOffset [{-llx:.6f} {-lly:.6f}]>> setpagedevice",
            "-f",
            str(source),
        ]
    )


def _plot_paper_table(
    frame: pd.DataFrame,
    *,
    caption: str,
    table_number: int,
    table_comment: str,
    paper_dir: Path,
    pdf_path: Path,
    png_path: Path,
) -> None:
    if pdf_path.exists() or png_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite table pair: {pdf_path}, {png_path}"
        )
    paper_dir = Path(paper_dir).resolve()
    if not (paper_dir / "aastex701.cls").is_file():
        raise FileNotFoundError(f"AASTeX class is missing: {paper_dir}")
    for executable in ("pdflatex", "gs", "pdftoppm"):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"Required table renderer is missing: {executable}")

    with tempfile.TemporaryDirectory(
        prefix=".aastex_table.", dir=pdf_path.parent
    ) as build_name:
        build = Path(build_name)
        (build / "epsf.sty").write_text(
            "\\ProvidesPackage{epsf}\n\\endinput\n", encoding="utf-8"
        )
        (build / "ulem.sty").write_text(
            "\\ProvidesPackage{ulem}\n"
            "\\DeclareOption*{}\n"
            "\\ProcessOptions\\relax\n"
            "\\endinput\n",
            encoding="utf-8",
        )
        tex_lines = [
            r"\documentclass[twocolumn]{aastex701}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            fr"\setcounter{{table}}{{{table_number - 1}}}",
            r"\begin{deluxetable*}{lcccccccccccc}",
            r"\tablewidth{\textwidth}",
            r"\tabletypesize{\small}",
            fr"\tablecaption{{{caption}}}",
            r"\tablehead{",
            r"\colhead{\raisebox{-0.65\normalbaselineskip}{Method}} &",
            r"\multicolumn{3}{c}{Gallery = 100} &",
            r"\multicolumn{3}{c}{Gallery = 500} &",
            r"\multicolumn{3}{c}{Gallery = 1000} &",
            r"\multicolumn{3}{c}{Gallery = 5000} \\",
            r"\cline{2-4}",
            r"\cline{5-7}",
            r"\cline{8-10}",
            r"\cline{11-13}",
            r"\colhead{} &",
            r"\colhead{R@1} & \colhead{R@10} & \colhead{MRR} &",
            r"\colhead{R@1} & \colhead{R@10} & \colhead{MRR} &",
            r"\colhead{R@1} & \colhead{R@10} & \colhead{MRR} &",
            r"\colhead{R@1} & \colhead{R@10} & \colhead{MRR}",
            r"}",
            r"\startdata",
            *_latex_table_rows(frame),
            r"\enddata",
            fr"\tablecomments{{{table_comment}}}",
            r"\end{deluxetable*}",
            r"\null",
            r"\clearpage",
            r"\end{document}",
        ]
        tex_path = build / "table.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

        environment = dict(os.environ)
        existing_texinputs = environment.get("TEXINPUTS", "")
        environment["TEXINPUTS"] = os.pathsep.join(
            [str(build), str(paper_dir), existing_texinputs]
        )
        _run_checked(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(build),
                str(tex_path),
            ],
            cwd=build,
            env=environment,
        )
        compiled_pdf = build / "table.pdf"
        cropped_pdf = build / "table_cropped.pdf"
        _crop_pdf_with_ghostscript(compiled_pdf, cropped_pdf)
        png_prefix = build / "table_cropped"
        _run_checked(
            [
                "pdftoppm",
                "-f",
                "1",
                "-singlefile",
                "-png",
                "-r",
                "300",
                str(cropped_pdf),
                str(png_prefix),
            ]
        )
        rendered_png = png_prefix.with_suffix(".png")
        if not cropped_pdf.is_file() or not rendered_png.is_file():
            raise RuntimeError("AASTeX table rendering did not produce PDF and PNG")
        os.replace(cropped_pdf, pdf_path)
        os.replace(rendered_png, png_path)


def _plot_retrieval_curves_mean(
    summary: pd.DataFrame,
    models: Sequence[str],
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    plt = _paper_pyplot()
    block = summary[
        (summary["task"] == "retrieval")
        & (summary["scope"] == "overall")
        & (summary["model"].isin(models))
    ].copy()
    gallery_sizes = sorted(
        int(value)
        for value in block.loc[
            block["metric"] == "recall_at_1", "gallery_size"
        ].dropna().unique()
    )
    if gallery_sizes != [10, 100, 500, 1000, 2000, 5000]:
        raise ValueError(f"Unexpected gallery sizes for paper curves: {gallery_sizes}")
    plot_models = [model for model in PAPER_PLOT_ORDER if model in models]
    if set(plot_models) != set(models):
        raise ValueError(f"Paper plot order is incomplete for models: {models}")

    figure, axes = plt.subplots(1, 3, figsize=(15, 6.4))
    ylabels = {
        "recall_at_1": "Recall@1",
        "recall_at_10": "Recall@10",
        "mrr": "MRR",
    }
    for axis, (metric_key, _) in zip(axes, PAPER_RETRIEVAL_METRICS):
        metric = block[block["metric"] == metric_key]
        for model in plot_models:
            model_rows = metric[metric["model"] == model].sort_values("gallery_size")
            if len(model_rows) != len(gallery_sizes):
                raise ValueError(f"Incomplete curve for {model}, {metric_key}")
            color, _, _ = PAPER_METHOD_STYLES[model]
            axis.plot(
                model_rows["gallery_size"],
                model_rows["mean"],
                color=color,
                marker="o",
                linestyle="-",
                linewidth=2,
                label=PAPER_METHOD_LABELS[model],
            )
        axis.set_xscale("log")
        axis.set_xlabel("Gallery size")
        axis.set_ylabel(ylabels[metric_key])
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, alpha=0.3)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=max(1, min(4, len(legend_labels))),
        frameon=False,
    )
    figure.canvas.draw()
    legend_bottom = legend.get_window_extent(
        renderer=figure.canvas.get_renderer()
    ).transformed(figure.transFigure.inverted()).y0
    for index, axis in enumerate(axes):
        axis.text(
            0.5,
            -0.24,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=17,
        )
    layout_top = max(0.80, min(0.94, legend_bottom - 0.012))
    figure.tight_layout(rect=(0, 0.06, 1, layout_top))
    _save_figure_atomic(figure, pdf_path, png_path)
    plt.close(figure)


def _average_redshift_macro_rows(
    payloads: Mapping[int, Mapping[str, Any]],
    *,
    identity_fields: Sequence[str],
    models: Sequence[str],
) -> pd.DataFrame:
    numeric_fields = (
        "redshift",
        "n_queries_mean",
        "macro_recall_at_1",
        "macro_recall_at_10",
        "macro_mrr",
    )
    indices: Dict[int, Dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    for seed in EVAL_SEEDS:
        rows = payloads[seed].get("redshift_macro_rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Seed {seed} has no redshift macro rows")
        index: Dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for row in rows:
            key = tuple(row[field] for field in identity_fields)
            if key in index:
                raise ValueError(f"Duplicate redshift macro identity for seed {seed}: {key}")
            if row.get("weight_scheme") != "log10(gallery_size)":
                raise ValueError(f"Unexpected redshift weighting for seed {seed}: {key}")
            if int(row.get("n_gallery_sizes", -1)) != 6:
                raise ValueError(f"Redshift macro row does not contain six galleries: {key}")
            index[key] = row
        indices[seed] = index
    reference_keys = set(indices[EVAL_SEEDS[0]])
    if any(set(indices[seed]) != reference_keys for seed in EVAL_SEEDS[1:]):
        raise ValueError("Redshift macro rows are not paired across all three seeds")

    averaged: list[Dict[str, Any]] = []
    for key in reference_keys:
        seed_rows = [indices[seed][key] for seed in EVAL_SEEDS]
        method = str(seed_rows[0]["method"])
        if method not in models:
            continue
        row: Dict[str, Any] = {
            field: seed_rows[0][field] for field in identity_fields
        }
        row["weight_scheme"] = "log10(gallery_size)"
        row["n_gallery_sizes"] = 6
        row["gallery_weight_sum"] = float(
            sum(float(value["gallery_weight_sum"]) for value in seed_rows) / 3.0
        )
        for field in numeric_fields:
            values = [float(value[field]) for value in seed_rows]
            if not all(np.isfinite(values)):
                raise ValueError(f"Non-finite redshift macro value: {key}, {field}")
            row[field] = float(sum(values) / 3.0)
        averaged.append(row)

    observed_models = {str(row["method"]) for row in averaged}
    if observed_models != set(models):
        raise ValueError(
            f"Redshift macro models differ from the paper set: {sorted(observed_models)}"
        )
    order = {model: index for index, model in enumerate(models)}
    return pd.DataFrame(
        sorted(
            averaged,
            key=lambda row: (order[str(row["method"])], float(row["redshift"])),
        )
    )


def _plot_redshift_macro_mean(
    frame: pd.DataFrame,
    models: Sequence[str],
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    plt = _paper_pyplot()
    plot_models = [model for model in PAPER_PLOT_ORDER if model in models]
    if set(plot_models) != set(models):
        raise ValueError(f"Paper plot order is incomplete for models: {models}")
    figure_size = (15, 6.4) if len(models) > 5 else (15, 5.875)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=figure_size,
        sharex=len(models) <= 5,
        sharey=False,
    )
    metrics = (
        ("macro_recall_at_1", "Macro Recall@1"),
        ("macro_recall_at_10", "Macro Recall@10"),
        ("macro_mrr", "Macro MRR"),
    )
    for axis, (metric, label) in zip(axes, metrics):
        for model in plot_models:
            model_rows = frame[frame["method"] == model].sort_values("redshift")
            color, _, _ = PAPER_METHOD_STYLES[model]
            axis.plot(
                model_rows["redshift"],
                model_rows[metric],
                color=color,
                marker="o",
                linestyle="-",
                linewidth=2,
                label=PAPER_METHOD_LABELS[model],
            )
        axis.set_xlabel("Redshift")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=max(1, min(4, len(legend_labels))),
        frameon=False,
    )
    figure.canvas.draw()
    legend_bottom = legend.get_window_extent(
        renderer=figure.canvas.get_renderer()
    ).transformed(figure.transFigure.inverted()).y0
    for index, axis in enumerate(axes):
        axis.text(
            0.5,
            -0.24,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=17,
        )
    layout_top = max(0.80, min(0.94, legend_bottom - 0.012))
    figure.tight_layout(rect=(0, 0.06, 1, layout_top))
    _save_figure_atomic(figure, pdf_path, png_path)
    plt.close(figure)


def _mean_logit_density(
    classification: pd.DataFrame,
    *,
    n_bins: int = 50,
) -> pd.DataFrame:
    full = classification[classification["model"] == "Full"].copy()
    if full.empty:
        raise ValueError("MAGIKS classification predictions are missing")
    observed_pairs = set(full["pair_type"].astype(str))
    if observed_pairs != set(PAPER_PAIR_STYLES):
        raise ValueError(f"Unexpected pair types: {sorted(observed_pairs)}")
    probability = pd.to_numeric(full["probability"], errors="raise").to_numpy(dtype=float)
    if np.any(~np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("Classification probabilities must be finite and within [0, 1]")
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    full["logit_margin"] = np.log(clipped / (1.0 - clipped))
    lo = min(float(full["logit_margin"].min()), 0.0)
    hi = max(float(full["logit_margin"].max()), 0.0)
    pad = max(0.05 * (hi - lo), 1e-3)
    bins = np.linspace(lo - pad, hi + pad, n_bins + 1)
    widths = np.diff(bins)

    rows: list[Dict[str, Any]] = []
    for pair_type in PAPER_PAIR_STYLES:
        counts_by_seed = []
        densities_by_seed = []
        for seed in EVAL_SEEDS:
            values = full[
                (full["seed"] == seed) & (full["pair_type"] == pair_type)
            ]["logit_margin"].to_numpy(dtype=float)
            if values.size == 0:
                raise ValueError(f"Missing {pair_type} predictions for seed {seed}")
            counts, _ = np.histogram(values, bins=bins)
            counts_by_seed.append(counts.astype(float))
            densities_by_seed.append(counts / (float(values.size) * widths))
        mean_count = np.mean(np.stack(counts_by_seed, axis=0), axis=0)
        mean_density = np.mean(np.stack(densities_by_seed, axis=0), axis=0)
        for index, density in enumerate(mean_density):
            rows.append(
                {
                    "pair_type": pair_type,
                    "bin_left": float(bins[index]),
                    "bin_right": float(bins[index + 1]),
                    "mean_count": float(mean_count[index]),
                    "mean_density": float(density),
                    "n_seeds": 3,
                }
            )
    return pd.DataFrame(rows)


def _plot_logit_density_mean(
    frame: pd.DataFrame,
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    from matplotlib.ticker import MaxNLocator

    plt = _paper_pyplot()
    figure, axis = plt.subplots(figsize=(10, 6))
    for pair_type, (color, label) in PAPER_PAIR_STYLES.items():
        block = frame[frame["pair_type"] == pair_type].sort_values("bin_left")
        edges = np.concatenate(
            [
                block["bin_left"].to_numpy(dtype=float),
                [float(block["bin_right"].iloc[-1])],
            ]
        )
        axis.stairs(
            block["mean_count"].to_numpy(dtype=float),
            edges,
            color=color,
            linewidth=2,
            alpha=0.6,
            label=label,
        )
    axis.set_xlabel("Logit", fontsize=17)
    axis.set_ylabel("Count", fontsize=17)
    axis.grid(True, alpha=0.3, linestyle="--")
    axis.legend(loc="upper left", fontsize=15)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    _save_figure_atomic(figure, pdf_path, png_path)
    plt.close(figure)


def _mean_confusion_rows(classification: pd.DataFrame) -> pd.DataFrame:
    full = classification[classification["model"] == "Full"].copy()
    if full.empty:
        raise ValueError("MAGIKS classification predictions are missing")
    full["label"] = pd.to_numeric(full["label"], errors="raise").astype(int)
    full["prediction"] = (
        pd.to_numeric(full["probability"], errors="raise") >= 0.5
    ).astype(int)
    rows: list[Dict[str, Any]] = []
    for source_type in ("bns", "nsbh"):
        matrices = []
        for seed in EVAL_SEEDS:
            block = full[
                (full["seed"] == seed) & (full["source_type"] == source_type)
            ]
            matrix = np.zeros((2, 2), dtype=float)
            for true_label in (0, 1):
                truth = block[block["label"] == true_label]
                if truth.empty:
                    raise ValueError(
                        f"Missing true label {true_label} for {source_type}, seed {seed}"
                    )
                for predicted_label in (0, 1):
                    matrix[true_label, predicted_label] = float(
                        (truth["prediction"] == predicted_label).mean()
                    )
            matrices.append(matrix)
        mean_matrix = np.mean(np.stack(matrices, axis=0), axis=0)
        for true_label in (0, 1):
            for predicted_label in (0, 1):
                rows.append(
                    {
                        "source_type": source_type,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "mean_fraction": float(mean_matrix[true_label, predicted_label]),
                        "n_seeds": 3,
                        "threshold": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def _plot_confusion_mean(
    frame: pd.DataFrame,
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    plt = _paper_pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, (axis, source_type) in enumerate(zip(axes, ("bns", "nsbh"))):
        block = frame[frame["source_type"] == source_type]
        matrix = np.zeros((2, 2), dtype=float)
        for _, row in block.iterrows():
            matrix[int(row["true_label"]), int(row["predicted_label"])] = float(
                row["mean_fraction"]
            )
        image_handle = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        figure.colorbar(image_handle, ax=axis, fraction=0.046, pad=0.04)
        for true_label in (0, 1):
            for predicted_label in (0, 1):
                value = matrix[true_label, predicted_label]
                axis.text(
                    predicted_label,
                    true_label,
                    f"{100.0 * value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                    fontsize=15,
                    fontweight="bold",
                )
        axis.set_xticks((0, 1), ("Negative", "Positive"))
        axis.set_yticks((0, 1), ("Negative", "Positive"))
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.text(
            0.5,
            -0.25,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=17,
        )
    figure.tight_layout(rect=(0, 0.08, 1, 1), w_pad=4.5)
    _save_figure_atomic(figure, pdf_path, png_path)
    plt.close(figure)


def _source_digests(paths: Sequence[Path]) -> Dict[str, str]:
    unique = sorted({Path(path).resolve() for path in paths}, key=str)
    return {str(path): _sha256_file(path) for path in unique}


def _generate_retrieval_paper_assets(
    stage: Path,
    summary: pd.DataFrame,
    payloads: Mapping[int, Mapping[str, Any]],
    classification: pd.DataFrame,
    paper_dir: Path,
    source_paths: Sequence[Path],
) -> Dict[str, Any]:
    artifacts: list[str] = []

    table = _paper_table_frame(summary, PAPER_RETRIEVAL_MODELS)
    table_csv = "table4_retrieval_ablation_mean3.csv"
    _write_frame_atomic(stage / table_csv, table)
    artifacts.append(table_csv)
    _plot_paper_table(
        table,
        caption=TABLE4_CAPTION,
        table_number=4,
        table_comment=TABLE4_COMMENT,
        paper_dir=paper_dir,
        pdf_path=stage / "table4_retrieval_ablation_mean3.pdf",
        png_path=stage / "table4_retrieval_ablation_mean3.png",
    )
    artifacts.extend(
        ["table4_retrieval_ablation_mean3.pdf", "table4_retrieval_ablation_mean3.png"]
    )

    _plot_retrieval_curves_mean(
        summary,
        PAPER_RETRIEVAL_MODELS,
        pdf_path=stage / "figure6_retrieval_curves_mean3.pdf",
        png_path=stage / "figure6_retrieval_curves_mean3.png",
    )
    artifacts.extend(
        ["figure6_retrieval_curves_mean3.pdf", "figure6_retrieval_curves_mean3.png"]
    )

    redshift = _average_redshift_macro_rows(
        payloads,
        identity_fields=("method", "redshift_bin_label", "bin_left", "bin_right"),
        models=PAPER_RETRIEVAL_MODELS,
    )
    redshift_csv = "figure7_redshift_macro_mean3.csv"
    _write_frame_atomic(stage / redshift_csv, redshift)
    artifacts.append(redshift_csv)
    _plot_redshift_macro_mean(
        redshift,
        PAPER_RETRIEVAL_MODELS,
        pdf_path=stage / "figure7_redshift_macro_mean3.pdf",
        png_path=stage / "figure7_redshift_macro_mean3.png",
    )
    artifacts.extend(
        ["figure7_redshift_macro_mean3.pdf", "figure7_redshift_macro_mean3.png"]
    )

    density = _mean_logit_density(classification)
    density_csv = "figure8_logit_distribution_mean3.csv"
    _write_frame_atomic(stage / density_csv, density)
    artifacts.append(density_csv)
    _plot_logit_density_mean(
        density,
        pdf_path=stage / "figure8_logit_distribution_mean3.pdf",
        png_path=stage / "figure8_logit_distribution_mean3.png",
    )
    artifacts.extend(
        ["figure8_logit_distribution_mean3.pdf", "figure8_logit_distribution_mean3.png"]
    )

    confusion = _mean_confusion_rows(classification)
    confusion_csv = "figure9_confusion_matrix_mean3.csv"
    _write_frame_atomic(stage / confusion_csv, confusion)
    artifacts.append(confusion_csv)
    _plot_confusion_mean(
        confusion,
        pdf_path=stage / "figure9_confusion_matrix_mean3.pdf",
        png_path=stage / "figure9_confusion_matrix_mean3.png",
    )
    artifacts.extend(
        ["figure9_confusion_matrix_mean3.pdf", "figure9_confusion_matrix_mean3.png"]
    )

    success = {
        "status": "complete",
        "eval_seeds": list(EVAL_SEEDS),
        "gallery_trials_per_seed": 10,
        "aggregation": "equal arithmetic mean of per-seed results",
        "classification_threshold": 0.5,
        "artifacts": artifacts,
        "source_sha256": _source_digests(source_paths),
        "notes": [
            "Only equal arithmetic means are included.",
            "Figure 8 averages per-seed histogram counts; logit margin is reconstructed as log(p/(1-p)) from the stored positive-class probability.",
            "Figure 9 averages per-seed row-normalized confusion matrices at threshold 0.5.",
        ],
        "not_generated": {
            "table5_classification_ablation": (
                "The three-seed predictions do not contain the paper's w/o fusion "
                "or w/o classification loss models, so an exact replacement would be unsupported."
            )
        },
    }
    write_json_atomic(stage / "_SUCCESS.json", success)
    return success


def _generate_gw170817_paper_assets(
    stage: Path,
    summary: pd.DataFrame,
    payloads: Mapping[int, Mapping[str, Any]],
    paper_dir: Path,
    source_paths: Sequence[Path],
) -> Dict[str, Any]:
    artifacts: list[str] = []

    table = _paper_table_frame(summary, PAPER_GW170817_MODELS)
    table_csv = "table6_gw170817_retrieval_mean3.csv"
    _write_frame_atomic(stage / table_csv, table)
    artifacts.append(table_csv)
    _plot_paper_table(
        table,
        caption=TABLE6_CAPTION,
        table_number=6,
        table_comment=TABLE6_COMMENT,
        paper_dir=paper_dir,
        pdf_path=stage / "table6_gw170817_retrieval_mean3.pdf",
        png_path=stage / "table6_gw170817_retrieval_mean3.png",
    )
    artifacts.extend(
        ["table6_gw170817_retrieval_mean3.pdf", "table6_gw170817_retrieval_mean3.png"]
    )

    _plot_retrieval_curves_mean(
        summary,
        PAPER_GW170817_MODELS,
        pdf_path=stage / "figure11_gw170817_retrieval_curves_mean3.pdf",
        png_path=stage / "figure11_gw170817_retrieval_curves_mean3.png",
    )
    artifacts.extend(
        [
            "figure11_gw170817_retrieval_curves_mean3.pdf",
            "figure11_gw170817_retrieval_curves_mean3.png",
        ]
    )

    redshift = _average_redshift_macro_rows(
        payloads,
        identity_fields=("method", "redshift_bin"),
        models=PAPER_GW170817_MODELS,
    )
    redshift_csv = "figure12_gw170817_redshift_macro_mean3.csv"
    _write_frame_atomic(stage / redshift_csv, redshift)
    artifacts.append(redshift_csv)
    _plot_redshift_macro_mean(
        redshift,
        PAPER_GW170817_MODELS,
        pdf_path=stage / "figure12_gw170817_redshift_macro_mean3.pdf",
        png_path=stage / "figure12_gw170817_redshift_macro_mean3.png",
    )
    artifacts.extend(
        [
            "figure12_gw170817_redshift_macro_mean3.pdf",
            "figure12_gw170817_redshift_macro_mean3.png",
        ]
    )

    success = {
        "status": "complete",
        "eval_seeds": list(EVAL_SEEDS),
        "gallery_trials_per_seed": 10,
        "aggregation": "equal arithmetic mean of per-seed results",
        "artifacts": artifacts,
        "source_sha256": _source_digests(source_paths),
        "notes": [
            "Only equal arithmetic means are included."
        ],
    }
    write_json_atomic(stage / "_SUCCESS.json", success)
    return success


def generate_paper_assets(
    config_path: Path, *, paper_dir: Path
) -> Dict[str, Any]:
    """Create mean-based paper replacement assets without touching the paper tree."""
    paper_dir = Path(paper_dir).expanduser().resolve()
    if not (paper_dir / "main.tex").is_file():
        raise FileNotFoundError(f"Paper source is missing: {paper_dir}")
    raw = _load_json(config_path)
    seeds = tuple(int(value) for value in raw.get("eval_seeds", ()))
    if seeds != EVAL_SEEDS:
        raise ValueError(f"eval_seeds must be exactly {list(EVAL_SEEDS)} in this order")
    if int(raw.get("gallery_trials", -1)) != 10:
        raise ValueError("gallery_trials must be exactly 10")
    output_root = _resolve(config_path.parent, str(raw["output_root"]))
    retrieval_root = output_root / "retrieval"
    gw170817_root = output_root / "gw170817"

    retrieval_mean_dir, retrieval_summary, retrieval_mean_sources = _load_completed_mean(
        retrieval_root
    )
    gw_mean_dir, gw_summary, gw_mean_sources = _load_completed_mean(gw170817_root)
    retrieval_payloads, classification, retrieval_seed_sources = _load_seed_plot_inputs(
        retrieval_root,
        result_name="ablation_comparison.json",
        require_classification=True,
    )
    gw_payloads, _, gw_seed_sources = _load_seed_plot_inputs(
        gw170817_root,
        result_name="gw170817a_retrieval.json",
        require_classification=False,
    )

    retrieval_output = retrieval_mean_dir / "paper_replacements"
    gw_output = gw_mean_dir / "paper_replacements"
    for output in (retrieval_output, gw_output):
        if output.exists():
            raise FileExistsError(f"Paper replacement output already exists: {output}")

    retrieval_stage = Path(
        tempfile.mkdtemp(prefix=".paper_replacements.", dir=retrieval_mean_dir)
    )
    gw_stage = Path(tempfile.mkdtemp(prefix=".paper_replacements.", dir=gw_mean_dir))
    paper_sources = [paper_dir / "main.tex", paper_dir / "aastex701.cls"]
    try:
        retrieval_success = _generate_retrieval_paper_assets(
            retrieval_stage,
            retrieval_summary,
            retrieval_payloads,
            classification,
            paper_dir,
            [
                *retrieval_mean_sources,
                *retrieval_seed_sources,
                *paper_sources,
            ],
        )
        gw_success = _generate_gw170817_paper_assets(
            gw_stage,
            gw_summary,
            gw_payloads,
            paper_dir,
            [*gw_mean_sources, *gw_seed_sources, *paper_sources],
        )
        for stage, output in (
            (retrieval_stage, retrieval_output),
            (gw_stage, gw_output),
        ):
            if output.exists():
                raise FileExistsError(f"Paper replacement output appeared during generation: {output}")
            os.replace(stage, output)
    finally:
        for stage in (retrieval_stage, gw_stage):
            if stage.exists():
                shutil.rmtree(stage)

    result = {
        "retrieval_output": str(retrieval_output),
        "gw170817_output": str(gw_output),
        "retrieval_artifacts": retrieval_success["artifacts"],
        "gw170817_artifacts": gw_success["artifacts"],
    }
    print(f"Generated paper replacement assets in {retrieval_output}")
    print(f"Generated paper replacement assets in {gw_output}")
    return result

def aggregate_seed_mean(
    config: Mapping[str, Any], *, config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    seeds = _validate_config(config)
    base = Path(config_dir or Path.cwd()).resolve()
    input_root = _resolve(base, str(config["input_root"]))
    output_dir = _resolve(
        base, str(config.get("output_dir", input_root / "mean3_summary"))
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Mean output directory already exists: {output_dir}"
        )
    reference_model = str(config.get("reference_model", "Full"))
    retrieval, classification, compatibility = _read_runs(input_root, seeds)
    retrieval = _validate_retrieval(retrieval, seeds, trials=10)
    per_seed_rows = summarize_retrieval_by_seed(retrieval, seeds)
    if not classification.empty:
        classification = _validate_classification(classification, seeds)
        per_seed_rows.extend(
            summarize_classification_by_seed(classification, seeds)
        )
    summary_rows, delta_rows = equal_seed_means(
        per_seed_rows, seeds, reference_model
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv_atomic(
        output_dir / "per_seed_metrics.csv",
        per_seed_rows,
        PER_SEED_FIELDS,
    )
    write_csv_atomic(
        output_dir / "mean_summary.csv", summary_rows, MEAN_FIELDS
    )
    write_csv_atomic(
        output_dir / "mean_deltas.csv", delta_rows, DELTA_FIELDS
    )
    payload = {
        "eval_seeds": list(seeds),
        "reference_model": reference_model,
        "per_seed_metrics": per_seed_rows,
        "mean_summary": summary_rows,
        "mean_deltas": delta_rows,
    }
    write_json_atomic(output_dir / "mean_summary.json", payload)
    _plot_diagnostics(
        per_seed_rows,
        summary_rows,
        output_dir / "mean_diagnostics.pdf",
        output_dir / "mean_diagnostics.png",
        seeds,
    )
    artifacts = [
        "per_seed_metrics.csv",
        "mean_summary.csv",
        "mean_summary.json",
        "mean_deltas.csv",
        "mean_diagnostics.pdf",
        "mean_diagnostics.png",
    ]
    write_json_atomic(
        output_dir / "_SUCCESS.json",
        {
            "status": "complete",
            "eval_seeds": list(seeds),
            "reference_model": reference_model,
            "experiment_digest": compatibility["experiment_digest"],
            "code_digest": compatibility["code_digest"],
            "artifacts": artifacts,
        },
    )
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Generate seed and mean configs")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--dry-run", action="store_true")
    aggregate_parser = commands.add_parser("aggregate", help="Compute the three-seed mean")
    aggregate_parser.add_argument("--config", required=True)
    paper_parser = commands.add_parser(
        "paper-assets", help="Create mean-based paper replacement PDFs and PNGs"
    )
    paper_parser.add_argument("--config", required=True)
    paper_parser.add_argument(
        "--paper-dir",
        default=str(Path(__file__).resolve().parents[4] / "paper_apj"),
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if args.command == "prepare":
        prepare_experiment(config_path, dry_run=args.dry_run)
    elif args.command == "aggregate":
        aggregate_seed_mean(
            _load_json(config_path), config_dir=config_path.parent
        )
    else:
        generate_paper_assets(
            config_path,
            paper_dir=Path(args.paper_dir),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
