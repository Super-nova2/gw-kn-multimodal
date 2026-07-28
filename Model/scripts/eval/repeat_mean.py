#!/usr/bin/env python3
"""Prepare and aggregate exactly three fixed-checkpoint evaluation seeds."""

from __future__ import annotations

import argparse
import json
import os
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
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if args.command == "prepare":
        prepare_experiment(config_path, dry_run=args.dry_run)
    else:
        aggregate_seed_mean(
            _load_json(config_path), config_dir=config_path.parent
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
