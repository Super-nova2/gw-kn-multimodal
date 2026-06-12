#!/usr/bin/env python
"""Benchmark pure forward compute speed for the GW-optical ALBEF model.

The timed regions assume all input tensors are already resident on the GPU.
Checkpoint loading, HDF5 reads, CPU preprocessing, and CPU-to-GPU transfers are
setup costs and are intentionally excluded from the reported latency numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import socket
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

import h5py
import numpy as np
import torch
from torch.amp import autocast


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


DEFAULT_BATCH_SIZES = [1, 8, 32, 128, 256, 512, 1024]
DEFAULT_PRECISION_MODES = ["fp32", "amp"]
VALID_PRECISION_MODES = {"fp32", "amp"}
DEFAULT_WARMUP_ITERATIONS = 20
DEFAULT_MEASURE_ITERATIONS = 100
CSV_FIELDS = [
    "checkpoint",
    "batch_size",
    "precision",
    "amp_dtype",
    "phase",
    "status",
    "warmup_iterations",
    "measure_iterations",
    "mean_ms",
    "median_ms",
    "p90_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "std_ms",
    "samples_per_sec_mean",
    "samples_per_sec_median",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "error",
]


@dataclass
class CpuBatch:
    gw_s: torch.Tensor
    gw_m: torch.Tensor
    opt_t: torch.Tensor
    opt_v: torch.Tensor
    opt_mask: torch.Tensor
    opt_err: torch.Tensor
    opt_coords: torch.Tensor
    gw_event_time_mjd: Optional[torch.Tensor]
    opt_event_time_mjd: Optional[torch.Tensor]
    input_shapes: Dict[str, List[int]]


@dataclass
class GpuBatch:
    gw_s: torch.Tensor
    gw_m: torch.Tensor
    opt_t: torch.Tensor
    opt_v: torch.Tensor
    opt_mask: torch.Tensor
    opt_err: torch.Tensor
    opt_coords: torch.Tensor
    opt_ref_t: torch.Tensor
    gw_event_time_mjd: Optional[torch.Tensor]
    opt_event_time_mjd: Optional[torch.Tensor]
    dt_days: torch.Tensor


def load_json_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Benchmark config must contain a JSON object: {path}")
    return cfg


def _require_path(cfg: Mapping[str, Any], key: str) -> str:
    value = str(cfg.get(key, "") or "").strip()
    if not value:
        raise ValueError(f"Required field missing in benchmark config: {key}")
    return value


def _normalize_int_list(value: Any, *, key: str, default: Iterable[int]) -> List[int]:
    if value is None:
        items = list(default)
    elif isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        raise ValueError(f"{key} must be a list or comma-separated string.")

    out: List[int] = []
    for item in items:
        try:
            parsed = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} contains a non-integer value: {item!r}") from exc
        if parsed <= 0:
            raise ValueError(f"{key} values must be positive; got {parsed}.")
        out.append(parsed)
    if not out:
        raise ValueError(f"{key} must not be empty.")
    return sorted(dict.fromkeys(out))


def _normalize_precision_modes(value: Any) -> List[str]:
    if value is None:
        items = list(DEFAULT_PRECISION_MODES)
    elif isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        raise ValueError("precision_modes must be a list or comma-separated string.")

    out: List[str] = []
    for item in items:
        mode = str(item).strip().lower()
        if mode not in VALID_PRECISION_MODES:
            raise ValueError(
                f"Invalid precision mode {item!r}; expected one of {sorted(VALID_PRECISION_MODES)}."
            )
        out.append(mode)
    if not out:
        raise ValueError("precision_modes must not be empty.")
    return list(dict.fromkeys(out))


def _positive_int(cfg: Mapping[str, Any], key: str, default: int, *, allow_zero: bool = False) -> int:
    raw = cfg.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer; got {raw!r}.") from exc
    if allow_zero:
        if value < 0:
            raise ValueError(f"{key} must be >= 0; got {value}.")
    elif value <= 0:
        raise ValueError(f"{key} must be > 0; got {value}.")
    return value


def normalize_config(cfg: Mapping[str, Any], *, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Validate and fill defaults for a benchmark config."""
    normalized = dict(cfg)
    normalized["checkpoint"] = _require_path(cfg, "checkpoint")
    normalized["data_path"] = _require_path(cfg, "data_path")
    normalized["output_dir"] = _require_path(cfg, "output_dir")
    model_config = str(cfg.get("model_config", "") or "").strip()
    normalized["model_config"] = model_config or None
    normalized["batch_sizes"] = _normalize_int_list(
        cfg.get("batch_sizes"),
        key="batch_sizes",
        default=DEFAULT_BATCH_SIZES,
    )
    normalized["precision_modes"] = _normalize_precision_modes(cfg.get("precision_modes"))
    normalized["warmup_iterations"] = _positive_int(
        cfg, "warmup_iterations", DEFAULT_WARMUP_ITERATIONS, allow_zero=True
    )
    normalized["measure_iterations"] = _positive_int(
        cfg, "measure_iterations", DEFAULT_MEASURE_ITERATIONS
    )
    normalized["seed"] = _positive_int(cfg, "seed", 42, allow_zero=True)
    normalized["device"] = str(cfg.get("device", "cuda") or "cuda")
    normalized["nonkn_cls_base_field"] = cfg.get("nonkn_cls_base_field", None)
    normalized["config_path"] = None if config_path is None else str(config_path)
    return normalized


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def summarize_latency_ms(values_ms: Iterable[float], batch_size: int) -> Dict[str, float]:
    values = np.asarray(list(values_ms), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty latency list.")
    mean_ms = float(values.mean())
    median_ms = float(np.percentile(values, 50))
    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "std_ms": float(values.std(ddof=0)),
        "samples_per_sec_mean": float(batch_size * 1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
        "samples_per_sec_median": float(batch_size * 1000.0 / median_ms) if median_ms > 0 else float("inf"),
    }


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in text
        or "cublas_status_alloc_failed" in text
        or "cuda error: out of memory" in text
    )


def _precision_context(device: torch.device, amp_dtype: torch.dtype, amp_enabled: bool):
    if device.type == "cuda":
        return autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
    return nullcontext()


def resolve_precision(mode: str, device: torch.device) -> Dict[str, Any]:
    normalized = str(mode).lower()
    if normalized == "fp32":
        return {
            "precision": "fp32",
            "amp_enabled": False,
            "amp_dtype": torch.float32,
            "amp_dtype_name": "fp32",
        }
    if normalized != "amp":
        raise ValueError(f"Invalid precision mode: {mode}")
    if device.type != "cuda":
        raise RuntimeError("AMP benchmark requires CUDA.")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "precision": "amp",
        "amp_enabled": True,
        "amp_dtype": amp_dtype,
        "amp_dtype_name": "bf16" if amp_dtype is torch.bfloat16 else "fp16",
    }


def _repeat_first_dim(tensor: torch.Tensor, target: int) -> torch.Tensor:
    if tensor.shape[0] >= target:
        return tensor[:target]
    indices = torch.arange(target) % int(tensor.shape[0])
    return tensor[indices]


def read_cpu_batch(
    data_path: str,
    max_batch_size: int,
    *,
    ref_start: float,
    ref_end: float,
) -> CpuBatch:
    """Load one real batch from HDF5 into CPU tensors.

    This function is setup-only. It must be called before any timed region.
    """
    from data_loader import apply_runtime_input_window_torch

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive.")

    with h5py.File(data_path, "r") as f:
        n_opt = int(f["events/optical_data/values"].shape[0])
        if n_opt <= 0:
            raise ValueError(f"No optical samples found in {data_path}")
        n_read = min(max_batch_size, n_opt)

        opt_v = torch.from_numpy(np.asarray(f["events/optical_data/values"][:n_read], dtype=np.float32))
        opt_err = torch.from_numpy(np.asarray(f["events/optical_data/errors"][:n_read], dtype=np.float32))
        opt_mask = torch.from_numpy(np.asarray(f["events/optical_data/masks"][:n_read], dtype=np.float32))
        opt_t = torch.from_numpy(np.asarray(f["events/optical_data/times"][:n_read], dtype=np.float32))
        opt_coords = torch.from_numpy(np.asarray(f["events/optical_data/coordinates"][:n_read], dtype=np.float32))

        parent = np.asarray(f["events/optical_data/parent_gw_idx"][:n_read], dtype=np.int64)
        unique_parent, inverse = np.unique(parent, return_inverse=True)
        gw_s_unique = np.asarray(f["events/gw_data/scalars"][unique_parent], dtype=np.float32)
        gw_m_unique = np.asarray(f["events/gw_data/skymaps"][unique_parent], dtype=np.float32)
        gw_s = torch.from_numpy(gw_s_unique[inverse])
        gw_m = torch.from_numpy(gw_m_unique[inverse])

        gw_event_time_mjd = None
        if "events/gw_data/event_time_mjd" in f:
            event_unique = np.asarray(f["events/gw_data/event_time_mjd"][unique_parent], dtype=np.float32)
            gw_event_time_mjd = torch.from_numpy(event_unique[inverse])

        opt_event_time_mjd = None
        if "events/optical_data/first_detection_mjd" in f:
            opt_event_time_mjd = torch.from_numpy(
                np.asarray(f["events/optical_data/first_detection_mjd"][:n_read], dtype=np.float32)
            )
        elif "events/optical_data/zero_time_mjd_base" in f:
            opt_event_time_mjd = torch.from_numpy(
                np.asarray(f["events/optical_data/zero_time_mjd_base"][:n_read], dtype=np.float32)
            )

    opt_t, opt_v, opt_mask, opt_err, _ = apply_runtime_input_window_torch(
        opt_t,
        opt_v,
        opt_mask,
        opt_err,
        None,
        window_start=ref_start,
        window_end=ref_end,
    )

    if n_read < max_batch_size:
        opt_v = _repeat_first_dim(opt_v, max_batch_size)
        opt_err = _repeat_first_dim(opt_err, max_batch_size)
        opt_mask = _repeat_first_dim(opt_mask, max_batch_size)
        opt_t = _repeat_first_dim(opt_t, max_batch_size)
        opt_coords = _repeat_first_dim(opt_coords, max_batch_size)
        gw_s = _repeat_first_dim(gw_s, max_batch_size)
        gw_m = _repeat_first_dim(gw_m, max_batch_size)
        if gw_event_time_mjd is not None:
            gw_event_time_mjd = _repeat_first_dim(gw_event_time_mjd, max_batch_size)
        if opt_event_time_mjd is not None:
            opt_event_time_mjd = _repeat_first_dim(opt_event_time_mjd, max_batch_size)

    input_shapes = {
        "gw_s": list(gw_s.shape),
        "gw_m": list(gw_m.shape),
        "opt_t": list(opt_t.shape),
        "opt_v": list(opt_v.shape),
        "opt_mask": list(opt_mask.shape),
        "opt_err": list(opt_err.shape),
        "opt_coords": list(opt_coords.shape),
    }
    return CpuBatch(
        gw_s=gw_s,
        gw_m=gw_m,
        opt_t=opt_t,
        opt_v=opt_v,
        opt_mask=opt_mask,
        opt_err=opt_err,
        opt_coords=opt_coords,
        gw_event_time_mjd=gw_event_time_mjd,
        opt_event_time_mjd=opt_event_time_mjd,
        input_shapes=input_shapes,
    )


def build_ref_time(batch_size: int, n_ref: int, ref_start: float, ref_end: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ref = torch.linspace(float(ref_start), float(ref_end), int(n_ref), dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(int(batch_size), 1)


def stage_gpu_batch(cpu_batch: CpuBatch, batch_size: int, device: torch.device, model_args: Mapping[str, Any]) -> GpuBatch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    gw_s = cpu_batch.gw_s[:batch_size].to(device=device, non_blocking=False)
    gw_m = cpu_batch.gw_m[:batch_size].to(device=device, non_blocking=False)
    opt_t = cpu_batch.opt_t[:batch_size].to(device=device, non_blocking=False)
    opt_v = cpu_batch.opt_v[:batch_size].to(device=device, non_blocking=False)
    opt_mask = cpu_batch.opt_mask[:batch_size].to(device=device, non_blocking=False)
    opt_err = cpu_batch.opt_err[:batch_size].to(device=device, non_blocking=False)
    opt_coords = cpu_batch.opt_coords[:batch_size].to(device=device, non_blocking=False)

    gw_event_time = None
    if cpu_batch.gw_event_time_mjd is not None:
        gw_event_time = cpu_batch.gw_event_time_mjd[:batch_size].to(device=device, dtype=torch.float32)
    opt_event_time = None
    if cpu_batch.opt_event_time_mjd is not None:
        opt_event_time = cpu_batch.opt_event_time_mjd[:batch_size].to(device=device, dtype=torch.float32)

    if gw_event_time is not None and opt_event_time is not None:
        dt_days = (opt_event_time - gw_event_time).to(dtype=torch.float32)
    else:
        dt_days = torch.zeros((batch_size,), device=device, dtype=torch.float32)

    opt_ref_t = build_ref_time(
        batch_size,
        int(model_args.get("n_ref", 64)),
        float(model_args.get("ref_start", -0.3)),
        float(model_args.get("ref_end", 0.6)),
        device,
        opt_t.dtype,
    )

    torch.cuda.synchronize(device)
    return GpuBatch(
        gw_s=gw_s,
        gw_m=gw_m,
        opt_t=opt_t,
        opt_v=opt_v,
        opt_mask=opt_mask,
        opt_err=opt_err,
        opt_coords=opt_coords,
        opt_ref_t=opt_ref_t,
        gw_event_time_mjd=gw_event_time,
        opt_event_time_mjd=opt_event_time,
        dt_days=dt_days,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def model_uses_cred_level(model: torch.nn.Module) -> bool:
    core = unwrap_model(model)
    fn = getattr(core, "uses_cred_level_input", None)
    return bool(fn()) if callable(fn) else False


def compute_cred_level_if_needed(model: torch.nn.Module, batch: GpuBatch) -> Optional[torch.Tensor]:
    if not model_uses_cred_level(model):
        return None
    from ALBEF_train import compute_credible_level

    return compute_credible_level(batch.gw_m, batch.opt_coords)


def encode_model(
    model: torch.nn.Module,
    batch: GpuBatch,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
):
    with _precision_context(device, amp_dtype, amp_enabled):
        return model.encode(
            batch.gw_s,
            batch.gw_m,
            batch.opt_coords,
            batch.opt_t,
            batch.opt_v,
            batch.opt_ref_t,
            batch.opt_mask,
            batch.opt_err,
        )


def similarity_with_time_compat(
    model: torch.nn.Module,
    feat_g: torch.Tensor,
    feat_o: torch.Tensor,
    gw_event_time_mjd: Optional[torch.Tensor],
    opt_event_time_mjd: Optional[torch.Tensor],
) -> torch.Tensor:
    core = unwrap_model(model)
    temperature = core.log_temp.exp().clamp(min=core.temp_min, max=core.temp_max)
    sim_g2o = torch.matmul(feat_g, feat_o.T) / temperature
    if gw_event_time_mjd is not None and hasattr(core, "_build_time_compat_bias"):
        time_bias = core._build_time_compat_bias(gw_event_time_mjd, opt_event_time_mjd)
        if time_bias is not None:
            sim_g2o = sim_g2o + time_bias.to(device=sim_g2o.device, dtype=sim_g2o.dtype)
    return sim_g2o


def full_multitask_forward(
    model: torch.nn.Module,
    batch: GpuBatch,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
):
    with _precision_context(device, amp_dtype, amp_enabled):
        g, z_l, h_l, h_gw = model.encode(
            batch.gw_s,
            batch.gw_m,
            batch.opt_coords,
            batch.opt_t,
            batch.opt_v,
            batch.opt_ref_t,
            batch.opt_mask,
            batch.opt_err,
        )
        feat_g, feat_o = model.get_contrastive_embeddings(g, z_l)
        sim_g2o = similarity_with_time_compat(
            model,
            feat_g,
            feat_o,
            batch.gw_event_time_mjd,
            batch.opt_event_time_mjd,
        )
        cred_level = compute_cred_level_if_needed(model, batch)
        logits = model.fusion_logits(
            g,
            h_l,
            z_l=z_l,
            H_gw=h_gw,
            cred_level=cred_level,
            gw_s=batch.gw_s,
            gw_m=batch.gw_m,
            opt_coords=batch.opt_coords,
            dt_days=batch.dt_days,
        )
    return sim_g2o, logits


def gw_encoder_forward(
    model: torch.nn.Module,
    batch: GpuBatch,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
):
    with _precision_context(device, amp_dtype, amp_enabled):
        return unwrap_model(model).gw_encoder(batch.gw_s, batch.gw_m)


def optical_encoder_forward(
    model: torch.nn.Module,
    batch: GpuBatch,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
):
    with _precision_context(device, amp_dtype, amp_enabled):
        return unwrap_model(model).optical_encoder(
            batch.opt_coords,
            batch.opt_t,
            batch.opt_v,
            batch.opt_ref_t,
            batch.opt_mask,
            errors_obs=batch.opt_err,
        )


def projection_retrieval_forward(
    model: torch.nn.Module,
    batch: GpuBatch,
    g: torch.Tensor,
    z_l: torch.Tensor,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
):
    with _precision_context(device, amp_dtype, amp_enabled):
        feat_g, feat_o = model.get_contrastive_embeddings(g, z_l)
        return similarity_with_time_compat(
            model,
            feat_g,
            feat_o,
            batch.gw_event_time_mjd,
            batch.opt_event_time_mjd,
        )


def fusion_classifier_forward(
    model: torch.nn.Module,
    batch: GpuBatch,
    g: torch.Tensor,
    z_l: torch.Tensor,
    h_l: torch.Tensor,
    h_gw: torch.Tensor,
    cred_level: Optional[torch.Tensor],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
):
    with _precision_context(device, amp_dtype, amp_enabled):
        return model.fusion_logits(
            g,
            h_l,
            z_l=z_l,
            H_gw=h_gw,
            cred_level=cred_level,
            gw_s=batch.gw_s,
            gw_m=batch.gw_m,
            opt_coords=batch.opt_coords,
            dt_days=batch.dt_days,
        )


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    batch_size: int,
    phase: str,
    precision: str,
    amp_dtype_name: str,
    warmup_iterations: int,
    measure_iterations: int,
    device: torch.device,
    checkpoint: str,
) -> Dict[str, Any]:
    torch.cuda.empty_cache()
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start_events: List[torch.cuda.Event] = []
    end_events: List[torch.cuda.Event] = []
    with torch.inference_mode():
        for _ in range(measure_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            start_events.append(start)
            end_events.append(end)
    torch.cuda.synchronize(device)

    elapsed_ms = [float(start.elapsed_time(end)) for start, end in zip(start_events, end_events)]
    stats = summarize_latency_ms(elapsed_ms, batch_size)
    peak_alloc = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / (1024.0 ** 2))
    return {
        "checkpoint": checkpoint,
        "batch_size": int(batch_size),
        "precision": precision,
        "amp_dtype": amp_dtype_name,
        "phase": phase,
        "status": "ok",
        "warmup_iterations": int(warmup_iterations),
        "measure_iterations": int(measure_iterations),
        **stats,
        "peak_allocated_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
        "error": "",
    }


def oom_result(
    *,
    checkpoint: str,
    batch_size: int,
    precision: str,
    amp_dtype_name: str,
    phase: str,
    warmup_iterations: int,
    measure_iterations: int,
    error: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "checkpoint": checkpoint,
        "batch_size": int(batch_size),
        "precision": precision,
        "amp_dtype": amp_dtype_name,
        "phase": phase,
        "status": "oom",
        "warmup_iterations": int(warmup_iterations),
        "measure_iterations": int(measure_iterations),
        "error": error,
    }
    for key in CSV_FIELDS:
        row.setdefault(key, None)
    return row


def error_result(
    *,
    checkpoint: str,
    batch_size: int,
    precision: str,
    amp_dtype_name: str,
    phase: str,
    warmup_iterations: int,
    measure_iterations: int,
    error: str,
) -> Dict[str, Any]:
    row = oom_result(
        checkpoint=checkpoint,
        batch_size=batch_size,
        precision=precision,
        amp_dtype_name=amp_dtype_name,
        phase=phase,
        warmup_iterations=warmup_iterations,
        measure_iterations=measure_iterations,
        error=error,
    )
    row["status"] = "error"
    return row


def _run_phase_safely(
    fn: Callable[[], Any],
    *,
    batch_size: int,
    phase: str,
    precision_info: Mapping[str, Any],
    cfg: Mapping[str, Any],
    checkpoint: str,
    device: torch.device,
) -> Dict[str, Any]:
    try:
        return benchmark_callable(
            fn,
            batch_size=batch_size,
            phase=phase,
            precision=str(precision_info["precision"]),
            amp_dtype_name=str(precision_info["amp_dtype_name"]),
            warmup_iterations=int(cfg["warmup_iterations"]),
            measure_iterations=int(cfg["measure_iterations"]),
            device=device,
            checkpoint=checkpoint,
        )
    except RuntimeError as exc:
        if is_cuda_oom(exc):
            torch.cuda.empty_cache()
            return oom_result(
                checkpoint=checkpoint,
                batch_size=batch_size,
                precision=str(precision_info["precision"]),
                amp_dtype_name=str(precision_info["amp_dtype_name"]),
                phase=phase,
                warmup_iterations=int(cfg["warmup_iterations"]),
                measure_iterations=int(cfg["measure_iterations"]),
                error=str(exc).splitlines()[0],
            )
        raise


def benchmark_case(
    model: torch.nn.Module,
    batch: GpuBatch,
    *,
    batch_size: int,
    precision_info: Mapping[str, Any],
    cfg: Mapping[str, Any],
    checkpoint: str,
    device: torch.device,
) -> List[Dict[str, Any]]:
    amp_dtype = precision_info["amp_dtype"]
    amp_enabled = bool(precision_info["amp_enabled"])

    rows: List[Dict[str, Any]] = []

    rows.append(
        _run_phase_safely(
            lambda: full_multitask_forward(
                model,
                batch,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            ),
            batch_size=batch_size,
            phase="full_multitask",
            precision_info=precision_info,
            cfg=cfg,
            checkpoint=checkpoint,
            device=device,
        )
    )

    rows.append(
        _run_phase_safely(
            lambda: gw_encoder_forward(
                model,
                batch,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            ),
            batch_size=batch_size,
            phase="gw_encoder",
            precision_info=precision_info,
            cfg=cfg,
            checkpoint=checkpoint,
            device=device,
        )
    )

    rows.append(
        _run_phase_safely(
            lambda: optical_encoder_forward(
                model,
                batch,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            ),
            batch_size=batch_size,
            phase="optical_encoder",
            precision_info=precision_info,
            cfg=cfg,
            checkpoint=checkpoint,
            device=device,
        )
    )

    try:
        with torch.inference_mode():
            g, z_l, h_l, h_gw = encode_model(
                model,
                batch,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            cred_level = compute_cred_level_if_needed(model, batch)
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        if is_cuda_oom(exc):
            torch.cuda.empty_cache()
            for phase in ("projection_retrieval", "fusion_classifier"):
                rows.append(
                    oom_result(
                        checkpoint=checkpoint,
                        batch_size=batch_size,
                        precision=str(precision_info["precision"]),
                        amp_dtype_name=str(precision_info["amp_dtype_name"]),
                        phase=phase,
                        warmup_iterations=int(cfg["warmup_iterations"]),
                        measure_iterations=int(cfg["measure_iterations"]),
                        error=f"precompute failed: {str(exc).splitlines()[0]}",
                    )
                )
            return rows
        raise

    rows.append(
        _run_phase_safely(
            lambda: projection_retrieval_forward(
                model,
                batch,
                g,
                z_l,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            ),
            batch_size=batch_size,
            phase="projection_retrieval",
            precision_info=precision_info,
            cfg=cfg,
            checkpoint=checkpoint,
            device=device,
        )
    )

    rows.append(
        _run_phase_safely(
            lambda: fusion_classifier_forward(
                model,
                batch,
                g,
                z_l,
                h_l,
                h_gw,
                cred_level,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            ),
            batch_size=batch_size,
            phase="fusion_classifier",
            precision_info=precision_info,
            cfg=cfg,
            checkpoint=checkpoint,
            device=device,
        )
    )

    return rows


def load_project_model(cfg: Mapping[str, Any], device: torch.device):
    from test_evaluate import load_model

    args = SimpleNamespace(
        checkpoint=cfg["checkpoint"],
        config=cfg.get("model_config"),
        test_data_path=cfg["data_path"],
        nonkn_cls_base_field=cfg.get("nonkn_cls_base_field"),
    )
    return load_model(args, device)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _safe_cudnn_version() -> Any:
    try:
        return torch.backends.cudnn.version()
    except Exception as exc:  # Environment metadata must not discard benchmark results.
        return {"error": str(exc).splitlines()[0]}


def collect_environment(device: torch.device) -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": _safe_cudnn_version(),
        "cuda_device": str(device),
        "gpu_name": props.name,
        "gpu_capability": [props.major, props.minor],
        "gpu_total_memory_mb": int(props.total_memory // (1024 ** 2)),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def build_output_paths(output_dir: str, checkpoint: str) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = Path(checkpoint).stem
    job_id = os.environ.get("SLURM_JOB_ID")
    suffix = job_id if job_id else time.strftime("%Y%m%d_%H%M%S")
    prefix = f"model_speed_{ckpt_stem}_{suffix}"
    return {
        "json": out_dir / f"{prefix}.json",
        "csv": out_dir / f"{prefix}.csv",
    }


def write_outputs(payload: Mapping[str, Any], rows: List[Mapping[str, Any]], output_paths: Mapping[str, Path]) -> None:
    with open(output_paths["json"], "w") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")

    with open(output_paths["csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_safe(row.get(field)) for field in CSV_FIELDS})


def print_summary(rows: List[Mapping[str, Any]]) -> None:
    print("\nBenchmark summary")
    print("=" * 88)
    print(f"{'B':>6} {'precision':>9} {'phase':>21} {'status':>8} {'median ms':>11} {'samples/s':>12}")
    for row in rows:
        median = row.get("median_ms")
        sps = row.get("samples_per_sec_median")
        median_text = "-" if median is None else f"{float(median):.3f}"
        sps_text = "-" if sps is None else f"{float(sps):.1f}"
        print(
            f"{int(row['batch_size']):6d} "
            f"{str(row['precision']):>9} "
            f"{str(row['phase']):>21} "
            f"{str(row['status']):>8} "
            f"{median_text:>11} "
            f"{sps_text:>12}"
        )
    print("=" * 88)


def run_benchmark(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    device = torch.device(str(cfg.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Model speed benchmark requires a CUDA device; CPU fallback is disabled.")

    set_reproducibility(int(cfg["seed"]))
    model, model_args, saved_args = load_project_model(cfg, device)
    model.eval()

    max_batch_size = max(int(v) for v in cfg["batch_sizes"])
    ref_start = float(model_args.get("ref_start", -0.3))
    ref_end = float(model_args.get("ref_end", 0.6))
    cpu_batch = read_cpu_batch(
        str(cfg["data_path"]),
        max_batch_size,
        ref_start=ref_start,
        ref_end=ref_end,
    )

    rows: List[Dict[str, Any]] = []
    for batch_size in cfg["batch_sizes"]:
        for precision_mode in cfg["precision_modes"]:
            precision_info = resolve_precision(str(precision_mode), device)
            print(
                f"Running batch_size={batch_size}, precision={precision_info['precision']} "
                f"({precision_info['amp_dtype_name']})"
            )
            try:
                batch = stage_gpu_batch(cpu_batch, int(batch_size), device, model_args)
            except RuntimeError as exc:
                if is_cuda_oom(exc):
                    torch.cuda.empty_cache()
                    for phase in (
                        "full_multitask",
                        "gw_encoder",
                        "optical_encoder",
                        "projection_retrieval",
                        "fusion_classifier",
                    ):
                        rows.append(
                            oom_result(
                                checkpoint=str(cfg["checkpoint"]),
                                batch_size=int(batch_size),
                                precision=str(precision_info["precision"]),
                                amp_dtype_name=str(precision_info["amp_dtype_name"]),
                                phase=phase,
                                warmup_iterations=int(cfg["warmup_iterations"]),
                                measure_iterations=int(cfg["measure_iterations"]),
                                error=f"input staging failed: {str(exc).splitlines()[0]}",
                            )
                        )
                    continue
                raise

            rows.extend(
                benchmark_case(
                    model,
                    batch,
                    batch_size=int(batch_size),
                    precision_info=precision_info,
                    cfg=cfg,
                    checkpoint=str(cfg["checkpoint"]),
                    device=device,
                )
            )
            del batch
            torch.cuda.empty_cache()

    payload = {
        "config": dict(cfg),
        "environment": collect_environment(device),
        "model": {
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "train_args_available": bool(saved_args),
            "model_args": model_args,
        },
        "input_shapes": cpu_batch.input_shapes,
        "results": rows,
    }
    output_paths = build_output_paths(str(cfg["output_dir"]), str(cfg["checkpoint"]))
    write_outputs(payload, rows, output_paths)
    print_summary(rows)
    print(f"\nWrote JSON: {output_paths['json']}")
    print(f"Wrote CSV:  {output_paths['csv']}")
    payload["output_paths"] = {k: str(v) for k, v in output_paths.items()}
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pure GPU forward speed for GW-optical ALBEF.")
    parser.add_argument("--config", required=True, help="Path to benchmark JSON config.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and file paths, then exit without loading model or requiring CUDA.",
    )
    return parser.parse_args()


def validate_file_paths(cfg: Mapping[str, Any]) -> None:
    for key in ("checkpoint", "data_path"):
        path = str(cfg[key])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key} not found: {path}")
    if cfg.get("model_config") and not os.path.exists(str(cfg["model_config"])):
        raise FileNotFoundError(f"model_config not found: {cfg['model_config']}")
    Path(str(cfg["output_dir"])).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    raw_cfg = load_json_config(args.config)
    cfg = normalize_config(raw_cfg, config_path=args.config)
    validate_file_paths(cfg)
    if args.validate_only:
        print("Benchmark config validation passed.")
        print(json.dumps(_json_safe(cfg), indent=2, ensure_ascii=False))
        return
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
