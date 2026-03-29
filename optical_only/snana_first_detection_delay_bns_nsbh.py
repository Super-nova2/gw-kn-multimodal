#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm.auto import tqdm

PROJECT_DIR = Path('<BASE_DIR>/gw-kn-multimodal/optical_only')
MODEL_DIR = Path('<BASE_DIR>/gw-kn-multimodal/Model')
DEFAULT_OUTPUT_BASE_DIR = PROJECT_DIR / 'outputs' / 'snana_first_detection_delay_bns_nsbh'
DEFAULT_CANONICAL_OFFSET_NPZ = Path('<BASE_DIR>/data/Optical_Only_dataset/delta_days_distribution.npz')
DEFAULT_BNS_SIM_ROOT = Path('<BASE_DIR>/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS_AUG')
DEFAULT_BNS_PREFIX = 'LSST_KN_BNS_AUG'
DEFAULT_NSBH_SIM_ROOT = Path('<BASE_DIR>/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_TRAIN')
DEFAULT_NSBH_PREFIX = 'LSST_KN_NSBH_TRAIN'
DEFAULT_SNR_THRESHOLD = 5.0
DEFAULT_FLUXCAL_ZP = 27.5
DEFAULT_PSFFLUX_ZP = 31.4

MERGE_HELPER_PATH = MODEL_DIR / 'lightcurve_merge.py'
_merge_spec = importlib.util.spec_from_file_location('lightcurve_merge', MERGE_HELPER_PATH)
if _merge_spec is None or _merge_spec.loader is None:
    raise ImportError(f'Unable to load merge helper from {MERGE_HELPER_PATH}')
_lightcurve_merge = importlib.util.module_from_spec(_merge_spec)
_merge_spec.loader.exec_module(_lightcurve_merge)
MERGE_WINDOW_HOURS = float(_lightcurve_merge.MERGE_WINDOW_HOURS)
merge_photometry_psfflux = _lightcurve_merge.merge_photometry_psfflux

MJD_EXPLODE_PATTERN = re.compile(r"MJD_EXPLODE:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Build merged SNR-based first-detection delay distribution and overwrite canonical optical-only offset NPZ.'
    )
    p.add_argument('--output-base-dir', type=Path, default=DEFAULT_OUTPUT_BASE_DIR)
    p.add_argument('--canonical-offset-npz', type=Path, default=DEFAULT_CANONICAL_OFFSET_NPZ)
    p.add_argument('--bns-sim-root', type=Path, default=DEFAULT_BNS_SIM_ROOT)
    p.add_argument('--bns-prefix', type=str, default=DEFAULT_BNS_PREFIX)
    p.add_argument('--nsbh-sim-root', type=Path, default=DEFAULT_NSBH_SIM_ROOT)
    p.add_argument('--nsbh-prefix', type=str, default=DEFAULT_NSBH_PREFIX)
    p.add_argument('--snr-threshold', type=float, default=DEFAULT_SNR_THRESHOLD)
    p.add_argument('--fluxcal-zp', type=float, default=DEFAULT_FLUXCAL_ZP)
    p.add_argument('--psfflux-zp', type=float, default=DEFAULT_PSFFLUX_ZP)
    p.add_argument('--num-workers', type=int, default=min(8, max(1, (os.cpu_count() or 1) // 2)))
    p.add_argument('--max-events-per-source', type=int, default=None)
    p.add_argument('--chunksize', type=int, default=16)
    p.add_argument('--save-detailed-deltas-csv', action='store_true')
    p.add_argument('--disable-parallel', action='store_true')
    return p.parse_args()


def parse_mjd_explode_from_readme(readme_path: Path) -> float:
    text = readme_path.read_text(encoding='utf-8', errors='ignore')
    match = MJD_EXPLODE_PATTERN.search(text)
    if match is None:
        raise ValueError(f'MJD_EXPLODE not found in README: {readme_path}')
    return float(match.group(1))


def build_detection_mask(flux: np.ndarray, fluxerr: np.ndarray, snr_threshold: float) -> np.ndarray:
    if flux.size == 0:
        return np.zeros((0,), dtype=bool)
    valid = np.isfinite(flux) & np.isfinite(fluxerr) & (fluxerr > 0)
    if not np.any(valid):
        return np.zeros(flux.shape, dtype=bool)
    snr = np.full(flux.shape, -np.inf, dtype=np.float64)
    snr[valid] = flux[valid] / fluxerr[valid]
    return np.asarray(snr > float(snr_threshold), dtype=bool)


def list_event_dirs(base_dir: Path, prefix: str) -> List[Path]:
    return sorted([p for p in base_dir.glob(f'{prefix}_*') if p.is_dir()])


def process_event_dir(
    event_dir: Path,
    source: str,
    snr_threshold: float,
    fluxcal_to_psfflux_factor: float,
) -> Dict:
    prefix = event_dir.name
    readme_path = event_dir / f'{prefix}.README'
    head_path = event_dir / f'{prefix}_HEAD.FITS'
    phot_path = event_dir / f'{prefix}_PHOT.FITS'

    event_id = -1
    try:
        event_id = int(prefix.rsplit('_', 1)[-1])
    except Exception:
        pass

    result = {
        'source': source,
        'event_dir': prefix,
        'event_id': event_id,
        'n_realizations': 0,
        'n_detected': 0,
        'n_no_detect': 0,
        'delta_days': np.empty((0,), dtype=np.float32),
        'error': None,
    }

    if not (readme_path.exists() and head_path.exists() and phot_path.exists()):
        result['error'] = 'missing_required_files'
        return result

    try:
        mjd_explode = parse_mjd_explode_from_readme(readme_path)
        with fits.open(head_path, memmap=False) as hdul_head, fits.open(phot_path, memmap=False) as hdul_phot:
            head = hdul_head[1].data
            phot = hdul_phot[1].data
            ptrobs_min = np.asarray(head['PTROBS_MIN'], dtype=np.int64)
            ptrobs_max = np.asarray(head['PTROBS_MAX'], dtype=np.int64)
            mjd_all = np.asarray(phot['MJD'], dtype=np.float64)
            flux_all = np.asarray(phot['FLUXCAL'], dtype=np.float64)
            fluxerr_all = np.asarray(phot['FLUXCALERR'], dtype=np.float64)
            flt_all = np.asarray(phot['BAND'])

        n_realizations = int(len(ptrobs_min))
        if n_realizations == 0:
            result['n_realizations'] = 0
            result['n_no_detect'] = 0
            return result

        deltas = np.empty((n_realizations,), dtype=np.float32)
        detected_mask = np.zeros((n_realizations,), dtype=bool)

        for i, (start_1based, end_1based) in enumerate(zip(ptrobs_min, ptrobs_max)):
            start = int(start_1based) - 1
            end = int(end_1based)
            if start < 0 or end <= start or end > mjd_all.size:
                continue

            local_mjd = mjd_all[start:end]
            local_flux = flux_all[start:end]
            local_fluxerr = fluxerr_all[start:end]
            local_flt = flt_all[start:end]

            merged_mjd, merged_psfflux, merged_psffluxerr, _ = merge_photometry_psfflux(
                mjd=local_mjd,
                fluxcal=local_flux,
                fluxcalerr=local_fluxerr,
                flt=local_flt,
                fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
            )
            if merged_mjd.size == 0:
                continue

            local_detect = build_detection_mask(merged_psfflux, merged_psffluxerr, snr_threshold=snr_threshold)
            if np.any(local_detect):
                first_local_idx = int(np.argmax(local_detect))
                first_detect_mjd = float(merged_mjd[first_local_idx])
                deltas[i] = np.float32(first_detect_mjd - mjd_explode)
                detected_mask[i] = True

        detected_deltas = deltas[detected_mask]
        result['n_realizations'] = n_realizations
        result['n_detected'] = int(detected_deltas.size)
        result['n_no_detect'] = n_realizations - result['n_detected']
        result['delta_days'] = detected_deltas
        return result
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        return result


def process_event_dir_worker(args: Tuple[str, str, float, float]) -> Dict:
    event_dir_str, source, snr_threshold, fluxcal_to_psfflux_factor = args
    return process_event_dir(
        Path(event_dir_str),
        source=source,
        snr_threshold=float(snr_threshold),
        fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
    )


def scan_source(
    source: str,
    base_dir: Path,
    prefix: str,
    snr_threshold: float,
    fluxcal_to_psfflux_factor: float,
    max_events: int | None,
    use_parallel: bool,
    n_workers: int,
    chunksize: int,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    event_dirs = list_event_dirs(base_dir, prefix)
    if max_events is not None:
        event_dirs = event_dirs[: int(max_events)]
    print(f'[{source}] total event dirs selected: {len(event_dirs)}')

    delta_chunks: List[np.ndarray] = []
    event_rows: List[Dict] = []
    error_rows: List[Dict] = []

    def append_result(res: Dict) -> None:
        event_rows.append({
            'source': res['source'],
            'event_dir': res['event_dir'],
            'event_id': res['event_id'],
            'n_realizations': res['n_realizations'],
            'n_detected': res['n_detected'],
            'n_no_detect': res['n_no_detect'],
        })
        if res['delta_days'].size > 0:
            delta_chunks.append(res['delta_days'])
        if res['error'] is not None:
            error_rows.append({
                'source': res['source'],
                'event_dir': res['event_dir'],
                'event_id': res['event_id'],
                'error': res['error'],
            })

    ran_in_parallel = False
    if use_parallel and n_workers > 1 and len(event_dirs) > 0:
        args_iter = [
            (str(p), source, float(snr_threshold), float(fluxcal_to_psfflux_factor))
            for p in event_dirs
        ]
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                for res in tqdm(
                    ex.map(process_event_dir_worker, args_iter, chunksize=max(1, int(chunksize))),
                    total=len(args_iter),
                    desc=f'{source} scan (parallel)',
                ):
                    append_result(res)
            ran_in_parallel = True
        except Exception as exc:
            print(f'[{source}] parallel mode failed ({type(exc).__name__}: {exc}); fallback to sequential mode.')
            delta_chunks.clear()
            event_rows.clear()
            error_rows.clear()

    if not ran_in_parallel:
        for event_dir in tqdm(event_dirs, desc=f'{source} scan (sequential)'):
            append_result(
                process_event_dir(
                    event_dir,
                    source=source,
                    snr_threshold=snr_threshold,
                    fluxcal_to_psfflux_factor=fluxcal_to_psfflux_factor,
                )
            )

    delta_days = np.concatenate(delta_chunks).astype(np.float32) if delta_chunks else np.empty((0,), dtype=np.float32)
    event_df = pd.DataFrame(event_rows)
    error_df = pd.DataFrame(error_rows)
    return delta_days, event_df, error_df


def summarize_distribution(delta_days: np.ndarray, n_total_realizations: int, n_detected: int, source: str) -> Dict:
    row = {
        'source': source,
        'n_total_realizations': int(n_total_realizations),
        'n_detected': int(n_detected),
        'n_no_detect': int(max(0, n_total_realizations - n_detected)),
        'detect_rate': float(n_detected / n_total_realizations) if n_total_realizations > 0 else 0.0,
    }
    if delta_days.size == 0:
        for key in ['mean_days', 'std_days', 'q01_days', 'q05_days', 'q25_days', 'q50_days', 'q75_days', 'q95_days', 'q99_days', 'min_days', 'max_days']:
            row[key] = np.nan
        return row

    q = np.quantile(delta_days, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    row.update({
        'mean_days': float(np.mean(delta_days)),
        'std_days': float(np.std(delta_days)),
        'q01_days': float(q[0]),
        'q05_days': float(q[1]),
        'q25_days': float(q[2]),
        'q50_days': float(q[3]),
        'q75_days': float(q[4]),
        'q95_days': float(q[5]),
        'q99_days': float(q[6]),
        'min_days': float(np.min(delta_days)),
        'max_days': float(np.max(delta_days)),
    })
    return row


def save_outputs(
    output_base_dir: Path,
    canonical_offset_npz: Path,
    bns_deltas: np.ndarray,
    nsbh_deltas: np.ndarray,
    combined_deltas: np.ndarray,
    event_summary_df: pd.DataFrame,
    error_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    snr_threshold: float,
    fluxcal_to_psfflux_factor: float,
    save_detailed_deltas_csv: bool,
) -> Dict[str, Path]:
    output_base_dir.mkdir(parents=True, exist_ok=True)
    run_tag = pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')
    run_dir = output_base_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = run_dir / 'summary_stats.csv'
    event_csv = run_dir / 'event_level_counts.csv'
    error_csv = run_dir / 'event_errors.csv'
    npz_path = run_dir / 'delta_days_distribution.npz'
    hist_csv = run_dir / 'histogram_counts.csv'
    plot_path = run_dir / 'first_detection_delay_histogram.png'

    summary_df.to_csv(summary_csv, index=False)
    event_summary_df.to_csv(event_csv, index=False)
    error_df.to_csv(error_csv, index=False)

    payload = {
        'delta_days_bns': bns_deltas,
        'delta_days_nsbh': nsbh_deltas,
        'delta_days_combined': combined_deltas,
        'snr_detection_threshold': np.float32(snr_threshold),
        'merge_window_hours': np.float32(MERGE_WINDOW_HOURS),
        'fluxcal_to_psfflux_factor': np.float32(fluxcal_to_psfflux_factor),
    }
    np.savez_compressed(npz_path, **payload)
    canonical_offset_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(canonical_offset_npz, **payload)

    if save_detailed_deltas_csv:
        rows = []
        if bns_deltas.size > 0:
            rows.append(pd.DataFrame({'source': 'BNS', 'delta_days': bns_deltas}))
        if nsbh_deltas.size > 0:
            rows.append(pd.DataFrame({'source': 'NSBH', 'delta_days': nsbh_deltas}))
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(run_dir / 'detected_delta_days_detailed.csv', index=False)

    if combined_deltas.size > 0:
        q_lo = float(np.quantile(combined_deltas, 0.001))
        q_hi = float(np.quantile(combined_deltas, 0.999))
        lo = np.floor(min(q_lo, 0.0))
        hi = np.ceil(max(q_hi, 10.0))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 300)
        hist_bns, edges = np.histogram(bns_deltas, bins=bins) if bns_deltas.size > 0 else (np.zeros(len(bins) - 1, dtype=int), bins)
        hist_nsbh, _ = np.histogram(nsbh_deltas, bins=bins) if nsbh_deltas.size > 0 else (np.zeros(len(bins) - 1, dtype=int), bins)
        hist_combined, _ = np.histogram(combined_deltas, bins=bins)
        pd.DataFrame({
            'bin_left': edges[:-1],
            'bin_right': edges[1:],
            'count_bns': hist_bns,
            'count_nsbh': hist_nsbh,
            'count_combined': hist_combined,
        }).to_csv(hist_csv, index=False)

        plt.figure(figsize=(10, 5))
        plt.step(edges[:-1], hist_combined, where='post', label='Combined', linewidth=2)
        if bns_deltas.size > 0:
            plt.step(edges[:-1], hist_bns, where='post', label='BNS', alpha=0.9)
        if nsbh_deltas.size > 0:
            plt.step(edges[:-1], hist_nsbh, where='post', label='NSBH', alpha=0.9)
        plt.axvline(0.0, color='k', linestyle='--', linewidth=1, alpha=0.7)
        plt.xlabel('Merged-SNR first detection delay (days) = first_detect_mjd - MJD_EXPLODE')
        plt.ylabel('Count')
        plt.title('SNANA first-detection delay distribution (merged same-band 2h + psfFlux SNR > 5)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

    (output_base_dir / 'latest_run.txt').write_text(str(run_dir), encoding='utf-8')
    return {
        'run_dir': run_dir,
        'summary_csv': summary_csv,
        'event_csv': event_csv,
        'error_csv': error_csv,
        'npz_path': npz_path,
        'canonical_npz_path': canonical_offset_npz,
        'hist_csv': hist_csv,
        'plot_path': plot_path,
    }


def main() -> None:
    args = parse_args()
    fluxcal_to_psfflux_factor = 10.0 ** (0.4 * (float(args.psfflux_zp) - float(args.fluxcal_zp)))
    sim_sources = [
        {'source': 'BNS', 'base_dir': args.bns_sim_root, 'prefix': args.bns_prefix},
        {'source': 'NSBH', 'base_dir': args.nsbh_sim_root, 'prefix': args.nsbh_prefix},
    ]
    use_parallel = (not args.disable_parallel) and int(args.num_workers) > 1

    print('SNR_THRESHOLD =', float(args.snr_threshold))
    print('MERGE_WINDOW_HOURS =', MERGE_WINDOW_HOURS)
    print('FLUXCAL_TO_PSFFLUX_FACTOR =', fluxcal_to_psfflux_factor)
    print('USE_PARALLEL =', use_parallel, 'N_WORKERS =', int(args.num_workers))
    print('CANONICAL_OFFSET_NPZ =', args.canonical_offset_npz)

    all_deltas: Dict[str, np.ndarray] = {}
    all_event_df: List[pd.DataFrame] = []
    all_error_df: List[pd.DataFrame] = []

    for cfg in sim_sources:
        source = cfg['source']
        delta_days, event_df, error_df = scan_source(
            source=source,
            base_dir=cfg['base_dir'],
            prefix=cfg['prefix'],
            snr_threshold=float(args.snr_threshold),
            fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
            max_events=args.max_events_per_source,
            use_parallel=use_parallel,
            n_workers=int(args.num_workers),
            chunksize=int(args.chunksize),
        )
        all_deltas[source] = delta_days
        all_event_df.append(event_df)
        if not error_df.empty:
            all_error_df.append(error_df)

    bns_deltas = all_deltas.get('BNS', np.empty((0,), dtype=np.float32))
    nsbh_deltas = all_deltas.get('NSBH', np.empty((0,), dtype=np.float32))
    combined_deltas = np.concatenate([arr for arr in [bns_deltas, nsbh_deltas] if arr.size > 0]).astype(np.float32) if (bns_deltas.size + nsbh_deltas.size) > 0 else np.empty((0,), dtype=np.float32)

    event_summary_df = pd.concat(all_event_df, ignore_index=True) if all_event_df else pd.DataFrame()
    error_df = pd.concat(all_error_df, ignore_index=True) if all_error_df else pd.DataFrame(columns=['source', 'event_dir', 'event_id', 'error'])

    total_bns = int(event_summary_df.loc[event_summary_df['source'] == 'BNS', 'n_realizations'].sum()) if not event_summary_df.empty else 0
    total_nsbh = int(event_summary_df.loc[event_summary_df['source'] == 'NSBH', 'n_realizations'].sum()) if not event_summary_df.empty else 0
    summary_rows = [
        summarize_distribution(bns_deltas, total_bns, int(bns_deltas.size), 'BNS'),
        summarize_distribution(nsbh_deltas, total_nsbh, int(nsbh_deltas.size), 'NSBH'),
        summarize_distribution(combined_deltas, total_bns + total_nsbh, int(combined_deltas.size), 'Combined'),
    ]
    summary_df = pd.DataFrame(summary_rows)

    paths = save_outputs(
        output_base_dir=args.output_base_dir,
        canonical_offset_npz=args.canonical_offset_npz,
        bns_deltas=bns_deltas,
        nsbh_deltas=nsbh_deltas,
        combined_deltas=combined_deltas,
        event_summary_df=event_summary_df,
        error_df=error_df,
        summary_df=summary_df,
        snr_threshold=float(args.snr_threshold),
        fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
        save_detailed_deltas_csv=bool(args.save_detailed_deltas_csv),
    )

    print('BNS detected realizations:', bns_deltas.size)
    print('NSBH detected realizations:', nsbh_deltas.size)
    print('Combined detected realizations:', combined_deltas.size)
    print('Events with processing errors:', len(error_df))
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == '__main__':
    main()
