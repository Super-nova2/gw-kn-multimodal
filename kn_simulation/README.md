# GWSamplegen to Rubin/SNANA production pipeline

This is the maintained optical-simulation entry point. The repository's
`dataset/` directory is historical and is not imported by this pipeline.

The supported data flow is:

```text
GWSamplegen catalog.csv + <simulation_id>.fits skymaps
    -> validated kn_catalog.csv
    -> Rubin baseline/ToO observation plans
    -> per-event SNANA simulations
    -> shard sidecars and consolidated status summary
```

## Layout

- `bin/kn-sim`: the only user-facing command;
- `src/`: internal Python implementation;
- `profiles/`: tracked BNS/NSBH train/test production profiles;
- `config/`: the shared Rubin ToO strategy;
- `templates/`: current BNS and NSBH SNANA templates;
- `slurm/worker.sh`: thin Slurm worker/finalizer launcher;
- `runs/`: ignored runtime inputs, manifests, audit products, and statuses.

Do not execute files under `src/` directly. Do not place legacy
`injections_*.csv`, `coincs.dat`, or `allsky.csv` files in `runs/`.

## Prerequisites

The Rubin-time GWSamplegen run must be complete. Copy its FITS maps to the
profile's configured skymap directory before preparing the catalog. The four
configured directories are:

| Profile | Skymap directory |
| --- | --- |
| `bns_train` | `$BASE_DIR/data/skymap/bns_skymap_train` |
| `bns_test` | `$BASE_DIR/data/skymap/bns_skymap_test` |
| `nsbh_train` | `$BASE_DIR/data/skymap/nsbh_skymap_train` |
| `nsbh_test` | `$BASE_DIR/data/skymap/nsbh_skymap_test` |

`BASE_DIR` defaults to `/fred/oz016/bgao_kn`. Profiles also require the Rubin
OpSim database, SNANA installation, SNDATA_ROOT models, and `sbatch`.

## Commands

Prepare a run without submitting compute work:

```bash
kn_simulation/bin/kn-sim prepare bns_train \
    --catalog /path/to/GWSamplegen/output/catalog.csv
```

This validates the complete catalog and every expected skymap before writing:

```text
kn_simulation/runs/bns_train/catalog.csv
kn_simulation/runs/bns_train/catalog.input.json
kn_simulation/runs/bns_train/kn_catalog.csv
kn_simulation/runs/bns_train/kn_catalog.manifest.json
```

Existing prepared files are protected. Use `--overwrite-prepared` only when an
intentional replacement is required.

Preview or submit the Slurm jobs:

```bash
kn_simulation/bin/kn-sim submit bns_train --dry-run
kn_simulation/bin/kn-sim submit bns_train
```

Prepare and submit in one command:

```bash
kn_simulation/bin/kn-sim run bns_train \
    --catalog /path/to/GWSamplegen/output/catalog.csv
```

Inspect consolidated event state:

```bash
kn_simulation/bin/kn-sim status bns_train
```

After a partial run, resubmit only failed or unprocessed events. Successful and
permanently footprint-uncovered events are not repeated:

```bash
kn_simulation/bin/kn-sim submit bns_train --resume
```

Resource overrides do not require editing a profile:

```bash
kn_simulation/bin/kn-sim submit bns_train \
    --batch-size 20 --max-concurrency 20
```

## Catalog contract

Only the complete GWSamplegen `catalog.csv` schema is accepted. In particular,
`network_snr` must be present, finite, and non-negative. Truth-level
`mej_dynamic` and `mej_wind` must both be positive and consistent with
`mej_total`. Recovered parameters are preserved but are not used to recompute
ejecta.

All input rows are retained. There is no SNR or localization-area catalog cut.
GPS times must map into MJD 61000--64500 and the OpSim coverage. The prepared
catalog records deterministic event seeds, degree coordinates, viewing angle,
original ejecta, SNANA-grid ejecta, and explicit clipping flags.

## Rubin routing and coordinate samples

- `network_snr <= 12`: baseline WFD/DDF observations;
- `network_snr > 12`: Gold, Silver, or baseline according to the 90% credible
  area recomputed from the event skymap;
- zero baseline footprint coverage: recorded as
  `rubin_footprint_not_covered`, not as a GW detection rejection.

Train profiles use 1000 `posterior_3d` coordinate samples. Test profiles use 64
`posterior_test` samples, including one truth position. All profiles use
Planck15, sampling nside 256, seed 42, and the shared Rubin ToO configuration.

## Slurm products

Array workers write disjoint event products and immutable shard sidecars. They
never append concurrently to shared status files. An `afterany` finalizer
merges the latest status for every event into:

```text
status/success_sim_ids.txt
status/failed_sim_ids.txt
status/skipped_sim_ids.txt
status/summary.json
```

Temporary SIMLIB and SNANA input files are removed after each event. Observation
plans, coordinate manifests, status sidecars, submission metadata, and SNANA
FITS outputs are retained for audit and resume.
