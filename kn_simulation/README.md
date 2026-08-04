# GWSamplegen to Rubin/SNANA production pipeline

This is the maintained optical-simulation entry point. The repository's
`dataset/` directory is historical and is not imported by this pipeline.

The supported data flow is:

```text
GWSamplegen catalog.csv + <simulation_id>.fits skymaps
    -> validated kn_catalog.csv
    -> Rubin baseline/ToO observation plans
    -> per-event SNANA simulations
    -> one intermediate HDF5 shard per Slurm array task
    -> simulation_intermediates.h5 and consolidated status summary
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

If a run reached a terminal state but its automatic finalizer did not complete,
compact it manually:

```bash
kn_simulation/bin/kn-sim compact bns_train
```

`compact` refuses incomplete runs. It uses the same validation and cleanup
path as the automatic finalizer.

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

Array workers keep each event's coordinate samples and observation plan in
memory while SNANA runs. Each array task then atomically writes one file under
`artifact_shards/<submission_id>_<task_index>.h5`, followed by its immutable
JSON status sidecar. Workers never append concurrently to a shared HDF5 or
status file.

For a fully terminal run, the `afterany` finalizer selects the artifact
referenced by each event's latest status sidecar and creates:

```text
runs/<profile>/simulation_intermediates.h5
```

The aggregate is ordered exactly like `kn_catalog.csv`. It contains event
status/reason, coordinate offsets and counts, the complete observation plan as
JSON, task provenance, and the coordinate columns:

```text
simulation_id, sample_index, ra, dec, distance_mpc,
posterior_probability, is_true_position, redshift, libid,
in_baseline_footprint, too_tile_index, too_nobs, too_mode
```

Coordinate datasets are chunked and gzip-compressed. Root attributes record the
schema version, profile/source/split, prepared-catalog SHA256, creation time,
event count, and coordinate count.

Before publishing the aggregate, the finalizer reopens the temporary file and
validates the catalog checksum, event IDs/order, JSON plans, coordinate offsets,
and per-event sample counts. Only after that validation succeeds does it
atomically install the aggregate and remove task HDF5 shards plus legacy
per-event `coordinate_samples/*.csv` and `observation_plans/*.json` files.
Validation failure leaves all source artifacts in place.

The existing lightweight status sidecars are retained, and the finalizer writes:

```text
status/success_sim_ids.txt
status/failed_sim_ids.txt
status/skipped_sim_ids.txt
status/summary.json
```

Temporary SIMLIB and SNANA input files are removed after each event. Observation
plans and coordinate samples remain available in the aggregate HDF5; status
sidecars, submission metadata, and SNANA FITS outputs remain separate for audit
and resume.

Legacy compatibility is automatic: a successful event from an older run that
has only `coordinate_samples/<id>.csv` and `observation_plans/<id>.json` is
imported into the final aggregate without rerunning SNANA. This is how an
already-successful probe event can be combined with later task shards.
