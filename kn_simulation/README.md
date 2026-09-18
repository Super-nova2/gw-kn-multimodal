# GWSamplegen to Rubin/SNANA production pipeline

[Project README](../README.md) | [中文说明](../README_zh.md) | [Environment](../ENVIRONMENT.md) | [Runtime products](runs/README.md)

This is the maintained production optical-simulation entry point. The historical
`dataset/` directory has been removed. Local backups under `$BASE_DIR/backups/`
are not distributed with the repository. The separate fixed-physics
[GW170817A scenario experiment](gw170817a/README.md) has its own wrapper.

The supported data flow is:

```text
GWSamplegen pos_catalog.csv + neg_catalog.csv + separate skymap roots
    -> validated positive-only kn_catalog.csv + retained neg_catalog.csv
    -> Rubin baseline/ToO observation plans
    -> per-event SNANA simulations in node-local JOBFS
    -> normalized raw SNANA optical rows in one HDF5 shard per Slurm array task
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

Run commands from the repository root with the Python environment activated.
Set `BASE_DIR` to the external data workspace; the profile loader derives
`REPO_ROOT` from the checkout and expands both variables in YAML. The four
tracked production profiles are `bns_train`, `nsbh_train`, `bns_test` and
`nsbh_test`; no `test_aug` profile is distributed.

The matching GWSamplegen dual bundle must be complete. Production profiles read positive and type-1 maps from
`data/skymap/positive/<source>_skymap_<split>` and
`data/skymap/negative/<source>_skymap_<split>`. Canonical catalogs and bundle
provenance live under
`GWSamplegen/outputs/production_am_bayestar/dual/<source>_<split>_seed_1234/`. `BASE_DIR` defaults to
`/fred/oz016/bgao_kn`. Profiles also require the Rubin OpSim database, SNANA
installation, SNDATA_ROOT models, `opsimsummaryv2`, and `sbatch`. See the
[environment guide](../ENVIRONMENT.md#english-version) for dependencies outside
the Python lock file. A GitHub checkout does not contain the catalogs, skymaps,
OpSim database or SNANA models.

## Commands

Prepare a run without submitting compute work:

```bash
kn_simulation/bin/kn-sim prepare bns_train \
    --pos-catalog /path/to/bns_train_seed_1234/pos_catalog.csv \
    --neg-catalog /path/to/bns_train_seed_1234/neg_catalog.csv
```

This validates the complete catalog and every expected skymap before writing:

```text
kn_simulation/runs/bns_train/catalog.csv
kn_simulation/runs/bns_train/catalog.input.json
kn_simulation/runs/bns_train/kn_catalog.csv
kn_simulation/runs/bns_train/kn_catalog.manifest.json
kn_simulation/runs/bns_train/neg_catalog.csv
kn_simulation/runs/bns_train/dual_catalog.manifest.json
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
    --pos-catalog /path/to/bns_train_seed_1234/pos_catalog.csv \
    --neg-catalog /path/to/bns_train_seed_1234/neg_catalog.csv
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

`compact` refuses runs with missing task shards or unprocessed events. Failed
events no longer block compaction: they are retained in the aggregate with
their status/reason and any generated coordinates, while successful and
skipped events keep complete coordinate samples. It uses the same validation
and cleanup path as the automatic finalizer.

After a partial run, resubmit only failed or unprocessed events. Successful and
permanently footprint-uncovered events are not repeated:

```bash
kn_simulation/bin/kn-sim submit bns_train --resume
```

Production workers request 5 GiB of node-local temporary storage and require
`SLURM_TMPDIR` or `JOBFS`. SIMLIB, SNANA input, HEAD/PHOT/README products, and
the task's staging HDF5 live there. After each event is archived, its SNANA
directory is removed; the worker's exit trap removes the unique task scratch
directory. `SNDATA_ROOT` on Lustre is still used read-only for SNANA models and
calibration data.

Resource overrides do not require editing a profile:

```bash
kn_simulation/bin/kn-sim submit bns_train \
    --batch-size 20 --max-concurrency 20
```

## Catalog contract

Both streams must use the complete GWSamplegen catalog schema. The production
interface takes `--pos-catalog` and `--neg-catalog` together; the retained
`--catalog` option is a legacy single-positive input, not a substitute for the
dual bundle. In particular,
`network_snr` must be present, finite, and non-negative. Truth-level
`mej_dynamic` and `mej_wind` must be finite and consistent with `mej_total`.
Recovered parameters are preserved but are not used to recompute ejecta.

Dual preparation requires canonical `sample_class` and
`event_uid=<source>_<split>_<pos|neg>_<simulation_id>` fields. The copied
`catalog.csv` and prepared `kn_catalog.csv` contain only the positive stream and
are the only events submitted to SNANA. The separately validated
`neg_catalog.csv` retains every physical double-zero type-1 event and its own
skymap; it is never treated as a failed optical simulation. Positive events are
physically valid when `mej_dynamic + mej_wind > 0`, although only events whose
original component values lie inside both closed SIMSED ranges can enter SNANA:

| Source | mej_dynamic | mej_wind |
| --- | --- | --- |
| BNS | [0.001, 0.02] | [0.01, 0.13] |
| NSBH | [0.01, 0.09] | [0.01, 0.09] |

Boundary values are retained. Out-of-range events are omitted and counted in
the manifest; ejecta values are never clipped, rounded, or replaced. There is
no SNR or localization-area catalog cut. GPS times for retained events must map
into MJD 61000--64500 and the OpSim coverage.

`viewing_costheta` is computed as `abs(cos(theta_jn))`, so validated
`theta_jn` values automatically map into the model range [0, 1]. BNS
`phi_deg` is deterministically sampled from a uniform [15, 75) degree
distribution using the event ID and catalog seed; NSBH uses the model's fixed
30-degree value. No additional catalog filtering is needed for either angular
parameter.

The current SNANA templates retain `GENSIGMA_COSTHETA = 0.01` and, for BNS,
`GENSIGMA_PHI = 1` degree. These produce internal SNANA scatter around the
catalog peaks while the configured `GENRANGE` remains enforced.

Prepared-catalog schema v3 validates the double-zero `neg_type` label and
records type-1 exclusions in the manifest while retaining v2's filtering
instead of clipping. Runs prepared with an earlier schema must be regenerated with
`--overwrite-prepared` before submission or resume.

## Rubin routing and coordinate samples

- `network_snr <= 12`: baseline WFD/DDF observations;
- `network_snr > 12`: Gold, Silver, or baseline according to the 90% credible
  area recomputed from the event skymap;
- zero baseline footprint coverage: recorded as
  `rubin_footprint_not_covered`, not as a GW detection rejection.

Train profiles use 1000 `posterior_3d` coordinate samples. Test profiles use 64
`posterior_test` samples, including one truth position. All profiles use
Planck15, sampling nside 256, seed 1234, and the shared Rubin ToO configuration.

## Slurm products

Array workers keep each event's coordinate samples and observation plan in
memory while SNANA runs. Each array task then atomically writes one file under
`artifact_shards/<submission_id>_<task_index>.h5`, followed by its immutable
JSON status sidecar. Workers never append concurrently to a shared HDF5 or
status file.

After the array completes, the `afterany` finalizer selects the artifact
referenced by each event's latest status sidecar and creates:

```text
runs/<profile>/simulation_intermediates.h5
```

The aggregate is ordered exactly like the positive-only `kn_catalog.csv`. It contains event
status/reason, coordinate and optical offsets/counts, the complete observation
plan as JSON, task provenance, normalized raw SNANA HEAD/PHOT columns, and the
coordinate columns. Failed events are included
but do not block aggregation; the usable event list is written to
`runs/<profile>/success_sim_ids.txt`.

```text
simulation_id, sample_index, ra, dec, distance_mpc,
posterior_probability, is_true_position, redshift, libid,
in_baseline_footprint, too_tile_index, too_nobs, too_mode
```

Coordinate and raw optical datasets are chunked and gzip-compressed. The raw
optical schema stores realization-level RA/Dec, NOBS, first-detection time and
observation offsets plus observation-level MJD, FLUXCAL, FLUXCALERR, BAND and
PHOTFLAG. Each event has a deterministic SHA256 payload checksum. Root
attributes record the schema version, profile/source/split, prepared-catalog
SHA256, creation time, and row counts.

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

Production workers keep coordinate samples in memory and never create
per-event `work/COORDINATES/*.csv` files. Observation plans, coordinate samples,
and successful optical outputs remain in the aggregate HDF5; status sidecars
and submission metadata remain separate for audit and resume. Direct standalone
use of `src/snana.py` retains legacy coordinate CSV output unless
`--no-coordinate-files` is supplied.

## Legacy optical migration and safe pruning

Old successful events cannot be compacted into schema v2 until their existing
HEAD/PHOT/README products are archived. Migrate each profile before pruning:

```bash
kn_simulation/bin/kn-sim migrate-optical bns_train
kn_simulation/bin/kn-sim validate-optical bns_train
kn_simulation/bin/kn-sim prune-snana bns_train
```

Migration is resumable and defaults to 200 events per shard to limit new inode
usage. It preserves the latest success/failed/skipped status, verifies every
successful payload checksum, and compacts terminal profiles into
`simulation_intermediates.h5`. `prune-snana` is a report-only dry run by default
and refuses to proceed without a current migration manifest and full optical
coverage. Review its event/file/byte counts, then explicitly remove only the
matching per-event directories:

```bash
kn_simulation/bin/kn-sim prune-snana bns_train --execute
```

Run this sequence independently for `bns_train`, `nsbh_train`, `bns_test`, and
`nsbh_test`. The combined BNS+NSBH dataset submit script now uses the four
profile aggregates by default; legacy `BNS_SIM_ROOT`/`NSBH_SIM_ROOT` remains an
explicit fallback when the artifact variables are empty.

## Build the Model Dataset

After the BNS and NSBH aggregates for the selected split are ready, run from
the repository root:

```bash
PROFILE=final_train DATASET_MODE=train \
    bash Model/scripts/data/submit_create_dataset_bns_nsbh.sh
PROFILE=astro_test DATASET_MODE=test \
    bash Model/scripts/data/submit_create_dataset_bns_nsbh.sh
```

These submit jobs, so prepare the matching inputs before each command. Default
outputs under `$BASE_DIR/data/ALBEF_dataset/` are `combined_dataset_train.h5`
and `combined_dataset_astro_test.h5`. `BNS_SIM_ARTIFACT` and
`NSBH_SIM_ARTIFACT` select alternative aggregates; the dataset builder also
requires the separate negative catalogs and negative skymap roots.

The aggregate archives optical simulation state; the final model HDF5 adds
GW parents and labels. Physical double-zero ejecta events become type-1 GW
negatives. Events with ejecta but no usable optical counterpart can supply
type-2 GW negatives. Optical rows reference positive parents only. External
non-KN optical distractors are another input, not the retained negative GW
catalog. See the [MAGIKS guide](../Model/args/README.md) for training inputs.
