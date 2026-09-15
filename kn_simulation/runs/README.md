# Runtime Directories

[Production pipeline](../README.md) | [Project README](../../README_en.md)

Only this guide is tracked here. Each production profile creates an ignored
run directory: `bns_train`, `nsbh_train`, `bns_test` or `nsbh_test`.
Runtime catalogs, HDF5 files and job state are generated locally and are not
available from a fresh GitHub checkout.

## Products

| Product within `runs/<profile>/` | Purpose |
| --- | --- |
| `catalog.csv`, `catalog.input.json` | Copied positive input and its provenance |
| `kn_catalog.csv`, `kn_catalog.manifest.json` | Validated, filtered positive SNANA catalog and audit |
| `neg_catalog.csv`, `dual_catalog.manifest.json` | Retained physical type-1 negatives and dual-bundle provenance |
| `submission.json`, `status/submissions/` | Latest submission plus per-submission IDs and metadata |
| `artifact_shards/` | Per-task HDF5 before successful compaction |
| `status/` | Event status sidecars, summary and success/failed/skipped ID lists |
| `simulation_intermediates.h5` | Validated aggregate containing event state, coordinates, plans and raw optical rows |
| `success_sim_ids.txt` | Usable event list for downstream dataset construction |

Only complete GWSamplegen positive/negative catalogs with matching skymaps and
bundle provenance should be used for production preparation. Do not copy
legacy `injections_*.csv`, `coincs.dat` or `allsky.csv` into these directories.

## Recovery and Retention

From the repository root, inspect a run and resume failed or unprocessed events:

```bash
kn_simulation/bin/kn-sim status bns_train
kn_simulation/bin/kn-sim submit bns_train --resume
```

The normal finalizer publishes the aggregate after validating it. If the run
has reached terminal state but automatic finalization failed, use:

```bash
kn_simulation/bin/kn-sim compact bns_train
```

Compaction refuses incomplete inputs, retains failed/skipped event state, and
removes source shards only after aggregate validation succeeds. Status and
submission records remain for audit and resume. Keep the aggregate together
with prepared catalogs and manifests; it replaces temporary task shards as
the downstream simulation input.

Prepared files are protected from accidental replacement. Older prepared
schemas require explicit regeneration with `--overwrite-prepared` as described
in the [catalog contract](../README.md#catalog-contract). Historical SNANA
outputs need the documented `migrate-optical` and `validate-optical` steps
before `prune-snana`; the latter is report-only unless `--execute` is supplied.
