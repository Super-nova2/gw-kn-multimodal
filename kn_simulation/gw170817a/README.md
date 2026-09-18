# GW170817A LSST scenario retrieval

[Project README](../../README.md) | [Production simulation](../README.md) | [Environment](../../ENVIRONMENT.md)

This directory builds one fixed-physics GW170817A retrieval benchmark. It does
not scan redshift. Every GW parent uses the same posterior-median detector-frame
masses, aligned spins and signed inclination cosine from
`GW170817_GWTC-1.hdf5`, together with the unmodified
`bayestar_no_virgo.fits` probability and distance layers. Kilonova simulations
use the NGC 4993 host distance of 40.7 Mpc; the SNANA redshift is the Planck15
luminosity-distance equivalent.

The benchmark is a complete crossed panel:

- 10 counterfactual trigger epochs, separated by one sidereal year and contained
  in the Rubin baseline v5.1 OpSim interval;
- one shared true sky position plus unique samples from the 90% GW posterior;
- 50 coordinates retained only when their light curves are usable in all 10
  observing scenarios;
- 3 distractor-gallery repeats for every positive coordinate.

## Build the dataset

Run from the repository root with the environment activated and `BASE_DIR`
set to the data workspace. Required external inputs include
`GW170817_GWTC-1.hdf5` (default posterior dataset
`IMRPhenomPv2NRT_lowSpin_posterior`), `bayestar_no_virgo.fits`, the baseline
v5.1 OpSim database, SNANA and its BNS models. By default the event inputs live
under `$BASE_DIR/data/GW_real_events/GW_data/GW170817A/`.

This wrapper resolves the code as `$WORKSPACE_ROOT/gw-kn-multimodal`, unlike
production profiles which derive the checkout path automatically. Keep that
checkout name and set its parent explicitly if data and code roots differ.
Submit all catalog, SNANA preparation, simulation and HDF5 stages with:

```bash
WORKSPACE_ROOT="$(dirname "$PWD")" \
  bash kn_simulation/gw170817a/submit_gw170817a_lsst_pipeline.sh
```

`STAGE=catalog|snana|sim|h5` can rerun an individual stage. Generated scenario
artifacts live under `data/GW_real_events/GW_data/GW170817A/lsst_scenario_experiment`;
the final dataset is
`$BASE_DIR/data/ALBEF_dataset/gw170817a_lsst_scenarios.h5`. Paths can be
overridden through the wrapper's environment variables, including `SKYMAP`,
`POSTERIOR_H5`, `OPSIM_DB`, `OUTPUT_DIR` and `OUTPUT_H5`. Re-running the SNANA
preparation stage replaces its generated SIM_INPUT, SIMLIB and coordinate
products; preserve any run that must remain reproducible before reusing its
output directory. The standard scenario path is separate from historical
redshift-grid results.

The HDF5 contains 10 GW scenario parents and 500 optical curves. Optical rows
record `scenario_id`, `coordinate_id`, true-position status, sky credible level,
observation counts, ToO metadata and first-detection time. Redshift-bin fields
are intentionally absent.

## Run retrieval and analysis

Wait for dataset construction to finish. Generate the ignored runtime JSON
from [retrieval_gw170817a_lsst.json.example](../../Model/args/eval/retrieval_gw170817a_lsst.json.example)
using the [README configuration command](../../README.md#configuration).
The current template contains hardcoded OzSTAR paths; that command translates
their data and repository prefixes as well as expanding placeholders in the
other templates. Copying this template alone does not adapt it to a different
workspace. The hardcoded source template also causes the existing portability
subtest to fail; generating a local JSON does not change that template.
Confirm the dataset,
negative HDF5/group, model configs and checkpoint paths, then submit:

```bash
bash Model/scripts/eval/submit_gw170817a_retrieval.sh \
  Model/args/eval/retrieval_gw170817a_lsst.json
```

The job delegates scoring to the common retrieval evaluator and compares only
`Full`, `Optical-only baseline`, and `Skymap-only`. Exhaustive selection expands
50 coordinates x 3 repeats into 150 trials per scenario. The event wrapper then
runs a 10,000-draw two-way bootstrap that independently resamples observing
scenarios and coordinates. It writes overall metrics, paired Full-minus-baseline
deltas, and condition summaries under `scenario_analysis/`.

Both wrappers still contain OzSTAR-specific Slurm defaults and log paths.
Inspect them before running on another cluster; setting `BASE_DIR` alone
does not replace every scheduler directive or hardcoded path.
