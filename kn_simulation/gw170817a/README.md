# GW170817A LSST scenario retrieval

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

Submit all catalog, SNANA preparation, simulation and HDF5 stages with:

```bash
kn_simulation/gw170817a/submit_gw170817a_lsst_pipeline.sh
```

`STAGE=catalog|snana|sim|h5` can rerun an individual stage. Generated scenario
artifacts live under `data/GW_real_events/GW_data/GW170817A/lsst_scenario_experiment`;
the final dataset is
`data/ALBEF_dataset/gw170817a_lsst_scenarios.h5`. Existing redshift-grid data
and historical result directories are not modified.

The HDF5 contains 10 GW scenario parents and 500 optical curves. Optical rows
record `scenario_id`, `coordinate_id`, true-position status, sky credible level,
observation counts, ToO metadata and first-detection time. Redshift-bin fields
are intentionally absent.

## Run retrieval and analysis

Copy the tracked example config if a local runtime config is not already present:

```bash
cp Model/args/eval/retrieval_gw170817a_lsst.json.example \
   Model/args/eval/retrieval_gw170817a_lsst.json
Model/scripts/eval/submit_gw170817a_retrieval.sh
```

The job delegates scoring to the common retrieval evaluator and compares only
`Full`, `Optical-only baseline`, and `Skymap-only`. Exhaustive selection expands
50 coordinates x 3 repeats into 150 trials per scenario. The event wrapper then
runs a 10,000-draw two-way bootstrap that independently resamples observing
scenarios and coordinates. It writes overall metrics, paired Full-minus-baseline
deltas, and condition summaries under `scenario_analysis/`.
