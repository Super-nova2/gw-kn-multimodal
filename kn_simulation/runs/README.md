# Runtime directories

This directory is intentionally empty in Git. Each production profile creates
one ignored runtime directory here containing `catalog.csv`, `kn_catalog.csv`,
audit manifests, Slurm submission metadata, and event status sidecars.

Only a complete catalog produced by the Rubin-time GWSamplegen profiles may be
used. Do not copy legacy `injections_*.csv` files into this directory.
