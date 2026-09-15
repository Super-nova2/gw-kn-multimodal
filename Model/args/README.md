# MAGIKS Configuration Guide

[Project README](../../README_en.md) | [中文说明](../../README.md) | [Environment](../../ENVIRONMENT.md)

## Templates and Runtime Files

Git tracks `*.json.example`. Generate ignored runtime `*.json` files with the
[repository configuration example](../../README_en.md#configuration), which
replaces both `<BASE_DIR>` and `<REPO_ROOT>`, translates legacy OzSTAR path
prefixes, and preserves existing JSON files. The GW170817A template currently
contains absolute paths instead of placeholders, so the existing template
portability subtest fails even though generated runtime paths can be translated.
After a pull, compare existing configurations with their updated templates.
Data files, checkpoints and output directories are external prerequisites.

## Training and Defaults

From the repository root, the training interface is:

```bash
bash Model/scripts/train/train.sh Model/args/MAGIKS_BNS_NSBH_full.json \
  Model/args/defaults/MAGIKS_BNS_NSBH_default.json
```

The experiment overrides values in the default JSON. The second argument is
optional: the wrapper looks for an experiment-specific default, then a
version-specific default when applicable, and otherwise uses
`defaults/MAGIKS_BNS_NSBH_default.json`.

| Template | Role |
| --- | --- |
| [Full](MAGIKS_BNS_NSBH_full.json.example) | Current HPO-based mixed-gallery training recipe |
| [No contrastive loss](MAGIKS_BNS_NSBH_no_itc_loss.json.example) | Contrastive objective ablation |
| [No retrieval loss](MAGIKS_BNS_NSBH_no_retrieval_loss.json.example) | Fusion-gallery objective ablation |
| [No classification loss](MAGIKS_BNS_NSBH_no_cls_loss.json.example) | Classification objective ablation |
| [No cross-attention](MAGIKS_BNS_NSBH_no_cross_atten.json.example) | `concat_proj` fusion with mixed-gallery training |
| [No fusion](MAGIKS_BNS_NSBH_no_fusion.json.example) | Contrastive scoring; classification, fusion-gallery losses and gallery validation disabled; selects by `g2o_mrr` |
| [Fiducial parameters](MAGIKS_BNS_NSBH_fiducial_params.json.example) | Typical search-space parameter comparison |
| [Shared defaults](defaults/MAGIKS_BNS_NSBH_default.json.example) | Architecture, preprocessing and fallback training settings |

`Model/scripts/train/submit_ablation_train.sh` submits Full and the five
component ablations as six jobs; the fiducial comparison is not in that batch.

### Current Full Recipe

These settings come from the Full template after overriding the shared defaults:

| Setting | Current value |
| --- | --- |
| Fusion | `physical_dual_hgw` |
| Schedule | 100 epochs, 100 steps/epoch; staged contrastive/classification/retrieval training |
| Training gallery | `mixed_kn_nonkn`, size 1000, KN distractor fraction 0.25 |
| Coordinates and timing | Shared positive coordinates; parent-relative KN times; empirical/uniform non-KN mixture |
| Validation | `mixed_kn_nonkn`, `training_aligned`, sizes 100/500/1000/2000/5000, every 5 epochs |
| Best checkpoint | `mixed_gallery_macro_retrieval_score` |
| GW-negative guardrail | Enabled, recall 0.85 |
| Additional hard-negative term | Enable flag true, but `gallery_hard_neg_weight=0.0`; the flag alone does not imply a positive loss contribution |

The shared defaults instead specify `synthetic_time_sky_hard` validation and
`hard_gallery_macro_retrieval_score`. Do not infer an experiment's effective
settings from the defaults alone. The no-fusion ablation is an explicit
exception to the Full validation and checkpoint-selection recipe.

The two tracked `*_legacy_comparison.json.example` templates retain earlier
Full and physical-pairing recipes. Their checkpoint paths are experiment
inputs, not downloadable artifacts. Any local `archive/`, `old/`,
`astro_args/` or selected HPO runtime directories are not a promise that those
files are distributed in a fresh checkout. Current portable HPO templates live
under [hpo/](hpo), including [HPO v8](hpo/hpo_v8_mixed_gallery.json.example).

<a id="evaluation"></a>

## Evaluation Templates

Run each wrapper from the repository root and pass the generated JSON explicitly:

| Tracked template | Wrapper under `Model/scripts/eval/` |
| --- | --- |
| [Full classification](eval/cls_test/MAGIKS_BNS_NSBH_eval_full.json.example) and other tracked `cls_test` templates | `submit_test_evaluate.sh` |
| [Retrieval comparison](eval/retrieval_comparison.json.example) | `submit_retrieval_comparison.sh` |
| [Single-parameter sensitivity](eval/gw_kn_single_parameter_sensitivity.json.example) | `submit_gw_kn_pairing_sensitivity.sh` |
| [Physics Ejecta Bridge fit](eval/physics_ejecta_bridge_fit.json.example) | `submit_physics_ejecta_bridge_fit.sh` |
| [GW170817A scenarios](eval/retrieval_gw170817a_lsst.json.example) | `submit_gw170817a_retrieval.sh` |
| [Model speed](eval/model_speed_benchmark.json.example) | `submit_model_speed_benchmark.sh` |

The retrieval comparison contains nine methods: Full, five ablations,
Optical-only baseline, Fink Random Forest and Skymap-only. Its
`synthetic_time_sky_hard` non-KN galleries are a test protocol distinct from
mixed KN/non-KN training galleries. Fink RF needs an external `model.joblib`;
the optical and multimodal methods need their own runtime configs and weights.
Check the negative HDF5/group and redshift catalogs: the latter still reference
the earlier `production_rubin_dual` bundle and must match the test HDF5.

The single-parameter template compares Mixed Gallery v1, Default MAGIKS and an
optical null model. It uses `pairing_mode=single_parameter`, reports interaction
and directional-win statistics, and applies Holm correction. Validate before
submitting:

```bash
DRY_RUN=true bash Model/scripts/eval/submit_gw_kn_pairing_sensitivity.sh \
  Model/args/eval/gw_kn_single_parameter_sensitivity.json
```

The wrapper's no-argument default names a joint-sensitivity JSON for which no
tracked example is currently shipped. Always pass the available single-parameter
configuration explicitly. Validation checks inputs and avoids job submission;
the shell wrapper can still create its log directory.

### Physics Ejecta Bridge v2

The fit template builds a frozen artifact from training data. Its
`test_data_path` is used only to check train/test overlap, not for fitting or
hyperparameter selection. The directional-win implementation in
[eval_gw_kn_directional_bridge.py](../scripts/eval/eval_gw_kn_directional_bridge.py)
is dispatched by the pairing evaluator when `primary_metric=directional_win_rate`.

No portable v2 evaluation template is currently tracked. A separately prepared
configuration must supply the named neural comparisons, a `GW-blind Control`,
a `Physics Ejecta Bridge` model with `type=physics_ejecta_bridge` and
`artifact_path`, single-parameter pairing settings, and a matching
`expected_pair_manifest_sha256`. v2 does not support interaction comparison.
The existing v1 JSON does not provide these requirements.

Standalone mixed-retrieval, fixed-checkpoint attribution and Bridge
label-shuffle runners were removed. Historical runtime JSON or saved results
do not restore those entry points; use Git history when reproducing that code.

## Compatibility and Inputs

`ALBEF_dataset`, `ALBEF/albef_best.pth` and legacy checkpoint keys retain their
names for compatibility. Templates name expected locations, not verified
artifacts. Keep the exact training config with each checkpoint and check source
and split, data groups, timing conventions and output directory before evaluation.
