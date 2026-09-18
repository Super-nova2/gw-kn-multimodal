# Optical-Only Configuration Guide

[Project README](../../README.md) | [中文说明](../../README_cn.md) | [MAGIKS configuration](../../Model/args/README.md)

## Current Templates

| Template | Initialization and purpose |
| --- | --- |
| [Baseline](optical_only_kn_baseline.json.example) | Random initialization (`pretrained_albef_ckpt=null`); independent comparison and launcher default |
| [v15 window 10/20](optical_only_kn_v15_win1020.json.example) | MAGIKS Full optical encoder initialization with runtime input-window cropping |
| [v16](optical_only_kn_v16.json.example) | Current pretrained single-view prefix fine-tuning recipe |

v16 uses two head-training epochs followed by twelve stage-2 epochs, with no
stage-3 training. Consistency losses, adversarial heads and gradient reversal
(GRL) are disabled. It is not the historical DANN experiment.

v15/v16 expect the MAGIKS Full checkpoint at
`<BASE_DIR>/data/model/checkpoints_bns_nsbh/bns_nsbh_full/ALBEF/albef_best.pth`.
The legacy `pretrained_albef_ckpt` key and `ALBEF/albef_best.pth` artifact name
are preserved for compatibility; the artifact is not included in Git.

## Prepare and Run

Generate runtime JSON using the [repository configuration example](../../README.md#configuration).
It replaces `<BASE_DIR>` and `<REPO_ROOT>` in tracked templates and preserves
existing local files. Runtime `*.json` is ignored by Git and needs manual
comparison with templates after updates.

From the repository root, after preparing data and any required pretrained weights:

```bash
bash optical_only/scripts/train/train.sh optical_only/args/optical_only_kn_baseline.json
```

Or select the pretrained recipe explicitly:

```bash
bash optical_only/scripts/train/train.sh optical_only/args/optical_only_kn_v16.json
```

The wrapper submits a Slurm job when outside an allocation and runs evaluation
after training, using [evaluate.py](../scripts/eval/evaluate.py). Prepare its
evaluation inputs before submission as well as the training inputs.

## Input Checklist

- `pos_data_path` points to the optical-only training HDF5.
- `neg_data_path` and `neg_group` select external optical negatives; current
  templates train on `ELASTICC2/optical_data`.
- `eval_pos_data_path`, `eval_neg_data_path` and `eval_neg_group` are independent
  of training paths. Baseline evaluates on ELASTICC negatives; v16 uses Tutorial
  negatives. Do not assume these protocols are interchangeable.
- `offset_dist_npz` supplies the empirical time-offset distribution. Check the
  file and configured key before using time-offset training or ensemble evaluation.
- `ckpt_path` is an output root; the wrapper derives the run-specific checkpoint
  location. Verify that the corresponding retrieval configuration points to the
  actual trained artifact.

The data-building entry point is
[submit_create_datasets.sh](../scripts/data/submit_create_datasets.sh).
Joint-model external negatives and optical-only negatives may live in different
directories; always use the HDF5 group specified by the selected configuration.

## Historical Configurations

Tracked templates under [old/two_branch/](old/two_branch) preserve v11-v13
experiments. Other historical or experimental runtime files may exist only in
the local workspace. In particular, the unsupported v17 DANN runtime config
is not a current portable template. Retain original configurations with old
checkpoints rather than assuming a current recipe reproduces them.
