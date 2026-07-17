# Optical-only configuration layout

- `optical_only_kn_baseline.json.example`: randomly initialized comparison baseline and launcher default.
- `optical_only_kn_v15_win1020.json.example`: v15 recipe with runtime input-window cropping.
- `optical_only_kn_v16.json.example`: current recommended pretrained optical-only recipe.
- `old/`: historical configurations retained for reproducibility.
- `old/two_branch/`: historical v11-v13 two-branch experiments.
- `old/experimental/`: incomplete or unsupported experiments; these are not current runnable templates.

Runtime `*.json` files are local and ignored by Git. Generate them from the current tracked templates by replacing `<BASE_DIR>` with the workspace data root. The legacy `pretrained_albef_ckpt` key and `ALBEF/albef_best.pth` artifact paths remain unchanged for checkpoint compatibility.
The current v15/v16 templates initialize from the latest MAGIKS full-model checkpoint; the checkpoint is still stored under the legacy `ALBEF/albef_best.pth` artifact path.
