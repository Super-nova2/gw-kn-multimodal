# Model configuration layout

- `MAGIKS_BNS_NSBH*.json.example`: current training and ablation templates.
- `defaults/`: shared current defaults; experiment JSON overrides these values.
- `eval/`: current evaluation, non-KN retrieval comparison, joint and
  single-parameter crossed-pair GW--KN sensitivity, reranking, and benchmark templates. Historical
  fixed-checkpoint attribution runners are retained for reproduction but are
  not part of the recommended workflow.
- `hpo/`: current HPO templates.
- `astro_args/` and `hpo_best_config/`: current specialised and selected configurations.
- `old/`: historical training, v11, debug, and evaluation configs kept for reproducibility; versioned defaults are retained alongside current defaults.

Runtime `*.json` files are local and ignored by Git. Generate one from a tracked template by replacing `<BASE_DIR>` and `<REPO_ROOT>` as shown in the repository README. Current configuration filenames use MAGIKS; historical ALBEF filenames are intentionally unchanged.
