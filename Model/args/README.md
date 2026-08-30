# Model configuration layout

- `MAGIKS_BNS_NSBH*.json.example`: current training and ablation templates. Full and its HPO-based ablations use the HPO v7 Trial 27 schedule; `fiducial_params` uses the archived typical values near the search-space centre. All active templates use mixed KN/non-KN `training_aligned` validation. The no-cross ablation trains the same mixed gallery through `concat_proj`; no-fusion disables classification and fusion-gallery training entirely and is evaluated with contrastive scoring.
- `MAGIKS_BNS_NSBH_*_legacy_comparison.json.example`: pre-HPO-v7 full and physical-pairing controls retained for direct comparison; their runtime JSON files continue to target the original checkpoints.
- `archive/pre_hpo_v7_mixed_gallery_20260828/`: local byte-for-byte backup of the nine runtime configs and templates before this migration.
- `defaults/`: shared current defaults; experiment JSON overrides these values.
- `eval/`: current evaluation, non-KN retrieval comparison, joint and
  single-parameter crossed-pair GW--KN sensitivity, directional-win Physics Ejecta Bridge, reranking, and benchmark templates. Historical
  fixed-checkpoint attribution runners are retained for reproduction but are
  not part of the recommended workflow.
- `hpo/`: current HPO templates.
- `astro_args/` and `hpo_best_config/`: current specialised and selected configurations.
- `old/`: historical training, v11, debug, and evaluation configs kept for reproducibility; versioned defaults are retained alongside current defaults.

Runtime `*.json` files are local and ignored by Git. Generate one from a tracked template by replacing `<BASE_DIR>` and `<REPO_ROOT>` as shown in the repository README. Current configuration filenames use MAGIKS; historical ALBEF filenames are intentionally unchanged.
