#!/usr/bin/env python3
"""Reproduce the partial HPO v7 analysis and portable report inputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = Path(
    "/fred/oz016/bgao_kn/data/model/hpo_results_bns_nsbh_v7_retrieval_robust"
)
OUT = Path(__file__).resolve().parent
SIZES = (100, 500, 1000, 2000, 5000)
SIZE_WEIGHTS = {100: 0.10, 500: 0.15, 1000: 0.20, 2000: 0.25, 5000: 0.30}
PARAMS = (
    "lr",
    "weight_decay",
    "gallery_loss_weight",
    "itc_weight",
    "cls_weight",
    "fusion_physical_weight",
    "stage_itc_epochs",
    "stage_joint_itc_start_weight",
)


def load_rows() -> list[dict]:
    rows = []
    result_paths = sorted(
        RESULT_ROOT.glob("results/trial_*/results_epoch_30.json"),
        key=lambda path: int(path.parent.name.split("_")[1]),
    )
    for result_path in result_paths:
        trial = int(result_path.parent.name.split("_")[1])
        result = json.loads(result_path.read_text())
        config = json.loads(
            (RESULT_ROOT / f"configs/trial_{trial}_epoch_30.json").read_text()
        )
        hard = result["val_hard_gallery"]
        source_macro = hard["source_macro"]
        score = sum(
            SIZE_WEIGHTS[size]
            * (
                0.8 * source_macro[f"gallery_{size}_mrr"]
                + 0.2 * source_macro[f"gallery_{size}_recall_at_1"]
            )
            for size in SIZES
        )
        row = {
            "trial": trial,
            "trial_label": f"Trial {trial}",
            "score": score,
            "saved_score": result["val_hard_gallery_macro_retrieval_score"],
            "auprc": result["val_auprc"],
            "auroc": result["val_auroc"],
            "neg_gw_min_recall": result["val_neg_gw_min_recall"],
            "best_epoch": result["best_epoch_1based"],
        }
        row.update({name: config[name] for name in PARAMS})
        for size in SIZES:
            row[f"g{size}_mrr"] = source_macro[f"gallery_{size}_mrr"]
            row[f"g{size}_r1"] = source_macro[f"gallery_{size}_recall_at_1"]
        for source in ("bns", "nsbh"):
            source_metrics = hard["by_source"][source]
            row[f"{source}_score"] = sum(
                SIZE_WEIGHTS[size]
                * (
                    0.8 * source_metrics[f"gallery_{size}_mrr"]
                    + 0.2 * source_metrics[f"gallery_{size}_recall_at_1"]
                )
                for size in SIZES
            )
        rows.append(row)
    assert len(rows) == 30
    baseline = rows[0]
    auprc_floor = baseline["auprc"] - 0.005
    for row in rows:
        row["score_delta_vs_full"] = row["score"] - baseline["score"]
        row["score_gain_pct_vs_full"] = (
            row["score_delta_vs_full"] / baseline["score"]
        )
        row["constraint_neg_recall_pass"] = row["neg_gw_min_recall"] >= 0.85
        row["constraint_auprc_pass"] = row["auprc"] >= auprc_floor
        row["feasible"] = bool(
            row["constraint_neg_recall_pass"] and row["constraint_auprc_pass"]
        )
        row["source_gap"] = abs(row["bns_score"] - row["nsbh_score"])
    for rank, row in enumerate(sorted(rows, key=lambda item: item["score"], reverse=True), 1):
        row["score_rank"] = rank
    other = sorted(
        rows[1:], key=lambda item: (item["feasible"], item["score"]), reverse=True
    )
    promoted = {0, *(row["trial"] for row in other[:19])}
    for row in rows:
        row["would_promote_to_60"] = row["trial"] in promoted
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    baseline = rows[0]
    ranked = sorted(rows, key=lambda item: item["score"], reverse=True)
    best = ranked[0]
    max_discrepancy = max(abs(row["score"] - row["saved_score"]) for row in rows)

    parameter_effects = []
    for parameter in PARAMS:
        rho, p_value = spearmanr(
            [row[parameter] for row in rows], [row["score"] for row in rows]
        )
        parameter_effects.append(
            {
                "parameter": parameter,
                "rho": float(rho),
                "abs_rho": abs(float(rho)),
                "p_value": float(p_value),
                "n_trials": len(rows),
                "interpretation": "observational association; TPE sampling is confounded",
            }
        )
    parameter_effects.sort(key=lambda item: item["abs_rho"], reverse=True)

    group_effects = []
    for parameter in PARAMS[2:]:
        for value in sorted({row[parameter] for row in rows}):
            subset = [row for row in rows if row[parameter] == value]
            group_effects.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "n": len(subset),
                    "mean_score": float(np.mean([row["score"] for row in subset])),
                    "score_sd": (
                        float(np.std([row["score"] for row in subset], ddof=1))
                        if len(subset) > 1
                        else None
                    ),
                    "feasible_rate": float(np.mean([row["feasible"] for row in subset])),
                }
            )

    gallery_comparison = []
    for size in SIZES:
        gallery_comparison.append(
            {
                "gallery_size": str(size),
                "weight": SIZE_WEIGHTS[size],
                "full_mrr": baseline[f"g{size}_mrr"],
                "trial25_mrr": best[f"g{size}_mrr"],
                "mrr_delta": best[f"g{size}_mrr"] - baseline[f"g{size}_mrr"],
                "full_r1": baseline[f"g{size}_r1"],
                "trial25_r1": best[f"g{size}_r1"],
            }
        )

    summary = {
        "generated_at": "2026-08-14T00:00:00Z",
        "job": {
            "job_id": 15354216,
            "state": "FAILED",
            "exit_code": "1:0",
            "elapsed": "1-08:32:04",
            "completed_initial_trials": 30,
            "completed_epoch60_trials": 0,
            "completed_epoch100_trials": 0,
            "confirmation_runs": 0,
            "failure_stage": "first epoch-60 resume, baseline trial 0",
            "failure_reason": "PyTorch 2.6 weights_only=True rejected NumPy RNG state in the trusted local checkpoint",
        },
        "validation": {
            "queries": 128,
            "queries_per_source": 64,
            "gallery_trials": 2,
            "evaluations_per_size": 256,
            "gallery_sizes": list(SIZES),
            "coverage": 1.0,
            "manifest_spotcheck_sha256": "8fba3ccb5fd2c4c9b89ebd452c95f9beb7713a7b1213291e17ded77e5a4cd0e4",
            "objective_max_abs_recompute_error": max_discrepancy,
        },
        "baseline": baseline,
        "best_observed": best,
        "feasible_trials": sum(row["feasible"] for row in rows),
        "auprc_constraint_failures": sum(
            not row["constraint_auprc_pass"] for row in rows
        ),
        "neg_recall_constraint_failures": sum(
            not row["constraint_neg_recall_pass"] for row in rows
        ),
        "best_epoch_counts": dict(Counter(row["best_epoch"] for row in rows)),
        "would_promote": [
            row["trial"] for row in rows if row["would_promote_to_60"]
        ],
        "parameter_effects": parameter_effects,
        "group_effects": group_effects,
        "chart_map": [
            {
                "section": "30-epoch ranking",
                "question": "Which configurations lead after the completed rung?",
                "type": "horizontalBar",
                "fields": ["trial_label", "score"],
            },
            {
                "section": "Gallery-size robustness",
                "question": "Does the leading trial improve large galleries?",
                "type": "bar grouped",
                "fields": ["gallery_size", "full_mrr", "trial25_mrr"],
            },
            {
                "section": "Parameter associations",
                "question": "Which sampled parameters associate with rung-30 score?",
                "type": "horizontalBar",
                "fields": ["parameter", "rho"],
            },
        ],
    }

    write_csv(OUT / "trial_summary.csv", sorted(rows, key=lambda item: item["trial"]))
    write_csv(OUT / "parameter_effects.csv", parameter_effects)
    write_csv(OUT / "parameter_group_effects.csv", group_effects)
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    top10 = ranked[:10]
    top_table = []
    for row in top10:
        top_table.append(
            {
                "rank": row["score_rank"],
                "trial": row["trial"],
                "score": row["score"],
                "delta": row["score_delta_vs_full"],
                "auprc": row["auprc"],
                "neg_recall": row["neg_gw_min_recall"],
                "feasible": row["feasible"],
                "lr": row["lr"],
                "gallery_weight": row["gallery_loss_weight"],
                "itc_weight": row["itc_weight"],
                "cls_weight": row["cls_weight"],
                "stage_itc_epochs": row["stage_itc_epochs"],
            }
        )

    source_results = {
        "id": "v7-results",
        "label": "HPO v7 rung-30 saved configs and result JSONs",
        "query": {
            "engine": "filesystem-json",
            "language": "python",
            "description": "Independent aggregation of all 30 trial configs and results_epoch_30.json files.",
            "executed_at": "2026-08-14T00:00:00Z",
            "tables_used": [
                "hpo_results_bns_nsbh_v7_retrieval_robust/configs/trial_*_epoch_30.json",
                "hpo_results_bns_nsbh_v7_retrieval_robust/results/trial_*/results_epoch_30.json",
            ],
            "filters": ["epoch rung = 30", "trials 0 through 29"],
            "metric_definitions": [
                "score = sum_g w_g * (0.8 * source_macro_MRR_g + 0.2 * source_macro_R@1_g)",
                "weights G100/G500/G1000/G2000/G5000 = 0.10/0.15/0.20/0.25/0.30",
                "feasible = min negative-GW recall >= 0.85 and AUPRC >= Full AUPRC - 0.005",
            ],
        },
    }
    source_job = {
        "id": "slurm-job",
        "label": "Slurm job 15354216 accounting and terminal log",
        "query": {
            "engine": "slurm-and-filesystem-log",
            "description": "sacct state plus MAGIKS_HPO_v7_retrieval_robust_15354216.out failure trace.",
            "executed_at": "2026-08-14T00:00:00Z",
            "tables_used": ["Slurm accounting", "HPO terminal log"],
        },
    }
    sources = [source_results, source_job]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "HPO v7 任务结果诊断",
            "description": "Job 15354216 的完成度、30-epoch 搜索结果、参数信号和恢复建议。",
            "generatedAt": "2026-08-14T00:00:00Z",
            "sources": sources,
            "cards": [
                {
                    "id": "completion",
                    "dataset": "headline",
                    "sourceId": "slurm-job",
                    "description": "完成的初始 trial；后续 rung 未执行。",
                    "metrics": [
                        {"label": "完成 trial", "field": "completed_trials", "format": "number"},
                        {"label": "计划 trial", "field": "planned_trials", "format": "number"},
                    ],
                },
                {
                    "id": "best-score",
                    "dataset": "headline",
                    "sourceId": "v7-results",
                    "description": "仅代表 30-epoch tune split 的最佳观测值。",
                    "metrics": [
                        {"label": "最佳 score", "field": "best_score", "format": "number"},
                        {"label": "较 Full", "field": "best_delta", "format": "number", "signed": True},
                    ],
                },
                {
                    "id": "feasible-count",
                    "dataset": "headline",
                    "sourceId": "v7-results",
                    "description": "同时通过负 GW 召回和相对 Full AUPRC 约束。",
                    "metrics": [
                        {"label": "可行 trial", "field": "feasible_trials", "format": "number"},
                        {"label": "总数", "field": "planned_trials", "format": "number"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "top-ranking",
                    "title": "30-epoch retrieval score 排名",
                    "subtitle": "前 10 个 trial；虚线为 Full 基线 0.2643",
                    "intent": "comparison",
                    "question": "Which configurations lead after the completed rung?",
                    "rationale": "Sorted horizontal bars make the discrete trial ranking and baseline gap directly comparable.",
                    "type": "horizontalBar",
                    "dataset": "top10",
                    "sourceId": "v7-results",
                    "xField": "trial_label",
                    "xAxisTitle": "Trial",
                    "yAxisTitle": "Weighted retrieval score",
                    "series": [{"field": "score", "label": "Score", "color": "blue"}],
                    "valueFormat": "number",
                    "layout": "full",
                    "maxRows": 10,
                    "referenceLines": [{"axis": "y", "value": baseline["score"], "label": "Full", "lineStyle": "dashed", "color": "neutral"}],
                    "settings": {"sort": "descending", "showValues": True},
                },
                {
                    "id": "gallery-comparison",
                    "title": "Full 与 Trial 25 的 source-macro MRR",
                    "subtitle": "同一固定 tune manifest；大 gallery 的相对增益更明显",
                    "intent": "comparison",
                    "question": "Does the leading trial improve large galleries?",
                    "rationale": "Grouped bars compare two configurations at five discrete gallery sizes without implying a continuous trend.",
                    "type": "bar",
                    "dataset": "gallery_comparison",
                    "sourceId": "v7-results",
                    "xField": "gallery_size",
                    "xAxisTitle": "Gallery size",
                    "yAxisTitle": "Source-macro MRR",
                    "series": [
                        {"field": "full_mrr", "label": "Full", "color": "neutral", "role": "baseline"},
                        {"field": "trial25_mrr", "label": "Trial 25", "color": "blue", "role": "actual"},
                    ],
                    "valueFormat": "number",
                    "layout": "full",
                    "settings": {"groupMode": "grouped", "sort": "none", "showValues": True},
                    "legend": {"position": "bottom", "sort": "spec"},
                },
                {
                    "id": "parameter-association",
                    "title": "参数与 30-epoch score 的 Spearman 相关",
                    "subtitle": "仅为 TPE 样本内关联，不是独立因果效应",
                    "intent": "relationship",
                    "question": "Which sampled parameters associate with rung-30 score?",
                    "rationale": "Signed bars expose direction and magnitude while a zero line distinguishes positive and negative associations.",
                    "type": "horizontalBar",
                    "dataset": "parameter_effects",
                    "sourceId": "v7-results",
                    "xField": "parameter",
                    "xAxisTitle": "Parameter",
                    "yAxisTitle": "Spearman rho",
                    "series": [{"field": "rho", "label": "rho", "color": "purple"}],
                    "valueFormat": "number",
                    "layout": "full",
                    "referenceLines": [{"axis": "y", "value": 0, "label": "0", "color": "neutral"}],
                    "settings": {"sort": "descending", "showValues": True},
                },
            ],
            "tables": [
                {
                    "id": "top-table",
                    "title": "前 10 个 trial 精确结果",
                    "subtitle": "按 weighted retrieval score 降序；约束使用 Full AUPRC - 0.005",
                    "dataset": "top_table",
                    "sourceId": "v7-results",
                    "density": "dense",
                    "layout": "full",
                    "defaultSort": {"field": "rank", "direction": "asc"},
                    "columns": [
                        {"field": "rank", "label": "Rank", "format": "number"},
                        {"field": "trial", "label": "Trial", "format": "number"},
                        {"field": "score", "label": "Score", "format": "number"},
                        {"field": "delta", "label": "vs Full", "format": "number", "movement": True},
                        {"field": "auprc", "label": "AUPRC", "format": "number"},
                        {"field": "neg_recall", "label": "Neg recall", "format": "number"},
                        {"field": "feasible", "label": "Feasible", "type": "text"},
                        {"field": "lr", "label": "LR", "format": "number"},
                        {"field": "gallery_weight", "label": "Gallery w", "format": "number"},
                        {"field": "itc_weight", "label": "ITC w", "format": "number"},
                        {"field": "cls_weight", "label": "CLS w", "format": "number"},
                    ],
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# HPO v7 任务结果诊断", "layout": "full"},
                {
                    "id": "technical-summary",
                    "type": "markdown",
                    "body": "## 技术摘要\n\n**任务在第一层之后失败，不能视为完成的 v7 HPO。** 30 个初始 trial 均完成到 epoch 30，但 60-epoch rung 的第一个 Full 基线在加载续训 checkpoint 时退出；没有任何 epoch-60、epoch-100 或三种子 confirmation 结果。\n\n在可用的第一层中，Trial 25 的 weighted retrieval score 为 **0.31184**，比 Full 的 **0.26426** 高 **0.04758（+18.0%）**，且通过两个约束。不过这只是固定 tune split、单训练种子、30 epochs 的中间结果，不能据此替换 Full。",
                    "layout": "full",
                },
                {"id": "headline-strip", "type": "metric-strip", "cardIds": ["completion", "best-score", "feasible-count"], "layout": "full"},
                {
                    "id": "ranking-finding",
                    "type": "markdown",
                    "sourceId": "v7-results",
                    "body": "## Trial 25 在完整第一层中领先\n\nTrial 25 在所有五个 gallery size 上都优于 Full；其 G2000/G5000 MRR 分别由 **0.2151/0.1274** 提升到 **0.2592/0.1656**。AUPRC 同时从 **0.8739** 升至 **0.8843**。负 GW 最小召回从 **0.8848** 降至 **0.8689**，仍高于 0.85 门槛。",
                    "layout": "full",
                },
                {"id": "ranking-chart", "type": "chart", "chartId": "top-ranking", "layout": "full"},
                {
                    "id": "gallery-finding",
                    "type": "markdown",
                    "sourceId": "v7-results",
                    "body": "## 增益并非只来自小 gallery\n\nTrial 25 的相对 MRR 增益随 gallery 变难而扩大：G100 约 **+6.3%**，G1000 约 **+22.6%**，G5000 约 **+30.0%**。这与 v7 对大 gallery 加权更高的目标一致。BNS 和 NSBH weighted score 都改善，但 BNS 增益更大，因此来源差距从 **0.0749** 扩至 **0.0878**。Trial 17 的总分略低（0.30193），但来源差距更小（0.0426）且负 GW 召回最高（0.8958），是值得在后续 rung 保留的稳健候选。",
                    "layout": "full",
                },
                {"id": "gallery-chart", "type": "chart", "chartId": "gallery-comparison", "layout": "full"},
                {
                    "id": "constraint-finding",
                    "type": "markdown",
                    "sourceId": "v7-results",
                    "body": "## 约束主要筛掉分类退化，而不是负 GW 召回\n\n**23/30** 个 trial 可行；所有 30 个 trial 的负 GW 最小召回都达到 0.85。7 个失败全部由 AUPRC 低于 Full−0.005 触发，包括 score 排名第 11 的 Trial 28。约束因此发挥了预期作用：阻止 retrieval 改善掩盖分类质量下降。",
                    "layout": "full",
                },
                {"id": "top-table-block", "type": "table", "tableId": "top-table", "layout": "full"},
                {
                    "id": "parameter-finding",
                    "type": "markdown",
                    "sourceId": "v7-results",
                    "body": "## 第一层信号偏向更高 LR、gallery weight 与 ITC weight\n\n样本内最强关联是 learning rate（Spearman ρ=**0.574**, p=0.0009）和 gallery loss weight（ρ=**0.497**, p=0.0052）。`itc_weight=0.5` 的组均值明显较低（0.2254），而 1.0 与 1.5 基本相同（0.2786 与 0.2789）。weight decay 没有可见单调关系（ρ=0.060）。这些比较来自自适应 TPE 的不均衡、相关采样，不能解释为参数的独立因果效应。",
                    "layout": "full",
                },
                {"id": "parameter-chart", "type": "chart", "chartId": "parameter-association", "layout": "full"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": "v7-results",
                    "body": "## 评测范围与指标定义\n\n第一层使用固定 tune manifest：128 个查询事件（BNS/NSBH 各 64），每个 gallery size 两次 trial，因此每个 size 有 256 次查询评测；G100、G500、G1000、G2000、G5000 均为 100% 完整 coverage。目标是各 size 的 source-macro `0.8×MRR + 0.2×R@1`，再按 0.10/0.15/0.20/0.25/0.30 加权。抽查 Trial 0、25、29 的 manifest SHA-256 相同，目标函数独立复算最大误差小于 6×10⁻¹⁷。",
                    "layout": "full",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": "## 分析方法\n\n重新读取全部 30 份 trial config 和 `results_epoch_30.json`，独立复算 weighted score、相对 Full AUPRC 约束、负 GW 召回约束和预期 30→60 晋级名单；同时按参数取值计算组均值，并计算参数与 score 的 Spearman 相关。Slurm accounting、输出文件清单和终端 traceback 用于判定任务完成度与失败阶段。",
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## 限制、失败原因与稳健性\n\n**高影响限制：** 没有后续 rung 和 confirmation，当前排名可能随训练时长和种子改变。26/30 个 trial 的最佳 checkpoint 正好位于 epoch 30，说明大多数配置仍在改善，第一层排名尚未收敛。\n\n**失败原因已验证：** PyTorch 2.6 将 `torch.load` 默认改为 `weights_only=True`；续训 checkpoint 保存了 NumPy RNG 状态，安全 loader 因 `numpy.core.multiarray._reconstruct` 未列入 allowlist 而拒绝加载。失败发生在模型开始 epoch 31 之前，不是训练数值崩溃。checkpoint 文件存在且 ZIP 结构可读，但本次分析未使用不安全反序列化去验证其全部对象内容。",
                    "layout": "full",
                },
                {
                    "id": "next-steps",
                    "type": "markdown",
                    "body": "## 建议的恢复顺序\n\n1. 修复 trusted local checkpoint 的加载兼容性，并补充 PyTorch 2.6 resume smoke test。\n2. 增加从已完成 rung-30 结果继续的入口；当前 driver 检测到非空 study 会拒绝重新启动，因此不能只原样重提同一配置。\n3. 在隔离输出目录先验证 Trial 0 能从 epoch 30 继续到 31，且 global step、LR、best state 和 RNG 恢复正确。\n4. 使用既定晋级名单继续 20 个 trial 到 epoch 60；不要重复已经完成的 900 trial-epochs。\n5. 完成 epoch 100 与三种子 confirmation 后，再依据预设 replacement gates 决定是否替换 Full。",
                    "layout": "full",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": "## 后续需要回答的问题\n\n- Trial 25 的优势在 epoch 60/100 是否保持，还是高 LR 带来的早期优势？\n- Trial 17 的更小来源差距能否在 confirmation 上带来更好的 worst-source 稳健性？\n- `itc_weight=1.0` 与 1.5 在完整训练长度下是否仍近似等价？",
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-08-14T00:00:00Z",
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        "completed_trials": 30,
                        "planned_trials": 30,
                        "best_score": best["score"],
                        "best_delta": best["score_delta_vs_full"],
                        "feasible_trials": summary["feasible_trials"],
                    }
                ],
                "top10": top10,
                "top_table": top_table,
                "gallery_comparison": gallery_comparison,
                "parameter_effects": parameter_effects,
            },
        },
        "sources": sources,
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
