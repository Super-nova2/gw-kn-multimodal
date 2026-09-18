# gw-kn-multimodal

[English](README.md) | [中文](README_cn.md)

面向千新星（kilonova, KN）识别的多模态研究仓库：从 GWSamplegen 引力波目录出发，经 Rubin/SNANA 光学模拟和 HDF5 数据集构建，训练与评估 GW + optical 联合模型 MAGIKS，并与 optical-only、Skymap-only 和 Fink Random Forest 基线比较。

## 研究流程

```text
外部 GWSamplegen 双目录及各自的 skymap
    -> kn_simulation: 正样本 Rubin/SNANA 模拟，保留物理负样本目录
    -> simulation_intermediates.h5
    -> 联合 HDF5 / optical-only HDF5
    -> MAGIKS / optical-only 训练
    -> 分类、检索消融、物理敏感性、GW170817A 场景评估
```

研究对象包括 BNS/NSBH 事件、GW 标量和 skymap、以首次探测对齐的 luptitude 光变，以及 GW 与光学候选的配对关系。数据、checkpoint 和运行结果需另行准备，不随代码分发。

## 目录与说明

```text
gw-kn-multimodal/
├── Model/
│   ├── model.py, data_loader.py       # MAGIKS 模型、数据读取与采样
│   ├── args/                         # 已跟踪的配置模板与共享默认值
│   └── scripts/                      # data / train / eval / hpo
├── optical_only/
│   ├── args/                         # 基线及预训练光学模型配置模板
│   └── scripts/                      # data / train / eval / analysis / plot
├── kn_simulation/
│   ├── bin/kn-sim                    # 光学生产模拟入口
│   ├── profiles/                     # BNS/NSBH train/test YAML
│   ├── gw170817a/                    # GW170817A LSST 场景实验
│   └── runs/                         # 被 Git 忽略的运行产物
├── plots_scripts/                    # 论文绘图与已有结果重绘
├── figures/                          # 已保存的图件
├── tests/                            # 配置、数据、模型和模拟测试
├── pyproject.toml, uv.lock           # 完整研究环境与锁定版本
└── requirements*.txt                 # 最小依赖 / 历史环境快照
```

| 说明 | 内容 |
| --- | --- |
| [环境配置](ENVIRONMENT.md) | Python、CUDA、开发工具与外部模拟依赖 |
| [MAGIKS 配置](Model/args/README.md) | 当前训练、消融、默认值和评估模板 |
| [Optical-only 配置](optical_only/args/README.md) | 随机初始化基线与 v15/v16 配置 |
| [光学模拟](kn_simulation/README.md) | 双目录准备、Slurm、聚合、恢复与迁移 |
| [运行产物](kn_simulation/runs/README.md) | 目录、manifest、状态及 HDF5 |
| [GW170817A 实验](kn_simulation/gw170817a/README.md) | 固定物理参数的跨观测场景检索 |

仓库曾命名为 `ML+GW+KN`。旧 `dataset/` 目录已移除；GW170817A 工作流位于 `kn_simulation/gw170817a/`。本地历史备份和实验笔记不属于 GitHub checkout 的必要组成部分。

## 环境与路径

完整环境使用 Python 3.10。仓库根目录中执行：

```bash
uv sync --locked
source .venv/bin/activate
```

`requirements.txt` 仅覆盖核心模型依赖，不能替代完整模拟环境。开发依赖、历史 pip 快照、SNANA、OpSim 和 `opsimsummaryv2` 的安装前提见 [ENVIRONMENT.md](ENVIRONMENT.md)。

以下命令均从仓库根目录运行。先区分数据工作区和代码目录：

```bash
export BASE_DIR=/your/data/workspace
cd /path/to/gw-kn-multimodal
export REPO_ROOT="$(pwd)"
```

`BASE_DIR` 在支持它的入口中默认是 `/fred/oz016/bgao_kn`；`REPO_ROOT` 表示代码 checkout。JSON 模板的 `<BASE_DIR>` / `<REPO_ROOT>` 必须先替换；模拟 YAML 的 `${BASE_DIR}` / `${REPO_ROOT}` 由配置加载器展开。部分评估入口另用 `WORKSPACE_ROOT`，部分 Slurm 日志、分区和 notebook 路径仍固定在 OzSTAR；只设置 `BASE_DIR` 不会重写这些内容。

### 外部数据位置

| 用途 | 默认位置或配置来源 |
| --- | --- |
| GW 双目录 | `$BASE_DIR/GWSamplegen/outputs/production_am_bayestar/dual/<source>_<split>_seed_1234/` |
| 正负 skymap | `$BASE_DIR/data/skymap/{positive,negative}/<source>_skymap_<split>/` |
| 模拟聚合 | `$REPO_ROOT/kn_simulation/runs/<profile>/simulation_intermediates.h5` |
| 联合 HDF5 | `$BASE_DIR/data/ALBEF_dataset/` |
| 光学 HDF5 | `$BASE_DIR/data/Optical_Only_dataset/` |
| 多模态外部光学负样本 | `$BASE_DIR/data/Optical_Negative_dataset/` |
| 模型权重 | `$BASE_DIR/data/model/` |
| Rubin / SNANA | YAML profile 中的 OpSim 数据库、SNANA 可执行文件和 SNDATA_ROOT |

`ALBEF_dataset`、checkpoint 内的 `ALBEF/albef_best.pth` 和部分旧配置键继续沿用兼容性命名。训练与评估可能使用不同的负样本文件及 HDF5 group，运行前需逐项核对。Fink RF 比较还需要外部 `fink-rf-reproduction` 生成的模型文件。

<a id="configuration"></a>

## 生成本地配置

Git 跟踪 `*.json.example`，实际运行的 `*.json` 被忽略。设置上面的环境变量后，下面的命令仅从已跟踪模板生成缺失的配置，同时替换两种占位符及 GW170817A 模板中遗留的 OzSTAR 路径前缀；已有配置保持不变：

```bash
python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

repo = Path.cwd()
base = str(Path(os.environ["BASE_DIR"]).expanduser().resolve())

def expand(value):
    if isinstance(value, str):
        for old, new in (
            ("/fred/oz016/bgao_kn/gw-kn-multimodal", str(repo)),
            ("/fred/oz016/bgao_kn", base),
        ):
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
        return value.replace("<BASE_DIR>", base).replace("<REPO_ROOT>", str(repo))
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value

names = subprocess.check_output([
    "git", "ls-files", "-z", "Model/args/*.json.example",
    "optical_only/args/*.json.example",
]).decode().split("\0")
for name in filter(None, names):
    template = repo / name
    target = template.with_suffix("")
    if target.exists():
        continue
    config = expand(json.loads(template.read_text(encoding="utf-8")))
    with target.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
PY
```

拉取更新后应对照模板检查已有 JSON；生成步骤不会自动迁移本地配置。模板中的 checkpoint、目录和 HDF5 group 也不保证已在本机存在。

## 常用工作流

### 1. 生成 Rubin/SNANA 光学模拟

先准备匹配的正负 GW 目录及各自 skymap；支持 `bns_train`、`nsbh_train`、`bns_test` 和 `nsbh_test` 四个生产 profile：

```bash
kn_simulation/bin/kn-sim prepare bns_train \
  --pos-catalog "$BASE_DIR/GWSamplegen/outputs/production_am_bayestar/dual/bns_train_seed_1234/pos_catalog.csv" \
  --neg-catalog "$BASE_DIR/GWSamplegen/outputs/production_am_bayestar/dual/bns_train_seed_1234/neg_catalog.csv"
kn_simulation/bin/kn-sim submit bns_train --dry-run
kn_simulation/bin/kn-sim submit bns_train
kn_simulation/bin/kn-sim status bns_train
```

`prepare` 会写入本地运行目录；`submit` 会提交计算任务。只有筛选后的正目录进入 SNANA，物理双零抛射物负样本单独保留。任务完成后自动聚合为 `simulation_intermediates.h5`。恢复、手动聚合和旧数据迁移见[光学模拟说明](kn_simulation/README.md)。

### 2. 构建联合 HDF5

完成对应 BNS 和 NSBH 的模拟聚合后：

```bash
PROFILE=final_train DATASET_MODE=train \
  bash Model/scripts/data/submit_create_dataset_bns_nsbh.sh
PROFILE=astro_test DATASET_MODE=test \
  bash Model/scripts/data/submit_create_dataset_bns_nsbh.sh
```

输出默认分别为 `combined_dataset_train.h5` 和 `combined_dataset_astro_test.h5`，位于 `data/ALBEF_dataset/`。入口默认读取四个 profile 的聚合 HDF5，同时读取独立负目录；可用 `OUTPUT_H5_PATH`、`NUM_WORKERS`、`BNS_SIM_ARTIFACT` 和 `NSBH_SIM_ARTIFACT` 覆盖路径或资源。

数据区分有可用 KN 光变的正样本、物理双零抛射物 type-1 负样本，以及有抛射物但未形成可用光变的 type-2 负样本。光学行通过 `parent_gw_idx` 关联正 GW；两类 GW 负样本与外部 non-KN 光学候选不是同一种负样本。

### 3. 训练 MAGIKS 与组件消融

```bash
mkdir -p logs/train
bash Model/scripts/train/train.sh Model/args/MAGIKS_BNS_NSBH_full.json

# 批量提交 Full 和五个组件消融，共六个 Slurm 作业
bash Model/scripts/train/submit_ablation_train.sh
```

训练入口合并共享默认配置与实验 JSON，实验值覆盖默认值。当前 Full 使用分阶段训练、混合 KN/non-KN 候选集合和与训练一致的验证条件，并按混合候选检索指标选取 checkpoint。共享默认文件单独使用时的候选集合和指标不同，详见 [MAGIKS 配置说明](Model/args/README.md)。

### 4. 分类评估与检索对照

```bash
bash Model/scripts/eval/submit_test_evaluate.sh \
  Model/args/eval/cls_test/MAGIKS_BNS_NSBH_eval_full.json
bash Model/scripts/eval/submit_retrieval_comparison.sh \
  Model/args/eval/retrieval_comparison.json
```

当前检索模板比较 Full、五个消融、Optical-only baseline、Skymap-only 和 Fink Random Forest 共九个方法；候选集合采用 `synthetic_time_sky_hard` non-KN 干扰源。该测试协议与训练时的 `mixed_kn_nonkn` 候选集合不同。运行前核对全部模型路径；红移分析模板的 catalog 路径仍指向早期 `production_rubin_dual` 数据，须确认与测试 HDF5 匹配后再使用。

独立 mixed-retrieval 和 fixed-checkpoint attribution 脚本已移除，旧配置或结果名称不代表当前可执行入口。当前可用模板见[评估配置索引](Model/args/README.md#evaluation)。

### 5. GW-KN 单参数物理敏感性

```bash
DRY_RUN=true bash Model/scripts/eval/submit_gw_kn_pairing_sensitivity.sh \
  Model/args/eval/gw_kn_single_parameter_sensitivity.json
bash Model/scripts/eval/submit_gw_kn_pairing_sensitivity.sh \
  Model/args/eval/gw_kn_single_parameter_sensitivity.json
```

该配置在同类 BNS 或 NSBH 事件间构造交叉配对，控制非目标参数差异，以共享天空位置和 GW 到首次探测时间差比较 interaction 与 directional-win。光变亮度、颜色、误差及内部时间演化保持原样。输出包括配对清单、逐对分数、bootstrap/permutation 统计和参数趋势图。校验模式不提交任务，但 shell 包装器会创建日志目录。

Physics Ejecta Bridge 的训练集拟合入口仍可用：

```bash
bash Model/scripts/eval/submit_physics_ejecta_bridge_fit.sh \
  Model/args/eval/physics_ejecta_bridge_fit.json
```

Bridge 使用训练集拟合 GW/光学特征的近邻后验，以共同物理空间中的分布重叠评分；它是经验物理参考。拟合配置中的测试数据仅用于重叠检查。directional-win v2 实现仍存在，但当前不提供对应的 `.json.example`：需另行准备模型、冻结 Bridge artifact、配对设置及预期 manifest 校验值一致的配置，不能直接用上述 v1 模板替代。详见[配置说明](Model/args/README.md#evaluation)。

### 6. Optical-only 数据与训练

```bash
DATASET_MODE=train BUILD_POSITIVE=true BUILD_NEGATIVE=true \
  bash optical_only/scripts/data/submit_create_datasets.sh
bash optical_only/scripts/train/train.sh optical_only/args/optical_only_kn_baseline.json

# 需先准备 MAGIKS Full checkpoint、评估数据及时间偏移分布
bash optical_only/scripts/train/train.sh optical_only/args/optical_only_kn_v16.json
```

光学数据预处理包括首次探测对齐、2 小时同波段合并和 luptitude 变换。随机初始化 baseline 用于独立对照；v16 从 MAGIKS Full 的光学编码器初始化，采用单视图 prefix 微调，不启用对抗头或 GRL。训练包装器会继续调用评估，因此评估输入也需提前就绪。见 [optical-only 配置说明](optical_only/args/README.md)。

### 7. GW170817A LSST 场景检索

```bash
WORKSPACE_ROOT="$(dirname "$REPO_ROOT")" \
  bash kn_simulation/gw170817a/submit_gw170817a_lsst_pipeline.sh
bash Model/scripts/eval/submit_gw170817a_retrieval.sh \
  Model/args/eval/retrieval_gw170817a_lsst.json
```

数据构建完成后再提交检索。该构建包装器要求代码目录名为 `gw-kn-multimodal`，并用 `WORKSPACE_ROOT` 指定其父目录。此实验固定 GW170817A 的物理参数与宿主距离，比较 10 个 Rubin 观测场景，每个场景包含共享的 50 个坐标，形成 500 条光变；不是红移扫描。输入、分阶段执行和统计方法见 [GW170817A 说明](kn_simulation/gw170817a/README.md)。

## 检查与复现

安装开发依赖后，从仓库根目录运行轻量配置与入口检查：

```bash
python -m pytest tests/config/test_config_templates.py tests/config/test_optical_only_layout.py
```

当前 `retrieval_gw170817a_lsst.json.example` 仍含硬编码路径，会导致模板可移植性子检查失败。上面的配置生成示例只转换生成的本地 JSON，不修改原模板。部分检查还涉及本地历史配置或科学计算依赖，不能将配置检查等同于完整科学流程验证。

完整测试位于 `tests/`。保存实际运行 JSON、数据版本及 checkpoint 来源，并核对 Slurm 分区、资源、日志目录和节点临时存储。绘图脚本可能写入相邻的论文目录，重绘前先核对输出位置。

## 数据、引用与许可

数据需通过外部 GWSamplegen 与本仓库模拟/构建流程生成；预构建数据集请联系作者。论文仍在准备中，引用信息见 [CITATION.cff](CITATION.cff)。本项目采用 [MIT License](LICENSE)。
