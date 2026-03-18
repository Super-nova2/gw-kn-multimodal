# gw-kn-multimodal

`gw-kn-multimodal` 是一个面向千新星（kilonova, KN）识别的多模态研究仓库，覆盖了从模拟数据生成、HDF5 数据集构建，到 GW+光学联合训练、optical-only 基线训练与评估的完整流程。

旧目录名是 `ML+GW+KN`。本次已改成不含 `+` 的名字，避免路径和导入上的额外兼容性问题。

## 仓库目标

这个仓库主要解决三类问题：

1. 把引力波事件信息和光学测光序列整理成统一的训练数据集。
2. 训练 GW + optical 的联合模型，用于 KN 配对、检索和分类。
3. 训练不依赖 GW 输入的 optical-only 基线，并分析它在真实/模拟负样本上的泛化表现。

当前代码里最核心的研究对象包括：

- BNS / NSBH 模拟事件及其对应的 kilonova 光变。
- GW 标量参数与 skymap 表示。
- 基于 luptitude、first-detection 对齐和时间偏移建模的光学序列表示。
- 多模态 ALBEF / 对比学习训练，以及 optical-only 分类基线。

## 目录结构

```text
gw-kn-multimodal/
├── Model/
│   ├── ALBEF_train.py/.sh              # 多模态训练主入口
│   ├── data_loader.py                  # HDF5 读取与采样器
│   ├── model.py                        # GW / optical 编码器与分类头
│   ├── args/                           # 训练与 HPO 配置
│   └── script/
│       ├── create_dataset_bns_nsbh.py
│       ├── submit_create_dataset_bns_nsbh.sh
│       └── submit_test_evaluate.sh
├── optical_only/
│   ├── create_optical_only_datasets.py # optical-only 数据集构建
│   ├── train_optical_only.py/.sh       # optical-only 训练与自动评估
│   ├── test_evaluate_optical_only.py   # optical-only 评估
│   └── args/                           # optical-only 配置
├── dataset/
│   ├── O5_sim_*                        # 模拟事件与分析材料
│   └── KN_sim/                         # SNANA / OpSim 工作流脚本
└── docs/                               # 设计说明与实验记录
```

## 依赖与运行环境

仓库没有单独维护 `requirements.txt`，当前代码实际依赖主要包括：

- Python 3.10+
- PyTorch
- h5py
- numpy
- pandas
- astropy
- healpy
- ligo.skymap
- tqdm
- matplotlib
- optuna
- tensorboard
- graphviz
- jq
- Slurm
- SNANA
- opsimsummaryv2

建议在 HPC/Slurm 环境中运行。现有 `.sh` 入口脚本默认就是按 Slurm 提交和资源申请来写的。

## 数据与路径约定

当前示例配置默认使用 `/fred/oz016/bgao_kn/data/...` 下的数据与输出目录，例如：

- 多模态训练 HDF5：`/fred/oz016/bgao_kn/data/ALBEF_dataset/`
- optical-only 训练 HDF5：`/fred/oz016/bgao_kn/data/Optical_Only_dataset/`
- checkpoint：`/fred/oz016/bgao_kn/data/model/`
- skymap / SNANA 数据：`/fred/oz016/bgao_kn/data/` 和 `/fred/oz016/bgao_kn/SNANA/`

如果你在别的机器或目录运行，需要优先修改：

- `Model/args/*.json`
- `optical_only/args/*.json`
- 通过环境变量覆盖的提交脚本参数

## 常用工作流

### 1. 构建 GW + Optical 联合数据集

这个流程会把 BNS / NSBH 的 GW 参数、skymap 和光学光变整理到同一个 HDF5 中。

```bash
cd /fred/oz016/bgao_kn/gw-kn-multimodal

PROFILE=final_train DATASET_MODE=train \
bash Model/script/submit_create_dataset_bns_nsbh.sh
```

常用环境变量：

- `PROFILE=test_aug|final_train`
- `DATASET_MODE=train|test`
- `OUTPUT_H5_PATH=/path/to/output.h5`
- `NUM_WORKERS=6`

主脚本：

- `Model/script/submit_create_dataset_bns_nsbh.sh`
- `Model/script/create_dataset_bns_nsbh.py`

### 2. 训练多模态 GW + Optical 模型

推荐直接用现成 JSON 配置提交：

```bash
cd /fred/oz016/bgao_kn/gw-kn-multimodal

bash Model/ALBEF_train.sh Model/args/ALBEF_BNS_NSBH.json
```

相关文件：

- `Model/ALBEF_train.py`
- `Model/model.py`
- `Model/data_loader.py`
- `Model/args/ALBEF_BNS_NSBH.json`

训练脚本会从 JSON 中读取数据路径、负样本路径、时间偏移设置、模型超参数和 checkpoint 目录。

### 3. 评估多模态模型

```bash
cd /fred/oz016/bgao_kn/gw-kn-multimodal

bash Model/script/submit_test_evaluate.sh /path/to/eval_args.json
```

评估入口：

- `Model/test_evaluate.py`
- `Model/script/submit_test_evaluate.sh`

支持检索、分类、OOD 监控和负样本时间偏移评估。

### 4. 构建 optical-only 数据集

这个流程会把光变序列做 first-detection 对齐、2 小时同波段合并、luptitude 变换，并输出 optical-only HDF5。

```bash
cd /fred/oz016/bgao_kn/gw-kn-multimodal

DATASET_MODE=train \
BUILD_POSITIVE=true \
BUILD_NEGATIVE=true \
bash optical_only/submit_create_optical_only_datasets.sh
```

主脚本：

- `optical_only/submit_create_optical_only_datasets.sh`
- `optical_only/create_optical_only_datasets.py`

### 5. 训练 optical-only 基线

```bash
cd /fred/oz016/bgao_kn/gw-kn-multimodal

bash optical_only/train_optical_only.sh optical_only/args/optical_only_kn_v14.json
```

相关文件：

- `optical_only/train_optical_only.py`
- `optical_only/test_evaluate_optical_only.py`
- `optical_only/args/optical_only_kn_v14.json`

该脚本训练结束后会自动调用 optical-only 评估脚本。

## 配置文件说明

配置主要放在两个目录：

- `Model/args/`
- `optical_only/args/`

你通常只需要复制一个现成 JSON，再改这些字段：

- 数据路径：`data_path` / `pos_data_path` / `neg_data_path`
- 输出路径：`ckpt_path` / `output_dir`
- batch / epoch / worker 数
- 时间偏移建模相关参数
- prefix / universal / OOD / shortcut audit 等实验开关

## 注意事项

- 很多脚本默认依赖 Slurm；本地直接运行时也会优先尝试 `sbatch` 自提交。
- 现有示例配置大量使用绝对路径，迁移环境前要先检查 JSON。
- `dataset/KN_sim/` 下的脚本依赖外部 SNANA、OpSim 数据和数据库文件。
- 部分 notebook 与实验文档保留了研究期的路径习惯，跑之前建议先核对。

## 建议起步顺序

1. 先确认 `/fred/oz016/bgao_kn/data/` 下的 HDF5、skymap、SNANA 数据是否齐全。
2. 再检查 `Model/args/ALBEF_BNS_NSBH.json` 或 `optical_only/args/optical_only_kn_v14.json`。
3. 先跑数据构建，再跑训练，再跑评估。
