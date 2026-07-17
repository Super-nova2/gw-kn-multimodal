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
- 多模态 MAGIKS / 对比学习训练，以及 optical-only 分类基线。

## 目录结构

```text
gw-kn-multimodal/
├── Model/
│   ├── data_loader.py                  # HDF5 读取与采样器
│   ├── model.py                        # MAGIKS 模型与编码器
│   ├── args/                           # 当前配置与可移植模板
│   └── scripts/
│       ├── train/                      # 训练入口
│       ├── eval/                       # 评估、检索与测速
│       ├── data/                       # 数据集构建
│       └── hpo/                        # 超参数优化
├── optical_only/
│   ├── args/                           # 当前配置模板与历史配置
│   ├── scripts/                        # train / eval / data / analysis / plot
│   └── notebooks/                      # optical-only 分析 notebook
├── dataset/
│   ├── O5_sim_*                        # 模拟事件与分析材料
│   └── KN_sim/                         # SNANA / OpSim 工作流脚本
└── docs/                               # 设计说明与实验记录
```

## 依赖与运行环境

Python 依赖见 `requirements.txt`，可通过以下命令安装：

```bash
pip install -r requirements.txt
```

主要依赖：Python 3.10+、PyTorch、h5py、numpy、pandas、astropy、healpy、ligo.skymap、tqdm、matplotlib、optuna、tensorboard、graphviz。

额外系统依赖：jq、Slurm（HPC 环境）、SNANA + opsimsummaryv2（仅数据生成流程需要）。

建议在 HPC/Slurm 环境中运行。现有 `.sh` 入口脚本默认就是按 Slurm 提交和资源申请来写的。

## 数据与路径约定

### 环境变量 `BASE_DIR`

所有 shell 脚本和 Python 源代码通过环境变量 `BASE_DIR` 确定数据根目录。如果未设置，默认值为 `/fred/oz016/bgao_kn`。

```bash
export BASE_DIR=/your/data/root
```

### JSON 配置文件（`.json.example` 模板）

配置文件采用模板模式：

- `*.json.example` 是跟踪在 Git 中的模板文件，路径使用 `<BASE_DIR>` 占位符
- `*.json` 是实际运行时使用的配置文件（已被 `.gitignore` 忽略）

首次使用时，复制模板并替换占位符：

```bash
cd <BASE_DIR>/gw-kn-multimodal
REPO_ROOT="$(pwd)"
find Model/args -name '*.json.example' -print0 | while IFS= read -r -d '' f; do
    sed -e "s|<BASE_DIR>|$BASE_DIR|g" -e "s|<REPO_ROOT>|$REPO_ROOT|g" "$f" > "${f%.example}"
done

# optical_only/args/ 和 dataset/KN_sim/ 下同理
```

### 数据目录结构

| 用途 | 路径 |
|------|------|
| 多模态训练 HDF5 | `$BASE_DIR/data/ALBEF_dataset/` |
| optical-only 训练 HDF5 | `$BASE_DIR/data/Optical_Only_dataset/` |
| 模型 checkpoint | `$BASE_DIR/data/model/` |
| skymap / SNANA 数据 | `$BASE_DIR/data/` 和 `$BASE_DIR/SNANA/` |

`ALBEF_dataset` 是现有外部数据存储路径，为兼容既有 HDF5 本轮不重命名。

## 常用工作流

### 1. 构建 GW + Optical 联合数据集

这个流程会把 BNS / NSBH 的 GW 参数、skymap 和光学光变整理到同一个 HDF5 中。

```bash
cd <BASE_DIR>/gw-kn-multimodal

PROFILE=final_train DATASET_MODE=train \
bash Model/scripts/data/submit_create_dataset_bns_nsbh.sh
```

常用环境变量：

- `PROFILE=test_aug|final_train`
- `DATASET_MODE=train|test`
- `OUTPUT_H5_PATH=/path/to/output.h5`
- `NUM_WORKERS=6`

主脚本：

- `Model/scripts/data/submit_create_dataset_bns_nsbh.sh`
- `Model/scripts/data/create_dataset_bns_nsbh.py`

### 2. 训练多模态 GW + Optical 模型

推荐直接用现成 JSON 配置提交：

```bash
cd <BASE_DIR>/gw-kn-multimodal

bash Model/scripts/train/train.sh Model/args/MAGIKS_BNS_NSBH_full.json
```

相关文件：

- `Model/scripts/train/train.py`
- `Model/model.py`
- `Model/data_loader.py`
- `Model/args/MAGIKS_BNS_NSBH_full.json`

训练脚本会从 JSON 中读取数据路径、负样本路径、时间偏移设置、模型超参数和 checkpoint 目录。

### 3. 评估多模态模型

```bash
cd <BASE_DIR>/gw-kn-multimodal

bash Model/scripts/eval/submit_test_evaluate.sh /path/to/eval_args.json
```

评估入口：

- `Model/scripts/eval/evaluate.py`
- `Model/scripts/eval/submit_test_evaluate.sh`

支持检索、分类、OOD 监控和负样本时间偏移评估。

### 4. 构建 optical-only 数据集

这个流程会把光变序列做 first-detection 对齐、2 小时同波段合并、luptitude 变换，并输出 optical-only HDF5。

```bash
cd <BASE_DIR>/gw-kn-multimodal

DATASET_MODE=train \
BUILD_POSITIVE=true \
BUILD_NEGATIVE=true \
bash optical_only/scripts/data/submit_create_datasets.sh
```

主脚本：

- `optical_only/scripts/data/submit_create_datasets.sh`
- `optical_only/scripts/data/create_datasets.py`

### 5. 训练 optical-only 基线

```bash
cd <BASE_DIR>/gw-kn-multimodal

bash optical_only/scripts/train/train.sh optical_only/args/optical_only_kn_v16.json
```

相关文件：

- `optical_only/scripts/train/train.py`
- `optical_only/scripts/eval/evaluate.py`
- `optical_only/args/optical_only_kn_v16.json`

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
- 迁移环境前需重新从 `.json.example` 模板生成配置文件。
- `dataset/KN_sim/` 下的脚本依赖外部 SNANA、OpSim 数据和数据库文件。
- 部分 notebook 与实验文档保留了研究期的路径习惯，跑之前建议先核对。

## 建议起步顺序

1. 安装依赖：`pip install -r requirements.txt`
2. 设置环境变量：`export BASE_DIR=/your/data/root`
3. 从 `.json.example` 模板生成本地配置文件（见上方说明）
4. 确认 `$BASE_DIR/data/` 下的 HDF5、skymap、SNANA 数据是否齐全
5. 先跑数据构建，再跑训练，再跑评估

## 数据获取

本仓库不包含训练数据和大型模拟文件。数据集需通过以下方式获取：

- **GW 模拟事件**：使用 `dataset/KN_sim/` 下的脚本配合 SNANA/OpSim 生成。
- **训练 HDF5**：使用 `Model/scripts/data/create_dataset_bns_nsbh.py` 和 `optical_only/scripts/data/create_datasets.py` 从模拟数据构建。
- 如需预构建数据集，请联系作者。

## Citation

如果你使用了本仓库的代码，请引用我们的论文：

```
[Paper in preparation]
```

详见 `CITATION.cff`。

## License

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。
