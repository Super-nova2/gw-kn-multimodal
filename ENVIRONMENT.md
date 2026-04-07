# Python Environment Setup — bgao_kn

> **Language / 语言**: [中文](#中文版) | [English](#english-version)

---

<a id="中文版"></a>

## 中文版

本项目的 Python 环境用于千新星（kilonova）多信使天文学研究，包含 GW 数据分析、Rubin/LSST 巡天模拟、机器学习等工具链。

### 环境概览

- **Python**: >=3.10, <3.11
- **核心依赖**: lalsuite, ligo.skymap, astropy, torch (CUDA 12.6), fink-client 等
- **管理工具**: [uv](https://docs.astral.sh/uv/) (推荐) 或 pip

### 需要分发的文件

在任意设备上重建环境，只需要以下文件：

| 文件 | 用途 |
|---|---|
| `pyproject.toml` | 项目依赖声明（直接依赖 + uv 配置） |
| `uv.lock` | 锁文件，包含所有依赖的精确版本（方法一使用） |
| `requirements-rubin.txt` | pip freeze 导出的完整依赖列表（方法二/三使用） |

将这三个文件复制到目标目录即可开始重建。

---

### 方法一：使用 `uv sync`（推荐）

从 `uv.lock` 锁文件精确重建环境，保证完全可复现。

#### 前提

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

需要系统中有 Python 3.10。如果没有，uv 可以自动下载：

```bash
uv python install 3.10
```

在 OzSTAR 集群上，也可以使用系统模块：

```bash
module load Python/3.10.4-GCCcore-11.3.0
```

#### 重建步骤

```bash
# 进入包含 pyproject.toml 和 uv.lock 的目录
cd /path/to/your/project

# 从 lockfile 创建 .venv 并安装所有依赖
uv sync
```

环境将创建在 `.venv/` 目录下。激活方式：

```bash
source .venv/bin/activate
```

或直接使用 `uv run` 执行脚本（无需手动激活）：

```bash
uv run python your_script.py
uv run jupyter lab
```

#### 添加 / 移除依赖

```bash
uv add <package-name>        # 自动更新 pyproject.toml 和 uv.lock
uv remove <package-name>     # 移除依赖
```

---

### 方法二：使用 `uv pip install`（从 requirements 文件）

不使用 project 模式，直接从冻结的依赖列表安装。适合只需快速搭建、不打算管理依赖变更的场景。

```bash
# 创建虚拟环境（uv 会自动查找或下载 Python 3.10）
uv venv myenv --python 3.10

# 安装所有依赖
uv pip install -r requirements-rubin.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    --python myenv/bin/python

# 激活
source myenv/bin/activate
```

> **说明**: `--extra-index-url` 和 `--index-strategy` 是 PyTorch CUDA 版本所需的参数。

---

### 方法三：使用 `pip`（备选，无需 uv）

```bash
# 确保有 Python 3.10，然后创建虚拟环境
python3.10 -m venv myenv
source myenv/bin/activate

# 安装所有依赖
pip install -r requirements-rubin.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

> **说明**: pip 安装速度较 uv 慢很多（约 10-100x），但不需要额外安装任何工具。

---

### 特殊包处理

#### opsimsummaryv2

`opsimsummaryv2` (作者: Bastien Carreres) 不在 PyPI 上，未包含在 `pyproject.toml` 的依赖中。
使用方法一或方法二重建环境后，需要手动安装：

```bash
# 如果有源码目录
uv pip install -e /path/to/opsimsummaryv2/   # uv 方式
pip install -e /path/to/opsimsummaryv2/       # pip 方式
```

该包的依赖（astropy, healpy, numpy, pandas, scikit-learn, sqlalchemy）已全部包含在环境中。

#### fink-client

`fink-client` 从 GitHub 仓库的指定 commit 直接安装，已在 `pyproject.toml` 和 `requirements-rubin.txt` 中配置，无需额外操作。

---

### 常见问题

**Q: `uv sync` 时提示找不到 Python 3.10？**
A: 运行 `uv python install 3.10`，uv 会自动下载并管理 Python 版本。

**Q: PyTorch 安装失败？**
A: 确保能访问 `https://download.pytorch.org/whl/cu126`。该 index 已在 `pyproject.toml` 中配置。如不需要 GPU，可将 `pyproject.toml` 中的 `cu126` 改为 `cpu`。

**Q: CUDA 显示不可用 (`torch.cuda.is_available()` 返回 False)？**
A: 需要在有 NVIDIA GPU 的机器上运行。在 HPC 集群中，需要通过作业调度器（如 Slurm）分配 GPU 节点。

**Q: 不需要 GPU / 只需要 CPU 版本的 PyTorch？**
A: 将 `pyproject.toml` 中 `[[tool.uv.index]]` 的 URL 改为 `https://download.pytorch.org/whl/cpu`，然后重新 `uv lock && uv sync`。

---

<a id="english-version"></a>

## English Version

This Python environment is designed for kilonova multi-messenger astronomy research, including GW data analysis, Rubin/LSST survey simulations, and machine learning pipelines.

### Environment Overview

- **Python**: >=3.10, <3.11
- **Key dependencies**: lalsuite, ligo.skymap, astropy, torch (CUDA 12.6), fink-client, etc.
- **Package manager**: [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Files to Distribute

To rebuild the environment on any machine, you only need these files:

| File | Purpose |
|---|---|
| `pyproject.toml` | Project dependency declaration (direct deps + uv config) |
| `uv.lock` | Lock file with exact versions of all dependencies (for Method 1) |
| `requirements-rubin.txt` | Frozen dependency list exported via pip freeze (for Method 2/3) |

Copy these three files to your target directory to get started.

---

### Method 1: `uv sync` (Recommended)

Rebuild the environment from the `uv.lock` lock file for fully reproducible results.

#### Prerequisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Python 3.10 is required. If not available on your system, uv can download it automatically:

```bash
uv python install 3.10
```

On the OzSTAR cluster, you can also use the system module:

```bash
module load Python/3.10.4-GCCcore-11.3.0
```

#### Rebuild Steps

```bash
# Navigate to the directory containing pyproject.toml and uv.lock
cd /path/to/your/project

# Create .venv and install all dependencies from the lock file
uv sync
```

The environment will be created in the `.venv/` directory. To activate:

```bash
source .venv/bin/activate
```

Or use `uv run` to execute scripts directly (no manual activation needed):

```bash
uv run python your_script.py
uv run jupyter lab
```

#### Adding / Removing Dependencies

```bash
uv add <package-name>        # Automatically updates pyproject.toml and uv.lock
uv remove <package-name>     # Remove a dependency
```

---

### Method 2: `uv pip install` (From requirements file)

Install directly from the frozen dependency list without using project mode. Suitable for quick setup when you don't plan to manage dependency changes.

```bash
# Create a virtual environment (uv will find or download Python 3.10 automatically)
uv venv myenv --python 3.10

# Install all dependencies
uv pip install -r requirements-rubin.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    --python myenv/bin/python

# Activate
source myenv/bin/activate
```

> **Note**: `--extra-index-url` and `--index-strategy` are required for the PyTorch CUDA build.

---

### Method 3: `pip` (Fallback, no uv needed)

```bash
# Make sure Python 3.10 is available, then create a virtual environment
python3.10 -m venv myenv
source myenv/bin/activate

# Install all dependencies
pip install -r requirements-rubin.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

> **Note**: pip is significantly slower than uv (~10-100x), but requires no additional tools.

---

### Special Packages

#### opsimsummaryv2

`opsimsummaryv2` (author: Bastien Carreres) is not available on PyPI and is not included in `pyproject.toml`.
After rebuilding the environment via any method above, install it manually:

```bash
# If you have the source code
uv pip install -e /path/to/opsimsummaryv2/   # uv
pip install -e /path/to/opsimsummaryv2/       # pip
```

All of its dependencies (astropy, healpy, numpy, pandas, scikit-learn, sqlalchemy) are already included in the environment.

#### fink-client

`fink-client` is installed directly from a pinned GitHub commit. It is already configured in both `pyproject.toml` and `requirements-rubin.txt` — no extra steps needed.

---

### FAQ

**Q: `uv sync` says it can't find Python 3.10?**
A: Run `uv python install 3.10` — uv will download and manage the Python version for you.

**Q: PyTorch installation fails?**
A: Make sure `https://download.pytorch.org/whl/cu126` is accessible. The index is already configured in `pyproject.toml`. If you don't need a GPU, change `cu126` to `cpu` in the index URL.

**Q: CUDA is not available (`torch.cuda.is_available()` returns False)?**
A: You need to run on a machine with an NVIDIA GPU. On HPC clusters, allocate a GPU node via the job scheduler (e.g., Slurm).

**Q: Don't need GPU / want CPU-only PyTorch?**
A: Change the `[[tool.uv.index]]` URL in `pyproject.toml` to `https://download.pytorch.org/whl/cpu`, then run `uv lock && uv sync`.
