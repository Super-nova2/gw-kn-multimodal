# Python Environment Setup: gw-kn-multimodal

[中文](#中文版) | [English](#english-version) | [项目说明 / Project README](README.md)

<a id="中文版"></a>

## 中文版

### 选择安装方式

本仓库是脚本与研究环境集合。完整环境由 `pyproject.toml` 和 `uv.lock` 定义，要求 **Python >=3.10,<3.11**；锁文件针对 Python 3.10，PyTorch 使用 CUDA 12.6 wheel。安装 Python 依赖不会安装外部数据、checkpoint、SNANA 或 Slurm。

| 文件 | 用途与限制 |
| --- | --- |
| [requirements.txt](requirements.txt) | 核心模型的最小依赖，多数未固定版本；不覆盖完整模拟和开发工具链 |
| [pyproject.toml](pyproject.toml) + [uv.lock](uv.lock) | 完整研究依赖、CUDA 索引和 `dev` 可选依赖；推荐以锁文件重建 |
| [requirements-rubin.txt](requirements-rubin.txt) | 历史 `rubin` 环境快照，包含 `opsimsummaryv2==0.1`；与当前锁文件不是同一组版本 |

例如，当前锁文件包含 torch `2.11.0+cu126` / torchvision `0.26.0+cu126`，历史快照固定为 `2.9.1+cu126` / `0.24.1+cu126`。请选择一种安装来源；不要把历史快照叠加到锁文件环境后仍视为同一个环境。

### 方法一：完整锁定环境

在安装了 uv 的机器上，从仓库根目录运行：

```bash
uv python install 3.10
uv sync --locked
source .venv/bin/activate
```

`uv sync --locked` 要求依赖声明与锁文件一致，不更新锁文件。OzSTAR 上也可先加载已有的 Python 3.10 模块。环境创建在仓库的 `.venv/`；现有工作区中的 `rubin/` 是另一个环境，不会被自动激活。

需要 pytest、Ruff 和 Black 时，启用已声明的 `dev` extra：

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest tests/config/test_config_templates.py tests/config/test_optical_only_layout.py
```

完整测试命令是 `python -m pytest`；部分测试依赖科学计算包或本地历史配置。此处的配置检查不会提交训练或模拟作业。

### 方法二：最小模型环境

仅做核心模型开发、且不需要完整研究工具链时，可在独立环境中安装：

```bash
python3.10 -m venv .venv-minimal
source .venv-minimal/bin/activate
python -m pip install -r requirements.txt
python -m pip install 'pytest>=7' ruff black
```

该依赖表不固定 PyTorch CUDA 构建，也不包括模拟入口所需的全部依赖。使用 SNANA、Bridge 拟合或完整分析工具链时，优先采用方法一。

### 方法三：历史 rubin 快照

此方式需要外部 `opsimsummaryv2` 源码，且源码包版本须满足快照中的 `0.1`。它不是仓库自带文件。以下 `/path/to/opsimsummaryv2` 必须替换为实际源码路径：

```bash
python3.10 -m venv .venv-rubin-snapshot
source .venv-rubin-snapshot/bin/activate
python -m pip install -e /path/to/opsimsummaryv2 \
  -r requirements-rubin.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

也可用 uv 的 pip 接口在单独环境中安装同一快照：

```bash
uv venv .venv-rubin-snapshot --python 3.10
uv pip install --python .venv-rubin-snapshot/bin/python \
  -e /path/to/opsimsummaryv2 -r requirements-rubin.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match
```

直接安装快照而不提供 `opsimsummaryv2` 的可用来源可能失败。快照和项目依赖中的 `fink-client` 都指向同一个 Git commit，需要 Git 以及对该源码仓库的访问。

### 模拟与集群依赖

`opsimsummaryv2` 未列入项目的直接依赖。采用方法一后，模拟用户需另外安装其源码到同一个 Python 环境：

```bash
uv pip install --python .venv/bin/python -e /path/to/opsimsummaryv2
```

之后使用会精确同步环境的 `uv sync` 时，需检查这个额外安装的包是否仍存在。模拟流程还需要：

- 外部 GWSamplegen 正负目录、各自的 skymap 和完整数据来源记录。
- Rubin baseline v5.1 OpSim 数据库，以及 profile 引用的 ToO 配置。
- SNANA 的 `snlc_sim.exe`、SNDATA_ROOT 模型和标定数据；Python 安装不能替代这些文件。
- Slurm 的 `sbatch` 等命令、可用分区和资源配额；部分训练/评估包装器还依赖 `jq`。
- 可写的节点临时目录。生产模拟要求 `SLURM_TMPDIR` 或 `JOBFS`，当前 profile 请求 5 GiB；模型任务的 HDF5 暂存要求另见对应脚本。

激活环境后再提交作业，包装器调用的是作业环境中的 `python`。GPU 可用性应在已分配的 GPU 节点检查，登录节点返回 False 不代表依赖安装失败：

```bash
python -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pip check
```

CPU-only 环境需单独选择兼容的 PyTorch 安装方案。锁文件和历史快照都包含 CUDA 构建，仅把索引 URL 改为 CPU 不会使现有 CUDA 版本约束变成 CPU 环境。

### 路径与配置

见 [README 配置生成步骤](README.md#configuration)。`BASE_DIR` 表示数据工作区，`REPO_ROOT` 表示代码根目录；JSON 占位符必须在运行前替换。模拟 YAML 会自动展开这两个变量；部分评估包装器使用 `WORKSPACE_ROOT`，默认也是 `/fred/oz016/bgao_kn`。

仍有硬编码的 OzSTAR 日志目录、Slurm 分区和论文输出位置。可在运行前只读搜索：

```bash
rg -n '/fred/oz016/bgao_kn|WORKSPACE_ROOT|#SBATCH' Model optical_only kn_simulation plots_scripts
```

各入口支持的资源覆盖变量并不相同，需查看所用 shell 脚本。外部数据、runtime JSON、日志目录、模型权重和模拟可执行文件全部就绪后，再提交计算任务。

---

<a id="english-version"></a>

## English Version

### Choose an Installation Source

This repository contains scripts and a research environment. The full environment is defined by `pyproject.toml` and `uv.lock` and requires **Python >=3.10,<3.11**. The lock targets Python 3.10 and CUDA 12.6 PyTorch wheels. Installing Python dependencies does not install datasets, checkpoints, SNANA or Slurm.

| File | Purpose and limits |
| --- | --- |
| [requirements.txt](requirements.txt) | Minimal core-model dependencies, mostly unpinned; not the full simulation or development toolchain |
| [pyproject.toml](pyproject.toml) + [uv.lock](uv.lock) | Full research dependencies, CUDA index and optional `dev` tools; preferred for rebuilding |
| [requirements-rubin.txt](requirements-rubin.txt) | Historical `rubin` snapshot including `opsimsummaryv2==0.1`; versions differ from the current lock |

For example, the current lock contains torch `2.11.0+cu126` / torchvision `0.26.0+cu126`, while the snapshot pins `2.9.1+cu126` / `0.24.1+cu126`. Choose one installation source. Installing the snapshot over a locked environment does not preserve that locked environment.

### Method 1: Full Locked Environment

With uv installed, run from the repository root:

```bash
uv python install 3.10
uv sync --locked
source .venv/bin/activate
```

`uv sync --locked` requires the dependency declaration and lock to agree without updating the lock. On OzSTAR, an available Python 3.10 module can also supply the interpreter. The environment is created at `.venv/` in this repository; an existing workspace `rubin/` is a separate environment and is not automatically activated.

Enable the declared `dev` extra for pytest, Ruff and Black:

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest tests/config/test_config_templates.py tests/config/test_optical_only_layout.py
```

Run the full suite with `python -m pytest`. Some tests depend on scientific packages or local historical configurations. The configuration checks above do not submit training or simulation jobs.

### Method 2: Minimal Model Environment

For core-model development without the complete research toolchain, use a separate environment:

```bash
python3.10 -m venv .venv-minimal
source .venv-minimal/bin/activate
python -m pip install -r requirements.txt
python -m pip install 'pytest>=7' ruff black
```

This list does not pin a PyTorch CUDA build or cover all simulation dependencies. Prefer Method 1 for SNANA workflows, Bridge fitting or the full analysis toolchain.

### Method 3: Historical rubin Snapshot

This requires an external `opsimsummaryv2` source checkout whose package version satisfies the snapshot's `0.1` pin. It is not bundled in this repository. Replace `/path/to/opsimsummaryv2` with the actual source directory:

```bash
python3.10 -m venv .venv-rubin-snapshot
source .venv-rubin-snapshot/bin/activate
python -m pip install -e /path/to/opsimsummaryv2 \
  -r requirements-rubin.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

Alternatively, install the same snapshot in a separate environment through uv's pip interface:

```bash
uv venv .venv-rubin-snapshot --python 3.10
uv pip install --python .venv-rubin-snapshot/bin/python \
  -e /path/to/opsimsummaryv2 -r requirements-rubin.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match
```

Installing the snapshot without an available `opsimsummaryv2` source can fail. Both the snapshot and project declaration obtain `fink-client` from the same Git commit, requiring Git and access to its source repository.

### Simulation and Cluster Dependencies

`opsimsummaryv2` is absent from the project's direct dependencies. After Method 1, simulation users must install its source into the same interpreter:

```bash
uv pip install --python .venv/bin/python -e /path/to/opsimsummaryv2
```

After subsequent exact environment synchronization with `uv sync`, check that this additional package is still installed. Simulation also requires:

- External GWSamplegen positive/negative catalogs, their separate skymaps and bundle provenance.
- The Rubin baseline v5.1 OpSim database and the ToO configuration referenced by the profile.
- SNANA `snlc_sim.exe`, SNDATA_ROOT models and calibration data; Python installation does not supply them.
- Slurm commands such as `sbatch`, suitable partitions and resources; some training/evaluation wrappers also need `jq`.
- Writable node-local storage. Production simulation requires `SLURM_TMPDIR` or `JOBFS` and currently requests 5 GiB; model HDF5 staging requirements are specified by each launcher.

Activate the environment before submission: wrappers call `python` from the job environment. Check GPU availability inside a GPU allocation; False on a login node does not by itself indicate a broken installation:

```bash
python -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pip check
```

A CPU-only environment requires a separate, compatible PyTorch installation choice. Both the lock and snapshot contain CUDA builds; changing an index URL alone does not convert CUDA version constraints into a CPU environment.

### Paths and Configuration

See [runtime configuration generation](README.md#configuration). `BASE_DIR` denotes the data workspace and `REPO_ROOT` the code root. Replace JSON placeholders before use; simulation YAML expands these variables automatically. Some evaluation wrappers instead use `WORKSPACE_ROOT`, also defaulting to `/fred/oz016/bgao_kn`.

Some OzSTAR log paths, Slurm partitions and paper output locations remain hardcoded. Search before running:

```bash
rg -n '/fred/oz016/bgao_kn|WORKSPACE_ROOT|#SBATCH' Model optical_only kn_simulation plots_scripts
```

Resource override variables differ between wrappers; inspect the shell entry point you will use. Prepare external data, runtime JSON, log directories, model weights and simulation executables before submitting compute work.
