# `scripts/` 变更日志

> 与全项目文档级变更的交叉索引见 **[`doc/course/changelog.md`](../doc/course/changelog.md)**（含 `doc/`、`readme.md` 等路径更新）。

---

## 2026-03-19 — Review 备注（结构重组后待收尾项）

> 结论：本次 `scripts/` 分类重组与路径锚点统一整体方向正确，主体实现与目录设计一致；但若以“完全收尾、规范化完成”为标准，仍有若干建议需要单独记录，便于后续继续清理。

### 已确认合理的部分

- `scripts/` 已按 `crawl/`、`data/`、`train/`、`inference/`、`viz/`、`tests/` 分组。
- 主仓库下 Python 脚本已基本统一使用 `Path(__file__).resolve().parents[2]` 作为仓库根锚点。
- 主仓库下 Shell 脚本已统一使用 `PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"`。
- `scripts/inference/batch_inference.sh` 已通过 `INF_PY="${PROJECT_ROOT}/scripts/inference/inference_eval.py"` 调用推理入口。
- `scripts/train/batch_merge.sh`、`scripts/train/probe_batch_size.sh` 已移除写死的本机绝对路径。
- `extract_annual_data.py`、`fix_double_escapes.py`、`tests/*` 中旧的本机路径已改为基于仓库根的相对解析。
- `upload/scripts/` 已同步拆为 `data/`、`train/`、`inference/`、`viz/`，并与主仓库脚本路径保持同构。

### 本轮 review 建议记录

#### 1. `tests/` 下脚本更接近“调试脚本”，尚未完全工程化

当前 `scripts/tests/test_actual_file.py`、`test_fix_function.py`、`test_fix_quotes.py`、`test_json_parse.py` 已完成迁移，但仍存在以下共性问题：

- 多数文件为“导入即执行”，缺少 `main()` 封装或更标准的测试入口。
- 多处仍使用 `print`，未统一到 `logging`。
- 若按仓库 review 标准衡量，部分函数缺少类型标注。
- 其中若干文件更适合作为调试脚本而非稳定单测，后续可继续保留在 `tests/`，或显式标注其 “debug / adhoc” 性质。

#### 2. 个别测试/调试脚本存在可维护性问题

- `scripts/tests/test_fix_quotes.py` 中函数命名与行为不完全一致：`convert_ascii_quotes_to_chinese()` 当前并未真正把 ASCII 引号转换为中文引号，容易误导维护者。
- `scripts/tests/test_json_parse.py` 中存在疑似无效或损坏的替换逻辑，说明该脚本更像临时诊断稿，后续宜清理。
- `scripts/tests/test_actual_file.py` 与 `test_fix_function.py` 直接依赖真实数据文件，若后续作为正式测试执行，可能带来副作用或环境依赖问题。

#### 3. 若按统一代码规范，仍建议继续收尾

建议后续逐步完成以下整理：

- 将公共脚本中仍残留的 `print` 迁移为 `logging`（例如 `scripts/crawl/extract_annual_data.py`、`scripts/data/fix_double_escapes.py`、`scripts/viz/gen_before_after.py` 等）。
- 为公共函数补全类型标注与必要 docstring。
- 避免裸 `except:`；优先收窄异常范围。
- 将硬编码的模型标签或结果文件名提为 CLI 参数或常量，提升可复用性。

#### 4. `upload/` 同构说明整体成立，但文档措辞宜保持准确

- 当前 `upload/scripts/` 与主仓库的 `data/`、`train/`、`inference/`、`viz/` 已保持同构。
- 但若文档使用“完全同构”表述，需注意 `upload/` 并未复制 `scripts/tests/`，因此更准确的说法是：对提交包所保留的核心脚本子目录保持同构。

### 建议的后续 TODO

- 清理并规范 `scripts/tests/` 下的调试脚本。
- 将 `gen_before_after.py` 中硬编码结果文件改为参数化输入。
- 持续将剩余 `print` 迁移到 `logging`。

## 2026-03-19 — 按流水线阶段分子目录 + 路径锚点统一

### 背景

原先脚本平铺在 `scripts/` 根下，不便按「抓取 → 数据处理 → 训练 → 推理 → 可视化」浏览；部分脚本使用 `Path(__file__).parent.parent` 或硬编码本机路径，在子目录化后会指向错误目录。

### `scripts/` 新结构

| 子目录 | 职责 | 主要文件 |
|--------|------|----------|
| **`crawl/`** | 从 HF CQIA、GitHub 年度语料等**抽取/预处理**（贴吧爬虫落盘数据仍在仓库根目录 **`crawler/`**，不在此目录） | `extract_cqia_data.py`、`extract_annual_data.py`、`process_ruozhiba_past_annual.py` |
| **`data/`** | 去重、LLM 分类（**YAML 配置与脚本同目录**）、校验修复、测试集去污染、ShareGPT 构建 | `filter_duplicates.py`、`classify_*.py`、`classify_*.yaml`、`check_*.py`、`fix_*.py`、`dedup_test_vs_train.py`、`build_sft_data.py` |
| **`train/`** | LoRA 训练启动、Batch 压测、批量 `export` 合并 | `run_training.sh`、`probe_batch_size.sh`、`batch_merge.sh` |
| **`inference/`** | vLLM 离线推理与 21 模型串行封装 | `inference_eval.py`、`batch_inference.sh` |
| **`viz/`** | 两阶段 JSON 评估、图表、Before/After 样本 | `eval_metrics.py`、`gen_before_after.py` |
| **`tests/`** | 引号/JSON 等调试与小型单测 | `test_*.py` |

### 路径约定（仓库根目录）

- **Python**：仓库根 = `Path(__file__).resolve().parents[2]`（适用于 `scripts/<子目录>/*.py`）。
- **Shell**：`PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"`（从 `scripts/train/` 等上溯两级）。
- **`batch_inference.sh`**：通过 `INF_PY="${PROJECT_ROOT}/scripts/inference/inference_eval.py"` 调用推理入口，避免依赖当前工作目录下的 `python scripts/...`。
- **其它修正**：`batch_merge.sh` / `probe_batch_size.sh` 去掉写死的绝对根路径；`extract_annual_data.py`、`fix_double_escapes.py`、`tests/*` 中旧的本机绝对路径改为基于仓库根；`data/` 内分类脚本的数据目录统一为 `parents[2] / "data"`。

### `upload/` 同步说明

最小提交包 **`upload/scripts/`** 与主仓库 **同构**：

| 主仓库 | `upload/` 对应 |
|--------|----------------|
| `scripts/data/build_sft_data.py` | `upload/scripts/data/build_sft_data.py` |
| `scripts/train/run_training.sh`、`batch_merge.sh` | `upload/scripts/train/*.sh` |
| `scripts/inference/inference_eval.py`、`batch_inference.sh` | `upload/scripts/inference/*` |
| `scripts/viz/eval_metrics.py`、`gen_before_after.py` | `upload/scripts/viz/*` |

**执行约定**：在**完整克隆的仓库根目录**下运行 `upload/scripts/...`，以便访问同级 `LLaMA-Factory/`、`models/`、`env_sft/`。

**行为对齐**：

- `upload/scripts/train/run_training.sh`：以仓库根为 `REPO_ROOT`，默认配置 `upload/configs/qwen3_4b_base.yaml`，`cd "${REPO_ROOT}/LLaMA-Factory"` 再调用 `llamafactory-cli`。
- `upload/scripts/train/batch_merge.sh`：`MERGE_CONFIG` 使用 `upload/configs/qwen3_4b_merge.yaml`，基座与 adapter 路径仍指向仓库根下 `models/`、`LLaMA-Factory/saves/`。
- `upload/scripts/inference/inference_eval.py`：`UPLOAD_ROOT`（上两级）用于默认测试集 `upload/data/ruozhiba_cqia_classified_v2.json` 与 `upload/configs/prompts.yaml`；**默认输出目录为仓库根 `results/`**，与主仓库推理及 `eval_metrics --results_dir results/` 一致。
- `upload/scripts/viz/gen_before_after.py`：以仓库根解析 `results/results_*.json`，与上述推理输出一致。

溯源表见 **[`upload/link.md`](../upload/link.md)**；复现步骤见 **[`upload/readme.md`](../upload/readme.md)**。

### 文档与入口

- 脚本总览：**[`readme.md`](readme.md)**  
- 复现流程与流程图：**[`doc/guides/reproduction.md`](../doc/guides/reproduction.md)**  
- 文档索引：**[`doc/readme.md`](../doc/readme.md)**  

---

## 此前变更

更早的脚本功能迭代（推理后端 vLLM、评估图表、`batch_merge` 等）记录在 **`doc/course/changelog.md`** 历史表格中；自本文件创建起，**以本文件为 `scripts/` 目录结构与路径约定的权威说明**，历史行中的旧路径（如 `scripts/eval_metrics.py`）在课程 changelog 中已逐步替换为 `scripts/viz/eval_metrics.py` 等形式以便检索。
