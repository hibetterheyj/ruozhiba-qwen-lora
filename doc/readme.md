# 文档索引（`doc/`）

**ruozhiba-qwen-lora** 的说明、复现与分析文档。建议阅读顺序：

**[guides/environment.md](guides/environment.md)** → **[guides/data.md](guides/data.md)** → **[guides/reproduction.md](guides/reproduction.md)** → **[guides/results_summary.md](guides/results_summary.md)**。

**脚本入口**：仓库根目录 [`scripts/readme.md`](../scripts/readme.md) — 按 **抓取 (`crawl/`)**、**数据处理 (`data/`)**、**训练与合并 (`train/`)**、**推理 (`inference/`)**、**评估与可视化 (`viz/`)**、**测试 (`tests/`)** 分类；其中 `tests/` 当前主要为 debug / adhoc 诊断脚本，命令均在仓库根目录执行。

---

## 目录结构一览

| 文件夹 | 用途 |
|--------|------|
| [`guides/`](guides/) | 环境、数据说明、复现流程、结果摘要（**优先读**） |
| [`analysis/`](analysis/) | 训练执行、训练/测试日志与 EDA 分析 |
| [`course/`](course/) | 作业要求、开发计划、变更日志、指导书 PDF |
| [`report/`](report/) | 正式实验报告（英文）与配图 |
| [`proposal/`](proposal/) | 早期选题与方案草稿 |

---

## `guides/` — 核心手册

| 文档 | 内容 |
|------|------|
| [environment.md](guides/environment.md) | Python/uv、依赖、硬件 |
| [data.md](guides/data.md) | 数据来源、规模、格式摘要 |
| [reproduction.md](guides/reproduction.md) | 流水线图与逐步命令 |
| [results_summary.md](guides/results_summary.md) | 实验矩阵、Loss/指标摘要 |

---

## `analysis/` — 训练与评估分析

| 文档 | 内容 |
|------|------|
| [training_execution.md](analysis/training_execution.md) | 双卡 tmux 训练操作 |
| [train_analysis1.md](analysis/train_analysis1.md) | 训练日志与 Loss 分析 |
| [test_analysis1.md](analysis/test_analysis1.md) | 评估与可视化产物 |
| [train_test_eda.md](analysis/train_test_eda.md) | Token 长度 EDA |

---

## `report/` — 实验报告

| 路径 | 说明 |
|------|------|
| [lab3_report.md](report/lab3_report.md) | 英文正式报告（Figure 1–8） |
| [lab3_report_v0_1.md](report/lab3_report_v0_1.md) | 报告早期版本 |
| [media/](report/media/) | 报告配图（见该目录 `README.md`） |

---

## `course/` — 作业与计划

| 文档 | 内容 |
|------|------|
| [assignment.md](course/assignment.md) | 课程作业要求 |
| [Lab3_SFT.pdf](course/Lab3_SFT.pdf) | 实验指导书 PDF |
| [howto.md](course/howto.md) | SFT 作业通用分步指南（非爬虫说明） |
| [dev_plan_0_1.md](course/dev_plan_0_1.md) … [dev_plan_0_4.md](course/dev_plan_0_4.md) | 开发计划迭代 |
| [changelog.md](course/changelog.md) | 阶段变更与产出日志 |
| `dev_plan_0_2.md.bak` | 开发计划备份（可忽略） |

---

## `proposal/` — 方案草稿

早期多模型讨论稿，仅供参考。
