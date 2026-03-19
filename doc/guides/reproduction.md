# 复现流程

## 前提

- Python 3.12 + CUDA 与 vLLM / PyTorch 匹配  
- 基座模型 `Qwen3-4B-Instruct-2507` 已置于 `models/`（或按 LLaMA-Factory 配置的路径）  
- 环境安装见 [`environment.md`](environment.md)

## 流水线总览

脚本按阶段分目录（均在 `scripts/` 下，详见 [`../../scripts/readme.md`](../../scripts/readme.md)）：`crawl/` 抓取与抽取 · `data/` 清洗/分类/去重/SFT 构建 · `train/` 训练与合并 · `inference/` 批量推理 · `viz/` 评估与图表 · `tests/` 调试/诊断脚本。

```
                             ┌──────────────────────┐
                             │  Baidu Tieba Crawl   │
                             │  (aiotieba crawler)  │
                             └──────────┬───────────┘
                                        │
┌────────────────────┐    ┌─────────────▼──────────────────────┐    ┌──────────────────┐
│  COIG-CQIA (HF)   │    │  scripts/crawl/                    │    │  GitHub corpus   │
│  240 pairs        │───▶│  extract_annual_data.py            │◀───│  1361 posts      │
└────────────────────┘    │  extract_cqia_data.py             │    └──────────────────┘
                          │  process_ruozhiba_past_annual.py   │
                          └─────────────┬──────────────────────┘
                                        │
                          ┌─────────────▼──────────────────────┐
                          │  scripts/data/                     │
                          │  filter_duplicates.py              │
                          └─────────────┬──────────────────────┘
                                        │
                          ┌─────────────▼──────────────────────┐
                          │  scripts/data/                     │
                          │  classify_jokes.py …               │
                          └─────────────┬──────────────────────┘
                                        │
                          ┌─────────────▼──────────────────────┐
                          │  scripts/data/                     │
                          │  check_and_repair*.py              │
                          │  dedup_test_vs_train.py            │
                          └─────────────┬──────────────────────┘
                                        │
                          ┌─────────────▼──────────────────────┐
                          │  scripts/data/build_sft_data.py    │
                          │  → ruozhiba_all.json               │
                          └─────────────┬──────────────────────┘
                                        │
                          ┌─────────────▼──────────────────────┐
                          │  LLaMA-Factory LoRA SFT            │
                          │  scripts/train/run_training.sh     │
                          └─────────────┬──────────────────────┘
                                        │
                          ┌─────────────▼──────────────────────┐
                          │  scripts/train/batch_merge.sh      │
                          │  → models/merged/                  │
                          └─────────────┬──────────────────────┘
                                        │
                          ┌─────────────▼──────────────────────┐
                          │  scripts/inference/                │
                          │  inference_eval.py + batch_*.sh    │
                          └─────────────┬──────────────────────┘
                                        │
                          ┌─────────────▼──────────────────────┐
                          │  scripts/viz/eval_metrics.py       │
                          └─────────────┬──────────────────────┘
                                        │
                          ┌─────────────▼──────────────────────┐
                          │  scripts/viz/gen_before_after.py   │
                          └────────────────────────────────────┘
```

## 分步命令

> **工作目录**：以下命令默认在**仓库根目录**执行（保证 `configs/`、`data/`、`models/`、`results/`、`LLaMA-Factory/` 等相对路径正确）。

### 1. 环境

见 [`environment.md`](environment.md)。

### 2. 训练数据（已构建可跳过）

数据已在 `data/LLaMA-Factory/data/` 时，复制到框架内：

```bash
cp data/LLaMA-Factory/data/ruozhiba_*.json LLaMA-Factory/data/
cp data/LLaMA-Factory/data/dataset_info.json LLaMA-Factory/data/
```

从头构建：

```bash
python scripts/data/build_sft_data.py
```

### 3. LoRA 训练

```bash
# 双卡示例：R8 / R16 并行
bash scripts/train/run_training.sh 0 8
bash scripts/train/run_training.sh 1 16

# 近三年子集
bash scripts/train/run_training.sh 0 8  configs/qwen3_4b_base_last3.yaml last3
bash scripts/train/run_training.sh 1 16 configs/qwen3_4b_base_last3.yaml last3
```

更细步骤见 [`../analysis/training_execution.md`](../analysis/training_execution.md)。

### 4. 合并 LoRA

```bash
bash scripts/train/batch_merge.sh
```

### 5. 批量推理

```bash
bash scripts/inference/batch_inference.sh 0
```

### 6. 评估与可视化

```bash
python scripts/viz/eval_metrics.py \
    --results_dir results/ \
    --gold data/CQIA/ruozhiba_cqia_classified_v2.json \
    --comparison
```

### 7. Before / After 样本

```bash
python scripts/viz/gen_before_after.py

# 可选：覆盖 baseline / 候选结果 / 输出路径
python scripts/viz/gen_before_after.py \
    --baseline results/results_baseline.json \
    --candidate results/results_r16_e5.json \
    --output results/before_after_samples.json
```

## 最小提交包

若只需可运行子集，见仓库 [`upload/readme.md`](../upload/readme.md)。
