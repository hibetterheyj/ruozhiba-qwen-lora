# Upload Package — Ruozhiba Humor Classification SFT

> Minimal, self-contained package for reproducing the LoRA SFT experiment on Qwen3-4B-Instruct.

---

## Directory Structure

```
upload/
├── readme.md                              # This file
├── link.md                                # File provenance (where each file comes from)
├── configs/
│   ├── prompts.yaml                       # Centralized system prompt + 8 humor categories
│   ├── qwen3_4b_base.yaml                # Training config — full dataset (2,785 samples)
│   ├── qwen3_4b_base_last3.yaml          # Training config — last-3-year dataset (1,025 samples)
│   └── qwen3_4b_merge.yaml               # LoRA merge config template
├── scripts/
│   ├── data/
│   │   └── build_sft_data.py              # Build ShareGPT training data (needs full data/tieba/)
│   ├── train/
│   │   ├── run_training.sh                # Launch LoRA SFT training (per-GPU)
│   │   └── batch_merge.sh                 # Merge 20 LoRA checkpoints → models/merged/
│   ├── inference/
│   │   ├── inference_eval.py              # vLLM offline batch inference
│   │   └── batch_inference.sh             # Run inference across 21 models
│   └── viz/
│       ├── eval_metrics.py                # Two-stage evaluation + visualization
│       └── gen_before_after.py            # Before/after comparison samples
├── data/
│   ├── ruozhiba_all.json                  # Full training set (2,785 ShareGPT conversations)
│   ├── ruozhiba_last3.json                # Last-3-year training set (1,025 conversations)
│   ├── ruozhiba_cqia_classified_v2.json   # Test set (240 CQIA samples with gold labels)
│   └── dataset_info.json                  # LLaMA-Factory dataset registry (ruozhiba only)
└── results/
    ├── eval_comparison.json               # 21-model evaluation comparison table
    └── before_after_samples.json          # 5 representative before/after examples
```

---

## Prerequisites

- Python 3.12 + CUDA 12.x
- NVIDIA GPU (≥ 24 GB VRAM for inference; 80 GB recommended for training)
- Base model: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

---

## Environment Setup

```bash
uv venv env_sft python 3.12
source env_sft/bin/activate
uv pip install 'llamafactory[metrics]' accelerate
uv pip install vllm json-repair seaborn matplotlib pyyaml
```

---

## Reproduction Steps

> 在**已克隆完整仓库的根目录**下执行以下命令（与主项目一致，需同级存在 `LLaMA-Factory/`、`models/`、`env_sft/`）。

### 1. Prepare LLaMA-Factory Data

Copy training data and dataset registry into LLaMA-Factory's data directory:

```bash
cp upload/data/ruozhiba_all.json    LLaMA-Factory/data/
cp upload/data/ruozhiba_last3.json  LLaMA-Factory/data/
cp upload/data/dataset_info.json    LLaMA-Factory/data/
```

> **Note**: If you already have a `dataset_info.json`, merge the two `ruozhiba_*` entries from `upload/data/dataset_info.json` into it instead of overwriting.

### 2. LoRA Fine-Tuning

```bash
# Full dataset — dual GPU parallel (two tmux panes)
bash upload/scripts/train/run_training.sh 0 8
bash upload/scripts/train/run_training.sh 1 16

# Last-3-year dataset
bash upload/scripts/train/run_training.sh 0 8  upload/configs/qwen3_4b_base_last3.yaml last3
bash upload/scripts/train/run_training.sh 1 16 upload/configs/qwen3_4b_base_last3.yaml last3
```

### 3. Merge LoRA Weights

```bash
bash upload/scripts/train/batch_merge.sh
```

This merges 20 LoRA checkpoints (4 experiments × epochs 3–7) into `models/merged/`.

### 4. Batch Inference

```bash
bash upload/scripts/inference/batch_inference.sh 0   # GPU 0
```

Runs greedy-decoding inference on 240 CQIA test samples for all 21 models (baseline + 20 merged).  
默认将 `results/results_*.json` 写入**仓库根**的 `results/`（与主仓库 `scripts/inference/` 行为对齐）。

### 5. Evaluation + Visualization

```bash
python upload/scripts/viz/eval_metrics.py \
    --results_dir results/ \
    --gold upload/data/ruozhiba_cqia_classified_v2.json \
    --comparison
```

Generates:
- `results/json/eval_*.json` — per-model metrics
- `results/json/eval_comparison.json` — cross-model comparison
- `results/confusion_matrices/` — confusion matrix plots
- `results/heatmaps/` — rank × epoch heatmaps
- `results/charts/` — trend & comparison charts

### 6. Before/After Comparison

```bash
python upload/scripts/viz/gen_before_after.py
```

---

## Key Results

| Metric | Baseline | Best (r16_e5) | Improvement |
|--------|----------|---------------|-------------|
| Strict Accuracy | 0.233 | **0.613** | +163% |
| Top-3 Hit Rate | 0.588 | **0.883** | +50% |
| JSON Strict Parse | 0.996 | **1.000** | — |
| Valid Sample Rate | 1.000 | 1.000 | — |

Best model: **R16, full data, epoch 5** (eval_loss = 0.8859).

---

## Path Configuration

Scripts use `PROJECT_ROOT` relative to their own location. If you move files, update the following:

| Script | Path variable | Default |
|--------|--------------|---------|
| `run_training.sh` | `PROJECT_ROOT` | `$(cd "$(dirname "$0")/.." && pwd)` |
| `batch_inference.sh` | `PROJECT_ROOT` | `/root/code/llm_ruozhiba` |
| `batch_merge.sh` | `PROJECT_ROOT` | `/root/code/llm_ruozhiba` |
| `inference_eval.py` | `PROJECT_ROOT` | `Path(__file__).resolve().parent.parent` |
| `eval_metrics.py` | `PROJECT_ROOT` | `Path(__file__).resolve().parent.parent` |
| `build_sft_data.py` | `PROJECT_ROOT` | `Path(__file__).resolve().parent.parent` |
