# 弱智吧幽默分类 SFT 微调 — 开发计划 v0.2

> 基于 `dev_plan_0_1.md` 更新。Phase 1 (数据工程) + Phase 2.1-2.6 (训练) + Phase 2.8 (last3 训练) 已全部完成。
>
> **本版本重点**: 
> 1. Phase 2.7 全量合并 — 将 4 组训练 (R8/R16 × all/last3) × Epoch 3-7 共 **20 个 checkpoint** 全部合并
> 2. Phase 3 评估 — 对 20 个合并模型 + 1 个 baseline 共 **21 组**进行定量评估
> 3. Phase 4 报告 — 基于全量对比结果撰写**英文**实验报告

---

## 已完成工作回顾

| Phase | 状态 | 产出 |
|-------|------|------|
| 1.1 CQIA thought_process 导师蒸馏 | ✅ 完成 | `data/CQIA/ruozhiba_cqia_classified_v2.json` (240 条) |
| 1.2 去重防污染 | ✅ 完成 | `data/tieba/best*_classified_dedup.json` (2786 条) |
| 1.3 数据统计验证 | ✅ 完成 | `data/dedup_report.json` |
| 2.1 ShareGPT 格式化 | ✅ 完成 | `LLaMA-Factory/data/ruozhiba_all.json` (2785 条) |
| 2.2 数据集注册 | ✅ 完成 | `dataset_info.json` 已注册 |
| 2.3 MVP 训练 | ✅ 完成 | `saves/qwen3-4b/lora/mvp_r8_e3/` |
| 2.4 BS 压测 | ✅ 完成 | BS=16×2 (有效 32) |
| 2.5 双卡并行训练 (all) | ✅ 完成 | R8 7 epochs + R16 7 epochs |
| 2.6 训练监控 | ✅ 完成 | 最优 eval_loss: R16 checkpoint-415 (0.8859) |
| 2.7 单次合并 (旧) | ✅ 完成 | `models/Qwen3-4B-Ruozhiba-Merged/` (仅 R16-E5) → **将删除** |
| 2.8 双卡并行训练 (last3) | ✅ 完成 | R8_last3 7 epochs + R16_last3 7 epochs |

### 训练产物清单

**四组训练实验总览**:

| 实验 | 数据集 | LoRA Rank | Train 样本 | Steps/Epoch | 总 Steps | Eval Points |
|------|--------|-----------|-----------|-------------|----------|-------------|
| R8 | ruozhiba_all (2785) | 8 | 2,645 | 83 | 581 | 5 |
| R16 | ruozhiba_all (2785) | 16 | 2,645 | 83 | 581 | 5 |
| R8_last3 | ruozhiba_last3 (1025) | 8 | 973 | 31 | 217 | 2 |
| R16_last3 | ruozhiba_last3 (1025) | 16 | 973 | 31 | 217 | 2 |

**Checkpoint → Epoch 映射**:

| Epoch | R8/R16 (all) Step | R8_last3/R16_last3 Step |
|-------|-------------------|------------------------|
| 1 | 83 | 31 |
| 2 | 166 | 62 |
| 3 | 249 | 93 |
| 4 | 332 | 124 |
| 5 | 415 | 155 |
| 6 | 498 | 186 |
| 7 | 581 | 217 |

**Eval Loss 对比** (可用的 eval 采样点):

| Step (all) | Epoch | R8 Eval | R16 Eval | R8_last3 Eval | R16_last3 Eval |
|------------|-------|---------|----------|---------------|----------------|
| 100 (~E1.2) | ~1.2 | 1.0295 | 0.9842 | — | — |
| 200 (~E2.4) | ~2.4 | 0.9258 | 0.9034 | — | — |
| 300 (~E3.6) | ~3.6 | 0.8988 | 0.8886 | — | — |
| 400 (~E4.8) | ~4.8 | 0.8885 | **0.8859** | — | — |
| 500 (~E6.0) | ~6.0 | **0.8870** | 0.8915 | — | — |
| 100 (last3 ~E3.2) | ~3.2 | — | — | 1.0310 | 0.9927 |
| 200 (last3 ~E6.5) | ~6.5 | — | — | 0.9849 | **0.9623** |

> 详细训练分析见 `doc/analysis/train_analysis1.md`。

---

## Phase 2.7 (更新): 全量 LoRA 权重合并

### 目标

将 4 组训练 × Epoch 3-7 共 **20 个 checkpoint** 全部合并为独立模型，供 Phase 3 统一评估。

**现有合并模型处理**: `models/Qwen3-4B-Ruozhiba-Merged/` (R16-E5) **将删除**，因为新合并矩阵已包含 `merged/r16_e5` 这一等价模型。

```bash
rm -rf models/Qwen3-4B-Ruozhiba-Merged/
```

### 合并矩阵 (20 个模型)

#### A. 全量数据 (ruozhiba_all) — 10 个

| 编号 | Tag | Rank | Epoch | Step | Adapter 路径 | 输出目录 |
|------|-----|------|-------|------|-------------|----------|
| 1 | r8_e3 | 8 | 3 | 249 | `saves/.../r8/checkpoint-249` | `models/merged/r8_e3` |
| 2 | r8_e4 | 8 | 4 | 332 | `saves/.../r8/checkpoint-332` | `models/merged/r8_e4` |
| 3 | r8_e5 | 8 | 5 | 415 | `saves/.../r8/checkpoint-415` | `models/merged/r8_e5` |
| 4 | r8_e6 | 8 | 6 | 498 | `saves/.../r8/checkpoint-498` | `models/merged/r8_e6` |
| 5 | r8_e7 | 8 | 7 | 581 | `saves/.../r8/checkpoint-581` | `models/merged/r8_e7` |
| 6 | r16_e3 | 16 | 3 | 249 | `saves/.../r16/checkpoint-249` | `models/merged/r16_e3` |
| 7 | r16_e4 | 16 | 4 | 332 | `saves/.../r16/checkpoint-332` | `models/merged/r16_e4` |
| 8 | r16_e5 | 16 | 5 | 415 | `saves/.../r16/checkpoint-415` | `models/merged/r16_e5` |
| 9 | r16_e6 | 16 | 6 | 498 | `saves/.../r16/checkpoint-498` | `models/merged/r16_e6` |
| 10 | r16_e7 | 16 | 7 | 581 | `saves/.../r16/checkpoint-581` | `models/merged/r16_e7` |

#### B. 近三年数据 (ruozhiba_last3) — 10 个

| 编号 | Tag | Rank | Epoch | Step | Adapter 路径 | 输出目录 |
|------|-----|------|-------|------|-------------|----------|
| 11 | r8_last3_e3 | 8 | 3 | 93 | `saves/.../r8_last3/checkpoint-93` | `models/merged/r8_last3_e3` |
| 12 | r8_last3_e4 | 8 | 4 | 124 | `saves/.../r8_last3/checkpoint-124` | `models/merged/r8_last3_e4` |
| 13 | r8_last3_e5 | 8 | 5 | 155 | `saves/.../r8_last3/checkpoint-155` | `models/merged/r8_last3_e5` |
| 14 | r8_last3_e6 | 8 | 6 | 186 | `saves/.../r8_last3/checkpoint-186` | `models/merged/r8_last3_e6` |
| 15 | r8_last3_e7 | 8 | 7 | 217 | `saves/.../r8_last3/checkpoint-217` | `models/merged/r8_last3_e7` |
| 16 | r16_last3_e3 | 16 | 3 | 93 | `saves/.../r16_last3/checkpoint-93` | `models/merged/r16_last3_e3` |
| 17 | r16_last3_e4 | 16 | 4 | 124 | `saves/.../r16_last3/checkpoint-124` | `models/merged/r16_last3_e4` |
| 18 | r16_last3_e5 | 16 | 5 | 155 | `saves/.../r16_last3/checkpoint-155` | `models/merged/r16_last3_e5` |
| 19 | r16_last3_e6 | 16 | 6 | 186 | `saves/.../r16_last3/checkpoint-186` | `models/merged/r16_last3_e6` |
| 20 | r16_last3_e7 | 16 | 7 | 217 | `saves/.../r16_last3/checkpoint-217` | `models/merged/r16_last3_e7` |

> 路径中 `saves/...` 均指 `LLaMA-Factory/saves/qwen3-4b/lora/`

**磁盘预算**: 20 × 7.6 GB = **~152 GB** (磁盘剩余 ~706 GB，充裕)  
删除旧 `Qwen3-4B-Ruozhiba-Merged/` 回收 7.6 GB → 净增 ~145 GB。

### 合并配置模板

`configs/qwen3_4b_merge.yaml` 保持现有内容不变（已有具体 adapter 路径），由 `batch_merge.sh` 通过 CLI 参数动态覆盖 `adapter_name_or_path` 和 `export_dir`。

### 批量合并脚本: `scripts/batch_merge.sh`

```bash
#!/bin/bash
# 批量合并 4 组训练 × Epoch 3-7 共 20 个 checkpoint
# 用法: bash scripts/batch_merge.sh
#
# 串行执行 (每次合并约 3-5 分钟)
# 总预计耗时: ~60-100 分钟

set -e

PROJECT_ROOT="/root/code/llm_ruozhiba"
BASE_MODEL="${PROJECT_ROOT}/models/Qwen3-4B-Instruct-2507"
SAVES_DIR="${PROJECT_ROOT}/LLaMA-Factory/saves/qwen3-4b/lora"
MERGED_DIR="${PROJECT_ROOT}/models/merged"
MERGE_CONFIG="${PROJECT_ROOT}/configs/qwen3_4b_merge.yaml"

source "${PROJECT_ROOT}/env_sft/bin/activate"

# 定义合并矩阵: TRAIN_TAG, RANK, EPOCH, STEP
JOBS=()
# === 全量数据 (ruozhiba_all): 83 steps/epoch ===
for RANK in 8 16; do
    STEPS_MAP=(249 332 415 498 581)  # epoch 3-7
    for i in 0 1 2 3 4; do
        EPOCH=$((i + 3))
        STEP=${STEPS_MAP[$i]}
        JOBS+=("r${RANK} ${RANK} ${EPOCH} ${STEP}")
    done
done
# === 近三年数据 (ruozhiba_last3): 31 steps/epoch ===
for RANK in 8 16; do
    STEPS_MAP=(93 124 155 186 217)  # epoch 3-7
    for i in 0 1 2 3 4; do
        EPOCH=$((i + 3))
        STEP=${STEPS_MAP[$i]}
        JOBS+=("r${RANK}_last3 ${RANK} ${EPOCH} ${STEP}")
    done
done

TOTAL=${#JOBS[@]}
echo "=== 批量合并: ${TOTAL} 个模型 ==="
echo "输出目录: ${MERGED_DIR}"
echo ""

COUNT=0
for JOB in "${JOBS[@]}"; do
    read -r TRAIN_TAG RANK EPOCH STEP <<< "$JOB"
    TAG="${TRAIN_TAG}_e${EPOCH}"
    ADAPTER_PATH="${SAVES_DIR}/${TRAIN_TAG}/checkpoint-${STEP}"
    EXPORT_DIR="${MERGED_DIR}/${TAG}"
    COUNT=$((COUNT + 1))

    echo "[${COUNT}/${TOTAL}] 合并 ${TAG} (rank=${RANK}, checkpoint-${STEP})..."

    if [[ -d "${EXPORT_DIR}" ]] && [[ -f "${EXPORT_DIR}/config.json" ]]; then
        echo "  ⏭️  已存在，跳过"
        continue
    fi

    if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ]]; then
        echo "  ❌ adapter 不存在: ${ADAPTER_PATH}"
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
        "${MERGE_CONFIG}" \
        --model_name_or_path "${BASE_MODEL}" \
        --adapter_name_or_path "${ADAPTER_PATH}" \
        --export_dir "${EXPORT_DIR}" \
        2>&1 | tail -3

    echo "  ✅ 完成 → ${EXPORT_DIR}"
    echo ""
done

echo "=== 全部合并完成 (${TOTAL} 个模型) ==="
ls -ld "${MERGED_DIR}"/*/
```

**执行方式**:

```bash
# 先删除旧的单次合并模型, 回收 7.6 GB
rm -rf models/Qwen3-4B-Ruozhiba-Merged/

# 批量合并
bash scripts/batch_merge.sh 2>&1 | tee logs/batch_merge.log
```

**关键设计**:
- **幂等性**: 检查输出目录是否已存在 `config.json`，跳过已合并的模型，支持中断后重跑
- **串行执行**: 合并操作是 CPU/IO 密集型，串行即可
- **统一命名**: `r{rank}_e{epoch}` (全量) / `r{rank}_last3_e{epoch}` (近三年)
- **LLaMA-Factory CLI 覆盖**: 通过命令行参数动态覆盖 YAML 中的值

---

## Phase 3: 评估

### 总览

```
Phase 3.1  批量推理          ──→  Phase 3.2  定量评估        ──→  Phase 3.3  LLM-as-Judge
                                                                  (仅 Top 2-3 模型)
┌────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐
│ 21 组 sglang 推理      │   │ 21 组两阶段 JSON 评估     │   │ 20 样本 × 双盲评分    │
│ (20 merged + baseline) │   │ Stage 1: 格式遵循        │   │ 位置互换消除偏置      │
│ 240 条 × 21 = 5040 次  │   │ Stage 2: 逻辑准确率      │   │ 裁判: DeepSeek-Chat  │
│ temperature=0.0        │   │ 混淆矩阵 + 对比总表      │   │                      │
│                        │   │ Rank×Epoch×Dataset 热力图│   │                      │
└────────────────────────┘   └──────────────────────────┘   └──────────────────────┘
```

### 评估矩阵 (21 组实验)

| 编号 | Tag | 模型 | 数据集 | 结果文件 |
|------|-----|------|--------|----------|
| 0 | baseline | Qwen3-4B-Instruct-2507 | — | `results/results_baseline.json` |
| **全量数据 (ruozhiba_all)** | | | | |
| 1 | r8_e3 | R8, Epoch 3 | all | `results/results_r8_e3.json` |
| 2 | r8_e4 | R8, Epoch 4 | all | `results/results_r8_e4.json` |
| 3 | r8_e5 | R8, Epoch 5 | all | `results/results_r8_e5.json` |
| 4 | r8_e6 | R8, Epoch 6 | all | `results/results_r8_e6.json` |
| 5 | r8_e7 | R8, Epoch 7 | all | `results/results_r8_e7.json` |
| 6 | r16_e3 | R16, Epoch 3 | all | `results/results_r16_e3.json` |
| 7 | r16_e4 | R16, Epoch 4 | all | `results/results_r16_e4.json` |
| 8 | r16_e5 | R16, Epoch 5 | all | `results/results_r16_e5.json` |
| 9 | r16_e6 | R16, Epoch 6 | all | `results/results_r16_e6.json` |
| 10 | r16_e7 | R16, Epoch 7 | all | `results/results_r16_e7.json` |
| **近三年数据 (ruozhiba_last3)** | | | | |
| 11 | r8_last3_e3 | R8, Epoch 3 | last3 | `results/results_r8_last3_e3.json` |
| 12 | r8_last3_e4 | R8, Epoch 4 | last3 | `results/results_r8_last3_e4.json` |
| 13 | r8_last3_e5 | R8, Epoch 5 | last3 | `results/results_r8_last3_e5.json` |
| 14 | r8_last3_e6 | R8, Epoch 6 | last3 | `results/results_r8_last3_e6.json` |
| 15 | r8_last3_e7 | R8, Epoch 7 | last3 | `results/results_r8_last3_e7.json` |
| 16 | r16_last3_e3 | R16, Epoch 3 | last3 | `results/results_r16_last3_e3.json` |
| 17 | r16_last3_e4 | R16, Epoch 4 | last3 | `results/results_r16_last3_e4.json` |
| 18 | r16_last3_e5 | R16, Epoch 5 | last3 | `results/results_r16_last3_e5.json` |
| 19 | r16_last3_e6 | R16, Epoch 6 | last3 | `results/results_r16_last3_e6.json` |
| 20 | r16_last3_e7 | R16, Epoch 7 | last3 | `results/results_r16_last3_e7.json` |

---

### 3.1 批量推理脚本

#### 新脚本: `scripts/inference_eval.py`

**核心设计**: 一次加载一个模型，对 240 条测试集进行 sglang 离线批量推理，然后 shutdown 引擎、释放显存，加载下一个模型。脚本支持单模型推理和批量模式。

```
输入:
  - 测试集: data/CQIA/ruozhiba_cqia_classified_v2.json (240 条)
  - 模型路径: 单个模型或模型目录列表
输出:
  - results/results_{model_tag}.json (每个模型一个文件)
```

**命令行接口**:

```bash
# 单模型推理
python scripts/inference_eval.py \
    --model_path models/Qwen3-4B-Instruct-2507 \
    --tag baseline \
    --gpu 0

# 批量推理 (所有合并模型)
python scripts/inference_eval.py \
    --model_dir models/merged \
    --gpu 0

# 指定模型列表
python scripts/inference_eval.py \
    --model_paths models/merged/r8_e3 models/merged/r16_e5 \
    --gpu 0
```

**关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `temperature` | 0.0 | Greedy Decoding，确保确定性输出 |
| `max_new_tokens` | 1500 | 覆盖 thought_process + top3_categories 完整输出 |
| `mem_fraction_static` | 0.85 | 为系统保留 15% 显存缓冲 |
| `tp_size` | 1 | 单卡推理 (4B 模型无需张量并行) |

**运行环境**: 使用 `env_sft` 虚拟环境 (`sglang 0.5.3` 已在 env_sft 中可用)。

```bash
source env_sft/bin/activate
python scripts/inference_eval.py --model_dir models/merged --gpu 0
```

**代码逻辑骨架**:

```python
import sglang as sgl
import json, yaml, argparse, time, gc, logging
from pathlib import Path
import torch

# 抑制 sglang 底层日志，保持终端清爽 (只留 tqdm 进度条)
logging.getLogger("sglang").setLevel(logging.ERROR)

def run_inference(model_path: str, test_data: list, system_prompt: str,
                  tag: str, output_dir: str, gpu_id: int = 0):
    """对单个模型执行 240 条批量推理"""
    
    # 1. 启动引擎
    engine = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        mem_fraction_static=0.85,
    )

    # 2. 构造 prompt (apply_chat_template)
    tokenizer = engine.get_tokenizer()
    prompts = []
    for item in test_data:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["instruction"]}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # 3. 批量推理
    sampling_params = {"temperature": 0.0, "max_new_tokens": 1500}
    outputs = engine.generate(prompts, sampling_params)

    # 4. 保存结果
    results = []
    for i, out in enumerate(outputs):
        results.append({
            "index": i,
            "instruction": test_data[i]["instruction"],
            "gold_classification": test_data[i].get("classification"),
            "model_output": out["text"],
            "model_tag": tag,
        })

    output_path = Path(output_dir) / f"results_{tag}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 5. 三重清理 —— 彻底释放 CUDA 显存
    engine.shutdown()          # ① 释放 sglang 引擎
    del engine                 # ② 删除 Python 对象引用
    gc.collect()               # ③ 强制垃圾回收
    torch.cuda.empty_cache()   # ④ 清空 CUDA 缓存
    time.sleep(3)              # 等待 GPU 上下文彻底回收
    return output_path
```

**推理顺序与显存管理**:

```
推理顺序 (21 组):
  1. baseline (基座模型) — 性能基线
  2. r8_e3 → r8_e4 → r8_e5 → r8_e6 → r8_e7           (全量 R8)
  3. r16_e3 → r16_e4 → r16_e5 → r16_e6 → r16_e7       (全量 R16)
  4. r8_last3_e3 → ... → r8_last3_e7                    (近三年 R8)
  5. r16_last3_e3 → ... → r16_last3_e7                  (近三年 R16)

每个模型推理完成后 (三重清理):
  → engine.shutdown()       # ① 释放 sglang 引擎
  → del engine              # ② 删除 Python 对象引用
  → gc.collect()            # ③ 强制垃圾回收
  → torch.cuda.empty_cache()# ④ 清空 CUDA 缓存
  → time.sleep(3)           # 等待 GPU 上下文回收
  → 加载下一个模型
```

> **显存隔离**: 默认使用三重清理 (shutdown → del → gc → empty_cache)。若仍遇到 OOM，升级为子进程隔离 (`subprocess` 每次调用独立 Python 进程，确保 CUDA context 彻底销毁)。

**预计耗时**: 每个模型 240 条推理约 2-5 分钟 (sglang 批量推理)，21 个模型总计 ~45-105 分钟。

#### 批量推理封装脚本: `scripts/batch_inference.sh`

```bash
#!/bin/bash
# 批量推理: 21 个模型 (baseline + 20 merged)
# 用法: bash scripts/batch_inference.sh [GPU_ID]

GPU_ID=${1:-0}
PROJECT_ROOT="/root/code/llm_ruozhiba"

source "${PROJECT_ROOT}/env_sft/bin/activate"

echo "=== 批量推理: GPU ${GPU_ID} ==="

# Baseline
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/inference_eval.py \
    --model_path "${PROJECT_ROOT}/models/Qwen3-4B-Instruct-2507" \
    --tag baseline --gpu 0

# 全量数据模型
for RANK in 8 16; do
    for EPOCH in 3 4 5 6 7; do
        TAG="r${RANK}_e${EPOCH}"
        MODEL_PATH="${PROJECT_ROOT}/models/merged/${TAG}"
        if [[ ! -d "${MODEL_PATH}" ]]; then echo "⏭️  跳过 ${TAG}"; continue; fi
        echo "--- 推理 ${TAG} ---"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/inference_eval.py \
            --model_path "${MODEL_PATH}" --tag "${TAG}" --gpu 0
    done
done

# 近三年数据模型
for RANK in 8 16; do
    for EPOCH in 3 4 5 6 7; do
        TAG="r${RANK}_last3_e${EPOCH}"
        MODEL_PATH="${PROJECT_ROOT}/models/merged/${TAG}"
        if [[ ! -d "${MODEL_PATH}" ]]; then echo "⏭️  跳过 ${TAG}"; continue; fi
        echo "--- 推理 ${TAG} ---"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/inference_eval.py \
            --model_path "${MODEL_PATH}" --tag "${TAG}" --gpu 0
    done
done

echo "=== 推理完成 ==="
```

---

### 3.2 定量评估 — 两阶段 JSON 评估协议

#### 新脚本: `scripts/eval_metrics.py`

```
输入:
  - 金标准: data/CQIA/ruozhiba_cqia_classified_v2.json (240 条)
  - 推理结果: results/results_*.json (21 个文件)
输出:
  - 单模型评估: results/eval_{tag}.json
  - 对比总表: results/eval_comparison.json
  - 混淆矩阵: results/confusion_matrix_{tag}.png
  - Rank×Epoch 热力图: results/heatmap_*.png (含 eval_loss)
```

**命令行接口**:

```bash
# 评估单个结果文件
python scripts/eval_metrics.py \
    --results results/results_r16_e5.json \
    --gold data/CQIA/ruozhiba_cqia_classified_v2.json

# 批量评估所有结果文件 + 生成对比总表
python scripts/eval_metrics.py \
    --results_dir results/ \
    --gold data/CQIA/ruozhiba_cqia_classified_v2.json \
    --comparison
```

#### Stage 1: JSON 格式遵循能力

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `json_strict` | 原始输出直接 `json.loads()` 成功率 | 原生 JSON 遵循能力 |
| `json_tolerant` | `re.search(r'\{.*\}', text, re.DOTALL)` 提取后解析 | 容错 JSON 提取 |
| `vsr` (Valid Sample Rate) | 经 `json-repair` 修复后可解析的比例 | 修复后可用率 |

> **VSR 阈值**: VSR < 80% 的模型视为"指令遵循不可用"，在报告中标记。

#### Stage 2: 逻辑准确率

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `top1_accuracy` | 预测 Top-1 == 金标准 Top-1 | 主分类准确率 |
| `top3_hit_rate` | 金标准 Top-1 ∈ 预测 Top-3 | 广义分类覆盖率 |
| `confidence_mae` | $\frac{1}{N}\sum\|c_{pred} - c_{gold}\|$ | 置信度校准误差 |
| `strict_accuracy` | JSON strict 通过 **且** Top-1 正确 | 端到端可靠性 |
| `repaired_accuracy` | JSON repair 后 Top-1 正确 | 逻辑智能上限 |

**JSON 失效惩罚 (Maximum Penalty)**:

解析失败且修复失败的样本:
- Top-1 准确率: 计为 $0$
- 置信度 MAE: 计为 $1.0$

$$\text{MAE}_{\text{final}} = \frac{1}{N} \left( \sum_{i \in \text{valid}} |c_{pred,i} - c_{gold,i}| + \sum_{j \in \text{invalid}} 1.0 \right)$$

$$\text{Format\_Failure\_Penalty\_Impact} = \text{MAE}_{\text{final}} - \text{MAE}_{\text{valid\_only}}$$

#### 置信度校准分析

| 指标 | 含义 |
|------|------|
| 正确时平均置信度 | Top-1 命中样本的 `confidence_score` 均值 |
| 错误时平均置信度 | Top-1 未命中样本的 `confidence_score` 均值 |

> 错误时置信度 > 0.8 → 过度自信; < 0.4 → 具备不确定性感知

#### 混淆矩阵

对每个模型生成 8×8 Top-1 混淆矩阵:
- `confusion_matrix_{tag}_counts.png` — 原始计数
- `confusion_matrix_{tag}_normalized.png` — 按行归一化

#### 模型对比总表: `results/eval_comparison.json`

```json
{
  "comparison_table": [
    {
      "model_tag": "baseline",
      "dataset": null,
      "rank": null,
      "epoch": null,
      "eval_loss": null,
      "json_strict": 0.55,
      "json_tolerant": 0.68,
      "vsr": 0.75,
      "top1_accuracy": 0.30,
      "top3_hit_rate": 0.55,
      "confidence_mae": 0.25,
      "strict_accuracy": 0.20,
      "repaired_accuracy": 0.28
    },
    {
      "model_tag": "r8_e3",
      "dataset": "all",
      "rank": 8,
      "epoch": 3,
      "eval_loss": 0.8988,
      "json_strict": 0.88,
      "...": "..."
    },
    {
      "model_tag": "r16_last3_e5",
      "dataset": "last3",
      "rank": 16,
      "epoch": 5,
      "eval_loss": null,
      "...": "..."
    }
  ],
  "best_model": {
    "by_strict_accuracy": "r16_e5",
    "by_repaired_accuracy": "r16_e4",
    "by_top3_hit_rate": "r16_e5"
  }
}
```

> **eval_loss 来源**: 从训练日志 (`trainer_log.jsonl`) 提取各 checkpoint 对应的 eval_loss，合并到对比总表中。注意 all 和 last3 的 eval_loss 使用不同的 eval 子集，**跨数据集不可直接对比**，但同一数据集内可用于排序。

#### Rank×Epoch 热力图

**全量计算 7 个指标** (6 evaluation metrics + eval_loss from training)，绘制两组热力图 (all / last3):

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

metrics_to_compute = [
    "strict_accuracy", "repaired_accuracy", "top1_accuracy",
    "top3_hit_rate", "json_strict", "vsr", "eval_loss"
]

# 色阶配置: 锁定 vmin/vmax 确保 all 与 last3 同一指标的颜色映射绝对一致
# → 便于论文中并排对比，颜色深浅直接反映绝对差异
COLOR_SCALE = {
    "accuracy": {"vmin": 0.0, "vmax": 1.0, "cmap": "YlOrRd"},
    "loss":     {"vmin": 0.5, "vmax": 1.2, "cmap": "YlOrRd_r"},  # 值越小越好
}

def get_color_params(metric_name: str) -> dict:
    """根据指标类型返回色阶参数"""
    if metric_name in ("eval_loss", "confidence_mae"):
        return COLOR_SCALE["loss"]
    return COLOR_SCALE["accuracy"]

for dataset_tag in ["all", "last3"]:
    for metric_name in metrics_to_compute:
        matrix = np.zeros((2, 5))  # R8, R16 × E3-E7
        for r_idx, rank in enumerate([8, 16]):
            for e_idx, epoch in enumerate([3, 4, 5, 6, 7]):
                tag = f"r{rank}_e{epoch}" if dataset_tag == "all" \
                      else f"r{rank}_last3_e{epoch}"
                matrix[r_idx, e_idx] = results[tag][metric_name]
        
        color = get_color_params(metric_name)
        plt.figure(figsize=(8, 3))
        sns.heatmap(matrix, annot=True, fmt=".3f",
                    xticklabels=["E3", "E4", "E5", "E6", "E7"],
                    yticklabels=["R8", "R16"],
                    cmap=color["cmap"],
                    vmin=color["vmin"], vmax=color["vmax"])
        plt.title(f"{metric_name} ({dataset_tag}) — Rank × Epoch")
        plt.savefig(f"results/heatmap_{dataset_tag}_{metric_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()  # 防止绘图重叠
```

**色阶对齐 (Color Alignment)** — 同一指标在 all 与 last3 热力图间使用相同的 `vmin`/`vmax`:

| 指标类型 | vmin | vmax | cmap | 说明 |
|---------|------|------|------|------|
| accuracy 类 (strict, repaired, top1, top3, json_strict, vsr) | 0.0 | 1.0 | YlOrRd | 亮黄→深红 = 低→高，统一 0-1 范围 |
| loss 类 (eval_loss, confidence_mae) | 0.5 | 1.2 | YlOrRd_r | 反转色阶: 深红→亮黄 = 低→高 (低 loss = 好) |

> **为什么锁定色阶**: 若使用 seaborn 默认的自适应色阶，all 的 strict_accuracy 范围可能是 [0.5, 0.8]，last3 可能是 [0.3, 0.6]，两张图的"深红"代表不同的绝对值，并排对比时产生误导。锁定后颜色深浅直接反映绝对性能差异。

**产出热力图** (7 指标 × 2 数据集 = 14 张):

| 指标 | 全量 (all) | 近三年 (last3) |
|------|-----------|---------------|
| `strict_accuracy` | `heatmap_all_strict_accuracy.png` | `heatmap_last3_strict_accuracy.png` |
| `repaired_accuracy` | `heatmap_all_repaired_accuracy.png` | `heatmap_last3_repaired_accuracy.png` |
| `top1_accuracy` | `heatmap_all_top1_accuracy.png` | `heatmap_last3_top1_accuracy.png` |
| `top3_hit_rate` | `heatmap_all_top3_hit_rate.png` | `heatmap_last3_top3_hit_rate.png` |
| `json_strict` | `heatmap_all_json_strict.png` | `heatmap_last3_json_strict.png` |
| `vsr` | `heatmap_all_vsr.png` | `heatmap_last3_vsr.png` |
| `eval_loss` | `heatmap_all_eval_loss.png` | `heatmap_last3_eval_loss.png` |

**报告只选 3 张核心热力图** (详见 Phase 4):
1. `heatmap_all_strict_accuracy.png` — 端到端工程可用性
2. `heatmap_all_repaired_accuracy.png` — 纯逻辑理解力 (剥离格式噪声)
3. `heatmap_all_eval_loss.png` — 训练信号 vs 实际评估表现对比

> **对比震撞点**: eval_loss 热力图最优格 (R16-E5, 0.8859) 与 strict_accuracy 热力图最优格很可能不在同一位置。这就是 "eval_loss 最低 ≠ 最佳实际表现" 的直观证据。报告中用一段话讨论这个 gap 的原因 (eval_loss 是 token-level cross-entropy，而 accuracy 是 sample-level end-to-end correctness)。

#### all vs last3 对比分析

在对比总表中加入 **all vs last3 配对对比**，对同一 (Rank, Epoch) 组合比较两种数据集的效果差异:

```json
{
  "all_vs_last3_comparison": [
    {
      "rank": 8,
      "epoch": 5,
      "all_strict_accuracy": 0.62,
      "last3_strict_accuracy": 0.48,
      "delta": -0.14
    }
  ]
}
```

> 该对比直接回答 train_analysis1.md 中提出的核心问题: "数据时效性 vs 数据量的权衡" — 近三年数据更贴合当下语境但样本量少 63%，在 CQIA 测试集上效果如何？

---

### 3.3 LLM-as-Judge (定性评估)

#### 新脚本: `scripts/llm_judge.py`

**评估范围**: 仅对 **Top 2-3 最优合并模型** 执行 (由 Phase 3.2 对比总表确定)，不对全部 20 个模型做。

```
输入:
  - 推理结果: results/results_{best_tag}.json (Top 2-3 模型)
  - 金标准: data/CQIA/ruozhiba_cqia_classified_v2.json
输出:
  - 评分报告: results/judge_{tag}.json
  - 汇总: results/judge_summary.json
```

**评估配置**:

| 参数 | 值 |
|------|-----|
| 抽样数量 | 20 条 (从 240 条中随机抽取, seed=42) |
| 评轮次 | 2 (双盲位置互换) |
| API 调用总数 | 20 × 2 × Top_K 个模型 |
| 裁判模型 | **DeepSeek-Chat** (deepseek-chat) |
| `temperature` | 0.0 |
| `top_p` | 0.001 |

**裁判模型选择: DeepSeek-Chat**

使用 DeepSeek 而非 Claude 作为裁判，原因:
1. **避免同源偏置**: 金标准 (`thought_process` + `top3_categories`) 由 Claude-Opus-4-6 生成。若裁判与标注模型同源，可能对自身风格的输出产生系统性偏好，降低评估客观性
2. **成本优势**: DeepSeek-Chat API 成本显著低于 Claude，20×2×3 = 120 次调用更经济
3. **中文理解力**: DeepSeek 在中文幽默理解方面表现优秀，适合评估弱智吧段子分析质量

> 若 DeepSeek 评分方差过大或结果不合理，可追加 Claude 作为第二裁判进行交叉验证。

**API 防御策略 (DeepSeek 特性适配)**:

DeepSeek API 与 OpenAI 兼容但有两个关键差异: (1) 不实施严格限流 (无 429)，但在高峰期可能返回 500/503；(2) 服务器过载时可能长时间静默 (>5 min) 而不返回错误。针对性防御:

```python
import openai
from tenacity import (
    retry, wait_exponential, stop_after_attempt,
    retry_if_exception_type
)

# 初始化 client 时设置 60 秒超时 —— 防止服务器静默挂起
client = openai.OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    timeout=60.0  # 60s 后主动掐断，触发 APIConnectionError → 重试
)

# 仅捕获服务器端/网络端异常，避免在 400 逻辑错误时死循环
retry_exceptions = (
    openai.RateLimitError,       # 429 (罕见但可能)
    openai.APIConnectionError,   # 超时或网络中断
    openai.InternalServerError,  # 500/503 服务器过载
)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(retry_exceptions)
)
def get_judge_score(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.0,
        top_p=0.001,
    )
    return response.choices[0].message.content
```

| 防御维度 | 策略 | 原因 |
|---------|------|------|
| **特定异常捕获** | 仅重试 429 / 500 / 503 / timeout | 避免 `400 Invalid Format` 等逻辑错误陷入死循环 |
| **主动防挂起** | `timeout=60.0` | DeepSeek 过载时可能静默等待 >5 min，主动掐断触发重试 |
| **指数退避** | `wait_exponential(min=2, max=60)` | 2s→4s→8s→16s→32s，5 次后放弃 |
| **失败隔离** | 单条评分失败不中断整体流程 | `try/except` 包裹，记录失败样本 ID，最终汇总 |

**评分维度** (1-10 分):

| 维度 | 说明 |
|------|------|
| 逻辑准确度 | 是否正确识别了段子的核心笑点和逻辑机制 |
| 幽默感捕捉 | 是否理解了段子的幽默内涵而非仅字面分析 |
| 分析深度 | thought_process 是否深入拆解了多层幽默属性 |

**双盲互换 (Position Swapping)**:

| 轮次 | 分析 A | 分析 B |
|------|--------|--------|
| Round 1 | 微调模型输出 | 金标准参考 |
| Round 2 | 金标准参考 | 微调模型输出 |

最终得分 = (Round 1 对微调模型的评分 + Round 2 对微调模型的评分) / 2

**评分锚点**:

```
- 1-3 分: 分析完全偏离段子核心, 或仅进行字面翻译, 未触及幽默机制
- 4-5 分: 识别了表面笑点, 但缺少深层逻辑拆解, 分类有明显偏差
- 6-7 分: 正确识别主要幽默机制, 分类合理, 但分析深度一般
- 8-9 分: 精准捕捉多层幽默属性, 分类准确且理由充分, 展现文化理解力
- 10 分: 分析堪称教科书级别, 兼具语言学深度和文化洞察力
```

---

### 3.4 Before vs After 对比样本

从 Phase 3.1 推理结果中，选取 **5 条代表性样本** 展示 Before (baseline) vs After (最优模型) 的输出差异。

**选取策略**:
1. 2 条: baseline 完全错误，微调后正确
2. 1 条: baseline 格式混乱，微调后 JSON 工整
3. 1 条: 两者都正确但微调后分析更深入
4. 1 条: 微调后仍然错误的 failure case (展示诚实分析)

**输出**: `results/before_after_samples.json`

---

## Phase 4: 报告与交付

### 4.1 报告结构 (英文, 对照 assignment.md)

| Section | Content | Data Source |
|---------|---------|-------------|
| **3.1 SFT Target** | Chinese humor classification: 8-class Top-3 + thought_process | `configs/prompts.yaml`, 2-4 Before/After examples |
| **3.2 Dataset** | Sources (Tieba/CQIA/GitHub), dedup, ShareGPT format, 2785 train / 240 test, all vs last3 | `data/dedup_report.json`, format examples |
| **3.3 Training Setup** | Qwen3-4B-Instruct-2507, LoRA (R8/R16 × all/last3), bf16, 2×L20Z | Training YAML configs, 4×7 experiment matrix |
| **3.4 Loss Curves** | 4-run training_loss + eval_loss comparison, warmup + overfitting analysis | `saves/qwen3-4b/lora/*/training_*.png`, `train_analysis1.md` |
| **3.5 Before/After** | 5 test samples: baseline vs best model | `results/before_after_samples.json` |
| **Appendix A** | 21-model comparison table + Rank×Epoch heatmaps (3 core) | `results/eval_comparison.json`, heatmaps |
| **Appendix B** | Confusion matrices (best model + baseline) | `results/confusion_matrix_*.png` |
| **Appendix C** | LLM-as-Judge scores (DeepSeek-Chat) | `results/judge_summary.json` |

**报告语言**: English (assignment.md 要求)。数据样例 (中文段子) 以原文呈现，分析和讨论用英文。

**报告中选用的热力图 (3 张)**:

| 热力图 | 展示意义 | 在报告中的位置 |
|--------|---------|---------------|
| `heatmap_all_strict_accuracy.png` | End-to-end engineering reliability: JSON format + correct classification | Section 3.5 or Appendix A |
| `heatmap_all_repaired_accuracy.png` | Pure logical comprehension: true classification ability after format noise removal | Section 3.5 or Appendix A |
| `heatmap_all_eval_loss.png` | Training signal vs actual evaluation performance alignment | Section 3.4 (alongside loss curves) |

> **对比震撞点**: eval_loss 热力图最优格 (R16-E5, 0.8859) 与 strict_accuracy 热力图最优格很可能不在同一位置。这就是 "eval_loss 最低 ≠ 最佳实际表现" 的直观证据。报告中 discussion 段论述: eval_loss is token-level cross-entropy, while accuracy is sample-level end-to-end correctness。

### 4.2 交付物清单

```
提交内容:
├── report.pdf                              # 英文实验报告
├── scripts/
│   ├── classify_cqia_updated.py            # ✅ Phase 1.1 — CQIA thought_process 补全
│   ├── dedup_test_vs_train.py              # ✅ Phase 1.2 — 去重脚本
│   ├── build_sft_data.py                   # ✅ Phase 2.1 — ShareGPT 格式化
│   ├── probe_batch_size.sh                 # ✅ Phase 2.4 — 显存 BS 压测
│   ├── run_training.sh                     # ✅ Phase 2.5 — 训练启动
│   ├── batch_merge.sh                      # 🆕 Phase 2.7 — 批量权重合并 (20 模型)
│   ├── inference_eval.py                   # 🆕 Phase 3.1 — sglang 批量推理
│   ├── batch_inference.sh                  # 🆕 Phase 3.1 — 批量推理封装
│   ├── eval_metrics.py                     # 🆕 Phase 3.2 — 两阶段评估 + 热力图
│   └── llm_judge.py                        # 🆕 Phase 3.3 — LLM-as-Judge (DeepSeek)
├── configs/
│   ├── prompts.yaml                        # ✅ 中心化 system prompt
│   ├── qwen3_4b_mvp.yaml                  # ✅ MVP 配置
│   ├── qwen3_4b_base.yaml                 # ✅ 正式训练基础配置 (全量)
│   ├── qwen3_4b_base_last3.yaml           # ✅ 近三年训练配置
│   └── qwen3_4b_merge.yaml                # ✅ LoRA 合并配置模板
├── models/
│   ├── Qwen3-4B-Instruct-2507/            # 基座模型
│   └── merged/                             # 🆕 全量合并模型 (20 个)
│       ├── r8_e3/ ... r8_e7/               #   全量 R8
│       ├── r16_e3/ ... r16_e7/             #   全量 R16
│       ├── r8_last3_e3/ ... r8_last3_e7/   #   近三年 R8
│       └── r16_last3_e3/ ... r16_last3_e7/ #   近三年 R16
├── results/                                # 🆕 评估产出
│   ├── results_baseline.json               # Baseline 推理结果
│   ├── results_r{8,16}_e{3-7}.json        # 10 个全量模型结果
│   ├── results_r{8,16}_last3_e{3-7}.json  # 10 个近三年模型结果
│   ├── eval_comparison.json                # 21 模型对比总表 (+ all vs last3 对比)
│   ├── eval_{tag}.json                     # 各模型详细评估
│   ├── heatmap_*.png                       # Rank×Epoch 热力图 (14 张, 报告选 3)
│   ├── confusion_matrix_*.png              # 混淆矩阵 (至少 baseline + 最优)
│   ├── before_after_samples.json           # 5 条 Before/After 对比
│   └── judge_summary.json                  # LLM-as-Judge 汇总
└── README.md                               # 英文复现指南
```

---

## 依赖

### 已安装 (env_sft)

```
sglang==0.5.3, transformers==4.57.0, peft==0.17.1, torch==2.8.0
scikit-learn==1.7.2, numpy==2.3.3, matplotlib==3.10.6
llamafactory, accelerate, openai, tenacity, tqdm, pyyaml
```

### 需要安装

```bash
source env_sft/bin/activate
uv pip install seaborn json-repair
```

| 包 | 用途 |
|---|---|
| `seaborn` | 热力图 + 混淆矩阵可视化 |
| `json-repair` | Stage 1 JSON 修复 (VSR 计算) |

---

## 执行顺序 Checklist

### Phase 2.7: 全量合并 (20 个模型)

- [ ] 安装缺失依赖: `uv pip install seaborn json-repair`
- [ ] 删除旧合并模型: `rm -rf models/Qwen3-4B-Ruozhiba-Merged/`
- [ ] 编写 `scripts/batch_merge.sh`
- [ ] 执行批量合并: `bash scripts/batch_merge.sh 2>&1 | tee logs/batch_merge.log`
- [ ] 验证: `models/merged/` 下 20 个目录均包含 `config.json` + `model*.safetensors`
- [ ] 磁盘检查: `du -sh models/merged/`

### Phase 3.1: 批量推理 (21 组)

- [ ] 编写 `scripts/inference_eval.py`
- [ ] 编写 `scripts/batch_inference.sh`
- [ ] 确认 GPU 空闲: `nvidia-smi`
- [ ] 执行批量推理: `bash scripts/batch_inference.sh 0 2>&1 | tee logs/batch_inference.log`
- [ ] 验证: `results/` 下 21 个 `results_*.json` 文件完整

### Phase 3.2: 定量评估

- [ ] 编写 `scripts/eval_metrics.py`
- [ ] 执行批量评估: `python scripts/eval_metrics.py --results_dir results/ --gold ... --comparison`
- [ ] 检查对比总表: `results/eval_comparison.json`
- [ ] 检查热力图: `results/heatmap_*.png` (14 张)
- [ ] 检查混淆矩阵: `results/confusion_matrix_*.png`
- [ ] 确定 Top 2-3 最优模型
- [ ] 分析 eval_loss vs strict_accuracy 的一致性/差异
- [ ] 分析 all vs last3 对比结果

### Phase 3.3: LLM-as-Judge

- [ ] 编写 `scripts/llm_judge.py` (裁判: DeepSeek-Chat)
- [ ] 对 Top 2-3 模型执行 LLM 评分
- [ ] 检查评分报告: `results/judge_summary.json`

### Phase 3.4: Before/After

- [ ] 从推理结果中选取 5 条代表性样本
- [ ] 生成 `results/before_after_samples.json`

### Phase 4: 报告

- [ ] 编写英文 PDF 报告 (对照 assignment.md 5 个章节 + 3 个附录)
- [ ] 选择 3 张核心热力图放入报告 (strict_accuracy, repaired_accuracy, eval_loss)
- [ ] 更新 README.md 英文复现指南
- [ ] 最终检查: 所有脚本可运行、路径正确

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| sglang 引擎 shutdown 后显存未完全释放 | 三重清理 (shutdown → del → gc.collect → empty_cache)；若仍 OOM 则升级为 subprocess 子进程隔离 |
| 20 个模型合并耗时过长 | 幂等设计，支持中断重跑; 串行约 60-100 min |
| 21 组推理耗时过长 | 每组约 2-5 min，总计 45-105 min; 可双卡并行分担 |
| 部分模型 JSON 输出格式极差 (VSR < 50%) | 报告中如实记录，用 Format Failure Penalty 量化影响 |
| LLM-as-Judge API 调用失败 | 仅重试 429/500/503/timeout (避免 400 死循环)；client timeout=60s 防挂起；指数退避 2s→60s，5 次后放弃 |
| 磁盘空间不足 | 20 × 7.6 GB ≈ 152 GB - 7.6 GB (删旧) = ~145 GB，剩余 700+ GB 充足 |
| eval_loss 与 accuracy 热力图不一致 | 正常现象——报告中作为 discussion point 讨论 |
| last3 eval_loss 采样点太少 (仅 2 个) | 在热力图中使用训练日志插值或标注 "estimated"；报告中说明局限性 |
| 金标准由 Claude 生成，裁判用 Claude 会有同源偏置 | 裁判使用 DeepSeek-Chat，避免同源模型对获 |
