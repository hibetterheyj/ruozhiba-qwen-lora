#!/bin/bash
# 批量推理: 21 个模型 (baseline + 20 merged)
# 用法: bash scripts/batch_inference.sh [GPU_ID]

set -eo pipefail

GPU_ID=${1:-0}
PROJECT_ROOT="/root/code/llm_ruozhiba"

source "${PROJECT_ROOT}/env_sft/bin/activate"

echo "=== 批量推理: GPU ${GPU_ID} ==="

# Baseline
echo "--- 推理 baseline ---"
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
