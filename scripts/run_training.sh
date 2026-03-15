#!/bin/bash
# Checkpoint-Based Training: 每个 Rank 训练 7 epochs, 自动保存所有 epoch checkpoint
# 用法:
#   bash scripts/run_training.sh 0 8                          # GPU 0, rank=8, 默认配置
#   bash scripts/run_training.sh 1 16                         # GPU 1, rank=16, 默认配置
#   bash scripts/run_training.sh 0 8 configs/qwen3_4b_base_last3.yaml  # 指定配置文件
#
# 并行调度 (两个 tmux pane):
#   终端 1: bash scripts/run_training.sh 0 8
#   终端 2: bash scripts/run_training.sh 1 16

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

GPU_ID=${1:?"用法: $0 <GPU_ID> <RANK> [CONFIG]  例: $0 0 8"}
RANK=${2:?"用法: $0 <GPU_ID> <RANK> [CONFIG]  例: $0 1 16"}
CONFIG_FILE="${3:-${PROJECT_ROOT}/configs/qwen3_4b_base.yaml}"
ALPHA=$((RANK * 2))
OUTPUT_DIR="saves/qwen3-4b/lora/r${RANK}"

# 激活虚拟环境
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source "${PROJECT_ROOT}/env_sft/bin/activate"
fi

cd "${PROJECT_ROOT}/LLaMA-Factory"

echo "=================================================="
echo " Qwen3-4B LoRA SFT — Checkpoint Training"
echo " GPU: ${GPU_ID} | rank=${RANK} | alpha=${ALPHA} | epochs=7"
echo " Config: ${CONFIG_FILE}"
echo " Output: ${OUTPUT_DIR}"
echo "=================================================="
echo ""

# 显示当前 GPU 状态
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader -i "${GPU_ID}" 2>/dev/null || true
echo "--------------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练开始"
echo ""

CUDA_VISIBLE_DEVICES="${GPU_ID}" llamafactory-cli train "${CONFIG_FILE}" \
    lora_rank="${RANK}" \
    lora_alpha="${ALPHA}" \
    output_dir="${OUTPUT_DIR}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练完成"
echo "Checkpoints: ${OUTPUT_DIR}/checkpoint-*"
