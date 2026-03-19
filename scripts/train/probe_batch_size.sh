#!/bin/bash
# LLaMA-Factory 动态 Batch Size 压测 (Qwen3-4B on L20Z 80GB)
# 目的: 探测最大可用 per_device_train_batch_size
# 用法: bash scripts/train/probe_batch_size.sh [GPU_ID]
#   默认使用 GPU 1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

GPU_ID=${1:-1}
BATCH_SIZES=(16 24 32 48 64 72 80)
MODEL_PATH="${PROJECT_ROOT}/models/Qwen3-4B-Instruct-2507"
DATASET="ruozhiba_all"
TEMPLATE="qwen3_nothink"
OUTPUT_DIR="${PROJECT_ROOT}/saves/probe_tmp"
MAX_STEPS=15
PROBE_CONFIG="${PROJECT_ROOT}/configs/probe_tmp.yaml"
PROBE_LOG="${PROJECT_ROOT}/probe_log.txt"
LAST_PASS=""

cd "${PROJECT_ROOT}/LLaMA-Factory"

# Activate virtual environment if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source "${PROJECT_ROOT}/env_sft/bin/activate"
fi

echo "=================================================="
echo " L20Z 80GB Batch Size 压测"
echo " GPU: ${GPU_ID} | 阶梯: ${BATCH_SIZES[*]}"
echo " MAX_STEPS: ${MAX_STEPS} | Model: $(basename ${MODEL_PATH})"
echo "=================================================="

# 显示当前 GPU 状态
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader -i "${GPU_ID}"
echo "--------------------------------------------------"

rm -rf "${OUTPUT_DIR}"

for BS in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "[$(date '+%H:%M:%S')] 压测 per_device_train_batch_size = ${BS} ..."

    cat <<EOF > "${PROBE_CONFIG}"
model_name_or_path: ${MODEL_PATH}
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
dataset: ${DATASET}
template: ${TEMPLATE}
cutoff_len: 2048
output_dir: ${OUTPUT_DIR}
overwrite_output_dir: true
per_device_train_batch_size: ${BS}
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
max_steps: ${MAX_STEPS}
logging_steps: 1
bf16: true
report_to: none
save_steps: 999999
EOF

    if CUDA_VISIBLE_DEVICES="${GPU_ID}" llamafactory-cli train "${PROBE_CONFIG}" > "${PROBE_LOG}" 2>&1; then
        PEAK=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "${GPU_ID}" 2>/dev/null || echo "N/A")
        echo "  ✅ BS=${BS} 通过 | GPU ${GPU_ID} 当前显存: ${PEAK}"
        # 从日志中提取 loss 信息作为 sanity check
        grep -E "'loss'" "${PROBE_LOG}" | tail -3 || true
        LAST_PASS="${BS}"
        echo "--------------------------------------------------"
    else
        echo "  ❌ BS=${BS} 失败 (OOM 或致命错误)"
        echo "  安全临界值: BS=${LAST_PASS}"
        # 显示错误关键行
        grep -iE "out of memory|CUDA|RuntimeError|Error" "${PROBE_LOG}" | tail -5 || true
        echo "--------------------------------------------------"
        break
    fi
done

# 清理
rm -f "${PROBE_CONFIG}"
rm -rf "${OUTPUT_DIR}"

echo ""
echo "=================================================="
echo " 压测结果汇总"
echo "=================================================="
if [ -n "${LAST_PASS}" ]; then
    echo " 最大安全 Batch Size: ${LAST_PASS}"
    echo " 推荐配置:"
    echo "   per_device_train_batch_size: ${LAST_PASS}"
    echo "   gradient_accumulation_steps: 1"
else
    echo " 所有 Batch Size 均失败，请检查 GPU 状态"
fi
echo "=================================================="
