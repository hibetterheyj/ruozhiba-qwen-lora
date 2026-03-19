#!/bin/bash
# 批量合并 4 组训练 × Epoch 3-7 共 20 个 checkpoint
# 用法: bash scripts/train/batch_merge.sh
#
# 串行执行 (每次合并约 3-5 分钟)
# 总预计耗时: ~60-100 分钟

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
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
        model_name_or_path="${BASE_MODEL}" \
        adapter_name_or_path="${ADAPTER_PATH}" \
        export_dir="${EXPORT_DIR}" \
        2>&1 | tail -3

    echo "  ✅ 完成 → ${EXPORT_DIR}"
    echo ""
done

echo "=== 全部合并完成 (${TOTAL} 个模型) ==="
ls -ld "${MERGED_DIR}"/*/
