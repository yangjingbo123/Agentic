#!/bin/bash
# SFT warm-up: train 4 role LoRA adapters in parallel on 4x H100
# Each role gets 1 dedicated GPU.
# Usage: bash scripts/train_sft_4role.sh [extra torchrun args]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_PATH=${MODEL_PATH:-"/data/yangjingbo/models/Qwen3-8B"}
DATA_PATH="${ROOT}/data/sft_train.jsonl"
SAVE_DIR="${ROOT}/checkpoints/sft"
EPOCHS=${SFT_EPOCHS:-3}
BATCH=${SFT_BATCH:-4}
LR=${SFT_LR:-2e-5}
MAX_LEN=${SFT_MAX_LEN:-1024}

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p "${SAVE_DIR}" logs

echo "========================================================"
echo "  SFT 4-role warm-up"
echo "  Model  : ${MODEL_PATH}"
echo "  Data   : ${DATA_PATH}  ($(wc -l < "${DATA_PATH}") episodes)"
echo "  SaveDir: ${SAVE_DIR}"
echo "  Epochs : ${EPOCHS}  Batch/GPU: ${BATCH}  LR: ${LR}"
echo "========================================================"

# ── Launch one process per role, each on 1 GPU ───────────────────────────

declare -A ROLE_GPUS=(
    [controller]="3"
    [proposer]="4"
    [critic]="7"
    [verifier]="7"
)

PIDS=()

for ROLE in controller proposer critic verifier; do
    GPU="${ROLE_GPUS[$ROLE]}"
    LOG_FILE="logs/sft_${ROLE}.log"
    echo "Starting ${ROLE} on GPU ${GPU} → ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    python "${ROOT}/train_sft_4role.py" \
            --role          "${ROLE}" \
            --model_path    "${MODEL_PATH}" \
            --data_path     "${DATA_PATH}" \
            --save_dir      "${SAVE_DIR}" \
            --epochs        "${EPOCHS}" \
            --batch_size    "${BATCH}" \
            --max_length    "${MAX_LEN}" \
            --lr            "${LR}" \
            "$@" \
        >"${LOG_FILE}" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All 4 roles training in parallel. PIDs: ${PIDS[*]}"
echo "Monitor logs with:"
echo "  tail -f logs/sft_controller.log logs/sft_proposer.log logs/sft_critic.log logs/sft_verifier.log"
echo ""

# ── Wait for all and report results ──────────────────────────────────────

FAILED=()
for i in "${!PIDS[@]}"; do
    ROLE=$(echo "controller proposer critic verifier" | tr ' ' '\n' | sed -n "$((i+1))p")
    PID="${PIDS[$i]}"
    if wait "${PID}"; then
        echo "✓ ${ROLE} finished"
    else
        echo "✗ ${ROLE} FAILED (exit $?)"
        FAILED+=("${ROLE}")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All adapters trained successfully."
    echo ""
    echo "Saved adapters:"
    for ROLE in controller proposer critic verifier; do
        ls -lh "${SAVE_DIR}/${ROLE}/adapter_model.safetensors" 2>/dev/null \
            && echo "  ${ROLE}: ${SAVE_DIR}/${ROLE}/" \
            || echo "  ${ROLE}: NOT FOUND"
    done
else
    echo "FAILED roles: ${FAILED[*]}"
    echo "Check logs/ for details."
    exit 1
fi
