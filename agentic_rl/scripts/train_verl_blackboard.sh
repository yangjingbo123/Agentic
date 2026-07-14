#!/bin/bash
# Train 4-role blackboard multi-agent RL on 8x H100
# Usage: bash scripts/train_verl_blackboard.sh [experiment_name]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── User settings ──────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-"/data/yangjingbo/models/Qwen3-8B"}
EXPERIMENT_NAME=${1:-"blackboard_4role_grpo_$(date +%Y%m%d_%H%M)"}
WANDB_PROJECT="agentic_rl_verl"

# ── Adapter checkpoints (produced by SFT warm-up) ─────────────────────────
CKPT_DIR="${PROJECT_ROOT}/checkpoints/sft"
CONTROLLER_ADAPTER="${CKPT_DIR}/controller/controller"
PROPOSER_ADAPTER="${CKPT_DIR}/proposer/proposer"
CRITIC_ADAPTER="${CKPT_DIR}/critic/critic"
VERIFIER_ADAPTER="${CKPT_DIR}/verifier/verifier"

# If SFT adapters don't exist yet, use base model path as placeholder
# (the rollout will run without LoRA until adapters are available)
for adapter in "$CONTROLLER_ADAPTER" "$PROPOSER_ADAPTER" "$CRITIC_ADAPTER" "$VERIFIER_ADAPTER"; do
    if [ ! -d "$adapter" ]; then
        echo "WARNING: adapter not found at $adapter — falling back to base model weights"
    fi
done

# ── PYTHONPATH: include project root so verl/ and envs/ are importable ────
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Ray cluster (single node, 4 GPUs) ─────────────────────────────────────
unset RAY_ADDRESS
export CUDA_VISIBLE_DEVICES=5,6

# Start a fresh local Ray head using the current conda Python.
# Use a private port (6399) and temp dir to avoid conflicting with other clusters.
RAY_PORT=6399
RAY_TMP="/tmp/ray_agentic_rl_$$"
mkdir -p "${RAY_TMP}"

echo "Starting local Ray head on port ${RAY_PORT}..."
# Do NOT run 'ray stop' — this is a shared server with other users' clusters
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
ray start --head \
    --port "${RAY_PORT}" \
    --temp-dir "${RAY_TMP}" \
    --num-cpus 16 \
    --num-gpus "${NUM_GPUS}" \
    --dashboard-host 0.0.0.0 2>/dev/null &
sleep 5   # give Ray time to initialize

export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
echo "Ray head started. RAY_ADDRESS=${RAY_ADDRESS}"

# ── Flash-Attention + memory ──────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

echo "============================================================"
echo "  Blackboard 4-Role GRPO Training"
echo "  Model  : ${MODEL_PATH}"
echo "  Exp    : ${EXPERIMENT_NAME}"
echo "  GPUs   : ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

cd "${PROJECT_ROOT}"

python -m verl.rema_trainer.main_ppo \
    --config-path "${PROJECT_ROOT}/configs/verl" \
    --config-name  blackboard_4role_grpo \
    \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.project_name="${WANDB_PROJECT}" \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.lora_adapter_paths.controller="${CONTROLLER_ADAPTER}" \
    actor_rollout_ref.rollout.lora_adapter_paths.proposer="${PROPOSER_ADAPTER}" \
    actor_rollout_ref.rollout.lora_adapter_paths.critic="${CRITIC_ADAPTER}" \
    actor_rollout_ref.rollout.lora_adapter_paths.verifier="${VERIFIER_ADAPTER}" \
    \
    data.train_files="${PROJECT_ROOT}/data/math_train_rl.parquet" \
    data.val_files="${PROJECT_ROOT}/data/math_test_clean.parquet" \
    \
    trainer.total_training_steps=500 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.logger="[console,wandb]" \
    \
    "$@"   # pass through any extra overrides
