#!/bin/bash
# Unified Primus entry point for every baseline experiment.
#
# Usage:
#   bash baselines/submit_primus.sh <method> [extra Hydra args]
#   METHOD=<method> bash baselines/submit_primus.sh
#
# Examples:
#   bash baselines/submit_primus.sh outcome_grpo
#   SEED=2 SMOKE=1 bash baselines/submit_primus.sh at_grpo
#   N_SAMPLES=8 bash baselines/submit_primus.sh self_consistency
#   bash baselines/submit_primus.sh --list
#   bash baselines/submit_primus.sh --dry-run gigpo
#
# One Primus job runs one method.  This is deliberate: separate jobs preserve
# per-method GPU-hour/token accounting and isolate failures/checkpoint resume.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
METHOD=${METHOD:-}
DRY_RUN=0

usage() {
  cat <<'EOF'
Unified baseline launcher

Trainable credit/system baselines:
  outcome_grpo          Shared terminal outcome GRPO on the adaptive four-role system
  role_reward           Simple role-wise reward GRPO
  at_grpo               AT-GRPO-Adapted
  gigpo                  GiGPO-Adapted
  single_agent_grpo      Proposer-only GRPO
  fixed_four_role_grpo   Fixed four-role pipeline + outcome GRPO

Evaluation-only system baselines:
  sft_cot                Single greedy SFT-CoT trajectory
  self_consistency       N-sample majority vote (N_SAMPLES=8 by default)
  self_refine            Initial answer + feedback + one refinement
  fixed_four_role        Fixed four-role SFT pipeline

Options:
  --list                 Print method names only
  --dry-run METHOD       Print the delegated command without running it
  -h, --help             Show this help

Common environment variables:
  MODEL_PATH_OVERRIDE, SFT_CKPT_OVERRIDE
  PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR
  PRIMUS_SAVE_CHECKPOINT_DIR
  SMOKE=1, SEED=1, MAX_STEPS=80, EXP_NAME=baseline_...
  VLLM_USE_V1_OVERRIDE, V1_VERIFIED=1, ENFORCE_EAGER=true

Evaluation variables:
  N_SAMPLES=8, BATCH_SIZE=16, TEMPERATURE=0.7

RACA is intentionally rejected. Use ../submit_primus.sh only for the existing
RACA experiment; it must not be rerun as a baseline.
EOF
}

list_methods() {
  printf '%s\n' \
    outcome_grpo role_reward at_grpo gigpo \
    single_agent_grpo fixed_four_role_grpo \
    sft_cot self_consistency self_refine fixed_four_role
}

if [[ ${1:-} == "--list" ]]; then
  list_methods
  exit 0
fi
if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if [[ ${1:-} == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ -n ${1:-} && ${1:-} != *=* ]]; then
  METHOD=$1
  shift
fi

if [[ -z ${METHOD} ]]; then
  usage >&2
  exit 2
fi

case "${METHOD}" in
  raca)
    echo "ERROR: RACA is not a baseline and cannot be launched here." >&2
    echo "Use the existing agentic_rl/submit_primus.sh only for RACA." >&2
    exit 2
    ;;
  outcome_grpo|role_reward|at_grpo|gigpo|single_agent_grpo|fixed_four_role_grpo)
    TARGET="${SCRIPT_DIR}/submit_primus_baseline.sh"
    COMMAND=(env "BASELINE_NAME=${METHOD}" bash "${TARGET}" "$@")
    ;;
  sft_cot|self_consistency|self_refine|fixed_four_role)
    if (( $# )); then
      echo "ERROR: evaluation baselines accept configuration through environment variables, not positional arguments." >&2
      echo "Use N_SAMPLES=..., BATCH_SIZE=..., or TEMPERATURE=... before the command." >&2
      exit 2
    fi
    TARGET="${SCRIPT_DIR}/submit_primus_system_eval.sh"
    COMMAND=(env "METHOD=${METHOD}" bash "${TARGET}")
    ;;
  all|all_train|all_eval)
    echo "ERROR: do not run multiple baselines serially in one Primus job." >&2
    echo "Submit one job per method so GPU-hours, tokens, failures, and resume state remain separable." >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown baseline method: ${METHOD}" >&2
    echo "Available methods:" >&2
    list_methods >&2
    exit 2
    ;;
esac

if (( DRY_RUN )); then
  printf 'METHOD=%q\n' "${METHOD}"
  printf 'COMMAND='
  printf '%s ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

printf '============================================================\n'
printf 'Baseline method: %s\n' "${METHOD}"
printf 'Delegating to:   %s\n' "${TARGET}"
printf '============================================================\n'
exec "${COMMAND[@]}"
