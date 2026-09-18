#!/bin/bash
# Primus launcher for inference-only system baselines.
set -xeuo pipefail
cd "$(dirname "$0")/.."
METHOD=${METHOD:-${1:-}}
case "${METHOD}" in sft_cot|self_consistency|self_refine|fixed_four_role|iterative_proposal_voting) ;; *)
  echo "Usage: METHOD={sft_cot|self_consistency|self_refine|fixed_four_role|iterative_proposal_voting} $0" >&2; exit 2;; esac

if [[ "${ALLOW_PIP:-1}" == 1 ]]; then
  PIP_INDEX=${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}
  MISSING=$(python - <<'PY'
import importlib.util
mods={"torch":"torch","transformers":"transformers","yaml":"PyYAML","peft":"peft",
      "safetensors":"safetensors","numpy":"numpy","vllm":"vllm==0.9.2"}
print(" ".join(pkg for mod,pkg in mods.items() if importlib.util.find_spec(mod) is None))
PY
)
  [[ -z "${MISSING// /}" ]] || pip install --no-cache-dir -i "${PIP_INDEX}" ${MISSING}
fi

NUM_GPUS=${NUM_ACCELERATORS:-8}
[[ "${NNODES:-1}" == 1 ]] || { echo "System baselines require one node" >&2; exit 1; }
(( NUM_GPUS >= 2 )) || { echo "At least two GPUs are required (model + vLLM)" >&2; exit 1; }

MODEL_PATH=""; SFT_CKPT=""
IFS=';' read -ra KV <<< "${PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR:-}"
for item in "${KV[@]}"; do
  case "${item}" in actor:*) MODEL_PATH="${item#actor:}";; sft:*) SFT_CKPT="${item#sft:}";; esac
done
MODEL_PATH=${MODEL_PATH:-${MODEL_PATH_OVERRIDE:-}}
SFT_CKPT=${SFT_CKPT:-${SFT_CKPT_OVERRIDE:-}}
[[ -n "${MODEL_PATH}" && -n "${SFT_CKPT}" ]] || {
  echo "actor and sft sources are required" >&2; exit 1; }

VLLM_PROBE=$(python - <<'PY'
try:
 import vllm
 v=vllm.__version__; a,b=(int(x) for x in v.split('.')[:2])
 print(f"{v} {'0' if (a,b)<(0,10) else '1'}")
except Exception as e:
 print(f"unknown 0 # {type(e).__name__}: {e}")
PY
)
VLLM_VER=$(echo "${VLLM_PROBE}" | awk '{print $1}')
VLLM_V1=${VLLM_USE_V1_OVERRIDE:-$(echo "${VLLM_PROBE}" | awk '{print $2}')}
[[ "${VLLM_VER}" != unknown ]] || { echo "vLLM unavailable: ${VLLM_PROBE}" >&2; exit 1; }

python baselines/preflight.py --method "${METHOD}"
python baselines/test_baselines.py

N_SAMPLES=${N_SAMPLES:-8}
N_INDEPENDENT=${N_INDEPENDENT:-3}
N_INTERACTIVE=${N_INTERACTIVE:-2}
MAX_TIE_BREAK=${MAX_TIE_BREAK:-2}
PROPOSAL_CONTEXT_CHARS=${PROPOSAL_CONTEXT_CHARS:-600}
RUN_LABEL=${METHOD}
[[ "${METHOD}" != self_consistency ]] || RUN_LABEL=${METHOD}_n${N_SAMPLES}
[[ "${METHOD}" != iterative_proposal_voting ]] || \
  RUN_LABEL=${METHOD}_i${N_INDEPENDENT}_x${N_INTERACTIVE}_t${MAX_TIE_BREAK}
OUT_ROOT=${PRIMUS_SAVE_CHECKPOINT_DIR:-$(pwd)/results}/baseline-results/${RUN_LABEL}
mkdir -p "${OUT_ROOT}" /tmp/triton-cache
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
export VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH:-/tmp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache}
export VLLM_USE_FLASHINFER_SAMPLER=0
cat > "${OUT_ROOT}/system_manifest.json" <<EOF
{
  "method": "${METHOD}",
  "n_samples": ${N_SAMPLES},
  "independent_proposals": ${N_INDEPENDENT},
  "interactive_proposals": ${N_INTERACTIVE},
  "max_tie_break_proposals": ${MAX_TIE_BREAK},
  "proposal_context_chars": ${PROPOSAL_CONTEXT_CHARS},
  "temperature": ${TEMPERATURE:-0.7},
  "vllm_version": "${VLLM_VER}",
  "vllm_use_v1": "${VLLM_V1}",
  "git_commit": "${GIT_COMMIT}"
}
EOF
EAGER_ARG="--enforce_eager"; [[ "${ENFORCE_EAGER:-true}" == true ]] || EAGER_ARG=""
python baselines/evaluate_system.py \
  --method "${METHOD}" --checkpoint "${SFT_CKPT}" --model_path "${MODEL_PATH}" \
  --suite all --math_data data/math_test.jsonl --aime_data data/aime_2022_2026.jsonl \
  --math_samples 1000 --aime_samples 150 --output_dir "${OUT_ROOT}" \
  --batch_size "${BATCH_SIZE:-16}" --n_samples "${N_SAMPLES}" \
  --independent_proposals "${N_INDEPENDENT}" \
  --interactive_proposals "${N_INTERACTIVE}" \
  --max_tie_break_proposals "${MAX_TIE_BREAK}" \
  --proposal_context_chars "${PROPOSAL_CONTEXT_CHARS}" \
  --temperature "${TEMPERATURE:-0.7}" --vllm_use_v1 "${VLLM_V1}" ${EAGER_ARG}
