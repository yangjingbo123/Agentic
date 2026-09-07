#!/bin/bash
# Primus launcher for trainable baselines only.  It never launches RACA.
set -xeuo pipefail

cd "$(dirname "$0")/.."
BASELINE_NAME=${BASELINE_NAME:-${1:-}}
case "${BASELINE_NAME}" in
  outcome_grpo|role_reward|at_grpo|gigpo|single_agent_grpo|fixed_four_role_grpo)
    BASELINE_CONFIG_FILE="baselines/configs/${BASELINE_NAME}.yaml" ;;
  raca)
    echo "RACA is not a baseline. Use agentic_rl/submit_primus.sh; do not rerun it here." >&2
    exit 2 ;;
  *)
    echo "Usage: BASELINE_NAME={outcome_grpo|role_reward|at_grpo|gigpo|single_agent_grpo|fixed_four_role_grpo} $0" >&2
    exit 2 ;;
esac
BASELINE_SEED=${SEED:-1}
export BASELINE_NAME BASELINE_SEED PYTHONHASHSEED=${BASELINE_SEED}

# Dependency bootstrap follows the production launcher but is isolated here.
if [[ "${ALLOW_PIP:-1}" == "1" ]]; then
  PIP_INDEX=${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}
  MISSING=$(python - <<'PY'
import importlib.util
mods={"torch":"torch","transformers":"transformers","hydra":"hydra-core",
      "omegaconf":"omegaconf","yaml":"PyYAML","wandb":"wandb","peft":"peft",
      "safetensors":"safetensors","accelerate":"accelerate","numpy":"numpy",
      "vllm":"vllm==0.9.2","bitsandbytes":"bitsandbytes"}
print(" ".join(pkg for mod,pkg in mods.items() if importlib.util.find_spec(mod) is None))
PY
)
  [[ -z "${MISSING// /}" ]] || pip install --no-cache-dir -i "${PIP_INDEX}" ${MISSING}
fi

if [[ -z "${BASELINE_CONFIG_JSON:-}" ]]; then
  BASELINE_CONFIG_JSON=$(python -c 'import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))' "${BASELINE_CONFIG_FILE}")
fi
export BASELINE_CONFIG_JSON

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
if [[ "${VLLM_V1}" == 1 && "${SMOKE:-0}" != 1 && "${V1_VERIFIED:-0}" != 1 ]]; then
  echo "vLLM V1 must be smoke-tested first; set V1_VERIFIED=1 only after KL≈0." >&2
  exit 1
fi

NUM_GPUS=${NUM_ACCELERATORS:-8}
[[ "${NNODES:-1}" == 1 ]] || { echo "Baselines require a single node" >&2; exit 1; }
VLLM_WORKERS=$((NUM_GPUS-1))
(( VLLM_WORKERS >= 1 )) || { echo "At least two GPUs are required" >&2; exit 1; }

MODEL_PATH=""; SFT_CKPT=""
IFS=';' read -ra KV <<< "${PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR:-}"
for item in "${KV[@]}"; do
  case "${item}" in actor:*) MODEL_PATH="${item#actor:}";; sft:*) SFT_CKPT="${item#sft:}";; esac
done
MODEL_PATH=${MODEL_PATH:-${MODEL_PATH_OVERRIDE:-}}
SFT_CKPT=${SFT_CKPT:-${SFT_CKPT_OVERRIDE:-}}
[[ -n "${MODEL_PATH}" ]] || { echo "Missing actor model source" >&2; exit 1; }
[[ -n "${SFT_CKPT}" ]] || { echo "Missing SFT checkpoint source" >&2; exit 1; }

EXP_NAME=${EXP_NAME:-baseline_${BASELINE_NAME}_seed${BASELINE_SEED}}
export EXP_NAME
case "${EXP_NAME}" in
  baseline_*) ;;
  *) echo "Baseline EXP_NAME must start with baseline_ to protect RACA checkpoints" >&2; exit 2;;
esac
SAVE_ROOT=${PRIMUS_SAVE_CHECKPOINT_DIR:-$(pwd)/checkpoints}
CKPT_DIR=${SAVE_ROOT}/rl-${EXP_NAME}
MANIFEST=${CKPT_DIR}/baseline_manifest.json
if [[ -d "${CKPT_DIR}" && -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" && ! -f "${MANIFEST}" ]]; then
  echo "Refusing non-empty checkpoint without baseline_manifest.json: ${CKPT_DIR}" >&2
  exit 2
fi
if [[ -f "${MANIFEST}" ]]; then
  python - "${MANIFEST}" <<'PY'
import json, os, sys
saved = json.load(open(sys.argv[1]))
current = json.loads(os.environ["BASELINE_CONFIG_JSON"])
if saved.get("baseline") != os.environ["BASELINE_NAME"]:
    raise SystemExit("checkpoint baseline mismatch: %s" % saved.get("baseline"))
if saved.get("config") != current:
    raise SystemExit("checkpoint baseline config mismatch; use a new EXP_NAME")
PY
fi
mkdir -p "${CKPT_DIR}" /tmp/triton-cache
export BASELINE_METRICS_JSONL="${CKPT_DIR}/baseline_metrics.jsonl"
python - "${MANIFEST}" <<'PY'
import json, os, subprocess, sys
manifest = {
    "baseline": os.environ["BASELINE_NAME"],
    "experiment": os.environ.get("EXP_NAME", ""),
    "seed": int(os.environ.get("BASELINE_SEED", "1")),
    "config": json.loads(os.environ["BASELINE_CONFIG_JSON"]),
    "git_commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True
    ).stdout.strip() or "unknown",
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
PY
export VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH:-/tmp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache}
export WANDB_MODE=${WANDB_MODE:-offline}
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python baselines/preflight.py --method "${BASELINE_NAME}"
python baselines/test_baselines.py
python test_grader.py
MAX_STEPS=${MAX_STEPS:-80}
EXTRA_ARGS=""
if [[ "${SMOKE:-0}" == 1 ]]; then
  MAX_STEPS=2
  EXTRA_ARGS="agentic.eval_samples=20 agentic.eval_aime_samples=10 agentic.val_before_train=false"
fi
ENTRY=baselines/train_credit.py
[[ "${BASELINE_NAME}" != single_agent_grpo ]] || ENTRY=baselines/train_single_agent_grpo.py
[[ "${BASELINE_NAME}" != fixed_four_role_grpo ]] || ENTRY=baselines/train_fixed_four_role_grpo.py

python "${ENTRY}" \
  exp_name="${EXP_NAME}" data=math \
  llm.model_path="${MODEL_PATH}" sft_checkpoint="${SFT_CKPT}" \
  ckpt_dir="${CKPT_DIR}" \
  agentic.vllm_num_workers="${VLLM_WORKERS}" \
  agentic.vllm_use_v1="${VLLM_V1}" \
  agentic.vllm_enforce_eager="${ENFORCE_EAGER:-true}" \
  agentic.max_steps="${MAX_STEPS}" hydra.run.dir=. \
  ${EXTRA_ARGS} "$@"
