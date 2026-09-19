#!/bin/bash
# RACA 组件消融 —— Primus 作业启动脚本（Table `tab:raca-ablation`）
#
# 与 submit_primus.sh 共用镜像/资源/模型约定，唯一区别是：改用
# train_ablation.py（非侵入式 monkeypatch 掉 compute_raca_advantages），
# 按 RACA_ABLATION 依次跑三个消融。RACA 原有代码零改动。
#
# Primus 作业命令（单行）：
#   bash /root/code/med-mul/agentic_rl/submit_primus_ablation.sh
#
# 三个消融（默认全跑，单机 8 卡只能串行）：
#   no_role     去掉 role 条件：不同角色奖励混入同一归一化组
#   no_context  去掉 σ 条件：不同黑板状态的 turn 一起比较
#   no_channel  去掉功能通道：proposer 主 turn 用单一标量优势
#
# 环境变量：
#   ABLATIONS   覆盖要跑的模式，空格分隔（默认 "no_role no_context no_channel"）
#   EXP_PREFIX  实验名前缀（默认 v34_raca_abl / SMOKE 时加 _smoke）
#   MAX_STEPS   训练步数（默认 80；SMOKE=1 强制 2）
#   SMOKE=1     2 步小跑，验证 rollout/更新/保存链路
#   其余（PRIMUS_*, SFT_CKPT_OVERRIDE, MODEL_PATH_OVERRIDE, ALLOW_PIP,
#   VLLM_USE_V1_OVERRIDE, V1_VERIFIED, WANDB_MODE ...）与 submit_primus.sh 一致。
#
# 完整对照建议：先单独跑一次未消融的 RACA（submit_primus.sh）作为基线，
# 三个消融各自与它对比，填 tab:raca-ablation 的 MATH-L5 / AIME 两列。

set -xeuo pipefail

cd "$(dirname "$0")" || exit 1
echo "REPO = $(pwd)"

# ---------------------------------------------------------------------------
# 依赖自举（同 submit_primus.sh）
# ---------------------------------------------------------------------------
if [[ "${ALLOW_PIP:-1}" == "1" ]]; then
    PIP_INDEX=${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}
    MISSING=$(python - <<'EOF'
import importlib.util
mods = {"torch": "torch", "transformers": "transformers", "hydra": "hydra-core",
        "omegaconf": "omegaconf", "wandb": "wandb", "peft": "peft",
        "safetensors": "safetensors", "accelerate": "accelerate",
        "numpy": "numpy", "vllm": "vllm==0.9.2", "bitsandbytes": "bitsandbytes"}
print(" ".join(p for m, p in mods.items() if importlib.util.find_spec(m) is None))
EOF
)
    if [[ -n "${MISSING// /}" ]]; then
        echo "== 镜像缺依赖: ${MISSING} → pip 安装（index=${PIP_INDEX}） =="
        pip install --no-cache-dir -i "${PIP_INDEX}" ${MISSING}
    fi
fi

# ---------------------------------------------------------------------------
# vLLM 引擎选择（同 submit_primus.sh）
# ---------------------------------------------------------------------------
VLLM_PROBE=$(python - <<'EOF'
try:
    import vllm
    ver = vllm.__version__
    major, minor = (int(x) for x in ver.split(".")[:2])
    print(f"{ver} {'0' if (major, minor) < (0, 10) else '1'}")
except Exception as e:
    print(f"unknown 0  # probe failed: {type(e).__name__}: {e}")
EOF
)
VLLM_VER=$(echo "${VLLM_PROBE}" | awk '{print $1}')
VLLM_V1=$(echo "${VLLM_PROBE}" | awk '{print $2}')
VLLM_V1=${VLLM_USE_V1_OVERRIDE:-${VLLM_V1}}
echo "vLLM ${VLLM_VER} → engine V${VLLM_V1}"
if [[ "${VLLM_VER}" == "unknown" ]]; then
    echo "!! 无法 import vllm（探测详情：${VLLM_PROBE}）——镜像缺 vllm 或安装损坏" >&2
    exit 1
fi
if [[ "${VLLM_V1}" == "1" && "${SMOKE:-0}" != "1" ]]; then
    echo "!! 首次在 V1 引擎上跑建议先 SMOKE=1 验收（看首步 kl 是否 ≈0）；"
    echo "   已验过可设 V1_VERIFIED=1 跳过本提示。" >&2
    [[ "${V1_VERIFIED:-0}" == "1" ]] || exit 1
fi

# ---------------------------------------------------------------------------
# 资源（Primus 注入，同 submit_primus.sh）
# ---------------------------------------------------------------------------
NUM_GPUS=${NUM_ACCELERATORS:-8}
if [[ "${NNODES:-1}" != "1" ]]; then
    echo "!! 本框架为单机架构（cuda:0 训练 + N-1 卡 vLLM），NNODES 必须为 1" >&2
    exit 1
fi
VLLM_WORKERS=$(( NUM_GPUS - 1 ))
if (( VLLM_WORKERS < 1 )); then
    echo "!! 至少需要 2 张卡（1 训练 + 1 推理），当前 NUM_GPUS=${NUM_GPUS}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 模型 / SFT checkpoint（同 submit_primus.sh）
# ---------------------------------------------------------------------------
MODEL_PATH=""
SFT_CKPT=""
IFS=';' read -ra _kv <<< "${PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR:-}"
for item in "${_kv[@]}"; do
    case "${item}" in
        actor:*) MODEL_PATH="${item#actor:}" ;;
        sft:*)   SFT_CKPT="${item#sft:}" ;;
    esac
done
MODEL_PATH=${MODEL_PATH:-${MODEL_PATH_OVERRIDE:-}}
SFT_CKPT=${SFT_CKPT:-${SFT_CKPT_OVERRIDE:-}}
[[ -z "${MODEL_PATH}" ]] && { echo "!! 缺 actor 模型源（PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR 或 MODEL_PATH_OVERRIDE）" >&2; exit 1; }
[[ -z "${SFT_CKPT}"   ]] && { echo "!! 缺 sft checkpoint 源（sft:... 或 SFT_CKPT_OVERRIDE）" >&2; exit 1; }
echo "MODEL_PATH = ${MODEL_PATH}"
echo "SFT_CKPT   = ${SFT_CKPT}"

# ---------------------------------------------------------------------------
# 临时目录 / 环境（同 submit_primus.sh，必须本地盘）
# ---------------------------------------------------------------------------
TMP_AVAIL_MB=$(df -Pm /tmp | awk 'NR==2 {print $4}')
echo "/tmp 可用 ${TMP_AVAIL_MB}MB（LoRA sync 峰值需约 400MB + Triton 缓存）"
if (( TMP_AVAIL_MB < 2048 )); then
    echo "!! /tmp 不足 2GB。不要把 TMPDIR 改到 OSS（会碎 ZMQ/Triton），"
    echo "   请改用容器内其他本地目录，例如 TMPDIR=/root/tmp 并确保已 mkdir。" >&2
fi
export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-/tmp}"
mkdir -p "${VLLM_RPC_BASE_PATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
mkdir -p "${TRITON_CACHE_DIR}"
export WANDB_MODE=${WANDB_MODE:-offline}
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# 启动前自检（CPU，秒级；含消融专用单测）
# ---------------------------------------------------------------------------
python preflight_v2.py --sft-ckpt "${SFT_CKPT}"
python test_raca_v2.py
python test_grader.py
python -m pytest test_raca_ablations.py -q 2>/dev/null \
    || python test_raca_ablations.py \
    || echo "== 注意：消融单测未通过 pytest 且无独立入口，请检查 =="

# ---------------------------------------------------------------------------
# 训练：按 RACA_ABLATION 依次跑
# ---------------------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
    DEFAULT_PREFIX=v34_raca_abl_smoke
else
    DEFAULT_PREFIX=v34_raca_abl
fi
EXP_PREFIX=${EXP_PREFIX:-${DEFAULT_PREFIX}}
ABLATIONS=${ABLATIONS:-"no_role no_context no_channel"}

MAX_STEPS=${MAX_STEPS:-80}
EXTRA_ARGS=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    MAX_STEPS=2
    EXTRA_ARGS="agentic.eval_samples=20 agentic.eval_aime_samples=10 agentic.val_before_train=false"
    echo "== SMOKE 模式：每个消融 max_steps=2 =="
fi

SAVE_ROOT="${PRIMUS_SAVE_CHECKPOINT_DIR:-$(pwd)/checkpoints}"

for ABL in ${ABLATIONS}; do
    case "${ABL}" in
        no_role|no_context|no_channel) ;;
        *) echo "!! 未知消融模式 ${ABL}（可选 no_role/no_context/no_channel）" >&2; exit 1 ;;
    esac
    EXP_NAME="${EXP_PREFIX}_${ABL}"
    CKPT_DIR="${SAVE_ROOT}/rl-${EXP_NAME}"
    mkdir -p "${CKPT_DIR}"
    echo "======================================================================"
    echo "== 消融 ${ABL} → exp=${EXP_NAME} ckpt=${CKPT_DIR}"
    echo "======================================================================"

    RACA_ABLATION="${ABL}" python train_ablation.py \
        exp_name="${EXP_NAME}" \
        data=math \
        llm.model_path="${MODEL_PATH}" \
        sft_checkpoint="${SFT_CKPT}" \
        ckpt_dir="${CKPT_DIR}" \
        agentic.vllm_num_workers="${VLLM_WORKERS}" \
        agentic.vllm_use_v1="${VLLM_V1}" \
        agentic.vllm_enforce_eager="${ENFORCE_EAGER:-true}" \
        agentic.max_steps="${MAX_STEPS}" \
        hydra.run.dir=. \
        ${EXTRA_ARGS} \
        "$@"

    echo "== done: ${ABL} checkpoints at ${CKPT_DIR} =="
    ls -la "${CKPT_DIR}"
done

echo "== 全部消融完成：${ABLATIONS} =="
