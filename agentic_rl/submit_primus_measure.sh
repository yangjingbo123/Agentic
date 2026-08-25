#!/bin/bash
# 离线通道测量（measure_channels.py）—— Primus 作业启动脚本
#
# Primus 作业命令（单行）：
#   bash /root/code/med-mul/agentic_rl/submit_primus_measure.sh
#
# 与 submit_primus.sh（训练）的差别：
#   - 资源只需 2 张卡：cuda:0 挂 HF 模型（仅用于向 vLLM 同步 LoRA，不做前反向），
#     第 2 张卡跑 vLLM 推理。给 8 卡也只用 2 张，脚本不会因此报错。
#   - 不训练 → 不写 checkpoint、不需要 wandb、不存在 V1 引擎的 kl 对齐风险，
#     所以这里不像训练脚本那样拦 V1（训练侧拦是因为 old_logprobs 是 ratio 分母）。
#   - 唯一产出是 stdout。Primus 作业结束后容器日志会丢，故强制 tee 到持久化盘。
#
# 模型源（PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR，格式 key:value;key:value）：
#   actor:<Qwen3-8B HF 权重路径>
#   sft:<被测 checkpoint 路径>  —— SFT 平铺结构，或 RL checkpoint（含 proposer/ 等
#   子目录，evaluate.load_finetuned_models 会逐 adapter 加载）皆可。
#   要测别的 checkpoint 用 CKPT_OVERRIDE=<path> 覆盖。
#
# 建议先 SMOKE=1 跑一遍（n=24，几分钟）确认加载链路通，再提正式作业。

set -xeuo pipefail

cd "$(dirname "$0")" || exit 1
echo "REPO = $(pwd)"

# ---------------------------------------------------------------------------
# 依赖自举（同训练脚本；ALLOW_PIP=0 关闭）
# ---------------------------------------------------------------------------
if [[ "${ALLOW_PIP:-1}" == "1" ]]; then
    PIP_INDEX=${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}
    MISSING=$(python - <<'EOF'
import importlib.util
mods = {"torch": "torch", "transformers": "transformers", "peft": "peft",
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
# vLLM 引擎选择：≥0.10 已删 V0，必须走 V1
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
VLLM_V1=${VLLM_USE_V1_OVERRIDE:-$(echo "${VLLM_PROBE}" | awk '{print $2}')}
echo "vLLM ${VLLM_VER} → engine V${VLLM_V1}"
if [[ "${VLLM_VER}" == "unknown" ]]; then
    echo "!! 无法 import vllm（探测详情：${VLLM_PROBE}）——镜像缺 vllm 或安装损坏" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 资源：需 ≥2 卡。训练模型固定 device_map="cuda:0"，vLLM 拿第 2 张。
# ---------------------------------------------------------------------------
NUM_GPUS=${NUM_ACCELERATORS:-8}
if (( NUM_GPUS < 2 )); then
    echo "!! 至少需要 2 张卡（cuda:0 挂 HF 权重 + 1 张跑 vLLM），当前 ${NUM_GPUS}" >&2
    exit 1
fi
VLLM_GPU=${VLLM_GPU:-1}

# ---------------------------------------------------------------------------
# 模型 / checkpoint
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
CKPT=${CKPT_OVERRIDE:-${SFT_CKPT:-}}
[[ -z "${MODEL_PATH}" ]] && { echo "!! 缺 actor 模型源（PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR 或 MODEL_PATH_OVERRIDE）" >&2; exit 1; }
[[ -z "${CKPT}"       ]] && { echo "!! 缺被测 checkpoint（sft:... 或 CKPT_OVERRIDE）" >&2; exit 1; }
echo "MODEL_PATH = ${MODEL_PATH}"
echo "CKPT       = ${CKPT}"

# 基座架构体检：平台模型库曾把内部 qwen3_eum 变体标成 Qwen3-8B，而 LoRA
# adapter 是在原版 qwen3 上训的，架构对不上会在加载时直接炸。这里几秒钟
# 就能拦住，比烧掉一次调度重来便宜。BASE_TYPE_EXPECT=skip 可跳过。
if [[ "${BASE_TYPE_EXPECT:-qwen3}" != "skip" ]]; then
    python - "${MODEL_PATH}" "${BASE_TYPE_EXPECT:-qwen3}" <<'EOF'
import json, sys, pathlib
path, want = sys.argv[1], sys.argv[2]
cfg = pathlib.Path(path) / "config.json"
got = json.loads(cfg.read_text()).get("model_type")
print(f"base model_type = {got} (expect {want})")
if got != want:
    sys.exit(f"!! 基座架构不符：{got} != {want}。LoRA 无法加载，换模型源或设 "
             f"BASE_TYPE_EXPECT={got} 显式放行。")
EOF
fi

# ---------------------------------------------------------------------------
# 临时目录：必须本地盘。sync_lora 每次落 /tmp，vLLM V1 的 ZMQ socket 与
# Triton JIT 产物也都不能放网络挂载（socket 语义 / 可执行映射均不支持）。
# ---------------------------------------------------------------------------
TMP_AVAIL_MB=$(df -Pm /tmp | awk 'NR==2 {print $4}')
echo "/tmp 可用 ${TMP_AVAIL_MB}MB（LoRA sync 峰值约 400MB + Triton 缓存）"
if (( TMP_AVAIL_MB < 2048 )); then
    echo "!! /tmp 不足 2GB。不要改到 OSS（会碎 ZMQ/Triton），换容器内其他本地目录。" >&2
fi
export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
mkdir -p "${VLLM_RPC_BASE_PATH}" "${TRITON_CACHE_DIR}"

export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 不设 CUDA_VISIBLE_DEVICES —— vLLM 子进程自己按 --vllm_gpu 设

# ---------------------------------------------------------------------------
# 输出落持久化盘（容器日志会随作业结束消失，测量结果是唯一产出）
# ---------------------------------------------------------------------------
SAVE_ROOT="${PRIMUS_SAVE_CHECKPOINT_DIR:-$(pwd)}"
MEAS_DIR="${SAVE_ROOT}/measure"
mkdir -p "${MEAS_DIR}"
TAG=${TAG:-$(date +%Y%m%d_%H%M%S)}
LOG="${MEAS_DIR}/measure_${TAG}.log"

# ---------------------------------------------------------------------------
# 采样规模。SMOKE=1 → n=24 / vote_k=3，只验加载与解析链路，数字无统计意义
# ---------------------------------------------------------------------------
N=${N:-300}
VOTE_K=${VOTE_K:-4}
if [[ "${SMOKE:-0}" == "1" ]]; then
    N=24; VOTE_K=3
    echo "== SMOKE 模式：n=24，仅验证链路 =="
fi

echo "== 日志 → ${LOG} =="
python measure_channels.py \
    --model_path "${MODEL_PATH}" \
    --checkpoint "${CKPT}" \
    --n "${N}" \
    --vote_k "${VOTE_K}" \
    --vllm_gpu "${VLLM_GPU}" \
    --vllm_use_v1 "${VLLM_V1}" \
    "$@" 2>&1 | tee "${LOG}"

echo "== done: ${LOG} =="
