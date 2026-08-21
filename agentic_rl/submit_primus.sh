#!/bin/bash
# Agentic RL v3 训练 —— Primus 作业启动脚本
#
# Primus 作业命令（单行）：
#   bash /root/code/med-mul/agentic_rl/submit_primus.sh
#
# 作业配置约定（参照 med_rl recipe/uni_sgs 的 Primus 模式）：
#   - 资源：单机 1×8 GPU（A100-80G / H100）。本框架是「cuda:0 训练 +
#     其余卡 vLLM 推理」的单机架构，不支持多机（NNODES 必须为 1）。
#   - 模型：PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR 需含
#       actor:<Qwen3-8B 的 HF 权重挂载路径>
#       sft:<SFT v3 checkpoint 挂载路径>（扁平结构，含 adapter_model.safetensors）
#     无 sft 源时回退 SFT_CKPT_OVERRIDE 环境变量。
#   - 输出：写入 PRIMUS_SAVE_CHECKPOINT_DIR/rl-${EXP_NAME}；抢占重排后
#     只要 EXP_NAME 不变即自动 resume（train.py 读同路径 trainer_state.json）。
#   - 数据：随代码仓库走（data/*.jsonl 共 ~10MB），不占用 PRIMUS_DATA_PATH。
#
# 镜像依赖（缺一不可）：
#   torch>=2.6  vllm==0.9.x（代码强制 VLLM_USE_V1=0 走 V0 引擎）
#   transformers（旧新版皆可，代码用 torch_dtype= 兼容写法） peft safetensors
#   bitsandbytes（AdamW8bit） flash-attn（trainable_llm 用 flash_attention_2）
#   hydra-core omegaconf wandb numpy
#   若镜像无 flash-attn：删 llm/trainable_llm.py 的 attn_implementation 参数（退化 sdpa）。
#
# 首次上线建议先提交一个 SMOKE=1 的验证作业（CPU 自检 + 2 步训练即退出），
# 通过后再提正式 200 步作业 —— 同 med_rl 的 verify → train 两段式惯例。

set -xeuo pipefail

cd "$(dirname "$0")" || exit 1
echo "REPO = $(pwd)"

# ---------------------------------------------------------------------------
# 依赖自举：镜像缺包时从 pypi 镜像安装（ALLOW_PIP=0 关闭）。
# RL 额外需要 vllm 0.9.x（代码强制 V0 引擎，≥0.10 已删 V0）与 bitsandbytes。
# flash-attn 故意不在列表：pip 安装需编译半小时，代码已回退 sdpa。
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
# 资源（Primus 注入）
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
# 模型 / SFT checkpoint：从 PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR 解析
# （格式 key:value;key:value，同 med_rl run_bt_rm_train.sh 的约定）
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
# 输出 / 临时目录：全部落持久化挂载
# ---------------------------------------------------------------------------
EXP_NAME=${EXP_NAME:-v3_primus}
SAVE_ROOT="${PRIMUS_SAVE_CHECKPOINT_DIR:-$(pwd)/checkpoints}"
CKPT_DIR="${SAVE_ROOT}/rl-${EXP_NAME}"
mkdir -p "${CKPT_DIR}"

# vLLM LoRA sync 每个 step 都写临时目录；容器 /tmp 常为小 rootfs/内存盘，
# 长跑必然写满 → 重定向到挂载盘（tempfile 遵循 TMPDIR）。
export TMPDIR="${SAVE_ROOT}/tmp-${EXP_NAME}"
mkdir -p "${TMPDIR}"

# Primus 容器无外网：wandb 离线，run 数据落在工作目录，事后手动 sync
export WANDB_MODE=${WANDB_MODE:-offline}
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 不设 CUDA_VISIBLE_DEVICES —— train.py 的 slot 逻辑按平台注入的可见卡自适应

# ---------------------------------------------------------------------------
# 启动前自检（CPU，秒级；失败即熔断，不烧 GPU 时）
# ---------------------------------------------------------------------------
python preflight_v2.py --sft-ckpt "${SFT_CKPT}"
python test_raca_v2.py
python test_grader.py

# ---------------------------------------------------------------------------
# 训练。SMOKE=1 → 2 步小跑验证保存/续训链路（不评估收敛性）
# ---------------------------------------------------------------------------
MAX_STEPS=${MAX_STEPS:-200}
EXTRA_ARGS=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    MAX_STEPS=2
    EXTRA_ARGS="agentic.eval_samples=20 agentic.val_before_train=false"
    echo "== SMOKE 模式：max_steps=2，验证 rollout/更新/保存链路 =="
fi

python train.py \
    exp_name="${EXP_NAME}" \
    data=math \
    llm.model_path="${MODEL_PATH}" \
    sft_checkpoint="${SFT_CKPT}" \
    ckpt_dir="${CKPT_DIR}" \
    agentic.vllm_num_workers="${VLLM_WORKERS}" \
    agentic.max_steps="${MAX_STEPS}" \
    hydra.run.dir=. \
    ${EXTRA_ARGS} \
    "$@"

echo "== done: checkpoints at ${CKPT_DIR} =="
ls -la "${CKPT_DIR}"
