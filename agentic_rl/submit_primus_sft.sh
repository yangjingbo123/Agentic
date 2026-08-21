#!/bin/bash
# Agentic SFT v3（v23 数据）—— Primus 作业启动脚本
#
# Primus 作业命令（单行）：
#   bash /root/code/med-mul/agentic_rl/submit_primus_sft.sh
#
# 作业配置约定：
#   - 资源：单机 1 GPU（A100-80G/H100 一张即可；8B bf16 + LoRA + grad-ckpt
#     实测 <40GB）。若平台最小配额是 8 卡，脚本会自动限制到第一张卡——
#     train_sft.py 的 device_map="auto" 在多卡下会做朴素模型并行（层间串行，
#     慢且浪费），必须限单卡。
#   - 模型：PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR 含 actor:<Qwen3-8B 路径>；
#     无则回退 MODEL_PATH_OVERRIDE。
#   - 数据：data/sft_train_v2.jsonl 与 data/sft_train_v3.jsonl 必须都在仓库中
#     （v23 是纯派生文件，本脚本现场 cat 生成，不入库）。
#   - 输出：PRIMUS_SAVE_CHECKPOINT_DIR/sft_v3（扁平结构）。后续 RL 作业以
#     sft:<该路径> 引用。
#   - VERIFY=1（默认）：训后跑 verify_sft_format.py 格式验收，不过则作业失败
#     ——避免拿着格式坏的 SFT 去启 RL。

set -xeuo pipefail

cd "$(dirname "$0")" || exit 1
echo "REPO = $(pwd)"

# ---------------------------------------------------------------------------
# 依赖自举：镜像缺包时从 pypi 镜像安装（ALLOW_PIP=0 关闭）。
# 首选仍是直接选带齐依赖的镜像（如 med_rl BT-RM 作业同款）；本段是兑底。
# flash-attn 故意不在列表：pip 安装需编译半小时，代码已回退 sdpa。
# ---------------------------------------------------------------------------
if [[ "${ALLOW_PIP:-1}" == "1" ]]; then
    PIP_INDEX=${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}
    MISSING=$(python - <<'EOF'
import importlib.util
mods = {"torch": "torch", "transformers": "transformers", "hydra": "hydra-core",
        "omegaconf": "omegaconf", "wandb": "wandb", "peft": "peft",
        "safetensors": "safetensors", "accelerate": "accelerate", "numpy": "numpy"}
print(" ".join(p for m, p in mods.items() if importlib.util.find_spec(m) is None))
EOF
)
    if [[ -n "${MISSING// /}" ]]; then
        echo "== 镜像缺依赖: ${MISSING} → pip 安装（index=${PIP_INDEX}） =="
        pip install --no-cache-dir -i "${PIP_INDEX}" ${MISSING}
    fi
fi

# ---------------------------------------------------------------------------
# 单卡约束（见头部说明）
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------------------------------------------------
# 模型路径
# ---------------------------------------------------------------------------
MODEL_PATH=""
IFS=';' read -ra _kv <<< "${PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR:-}"
for item in "${_kv[@]}"; do
    [[ "${item}" == actor:* ]] && MODEL_PATH="${item#actor:}"
done
MODEL_PATH=${MODEL_PATH:-${MODEL_PATH_OVERRIDE:-}}
[[ -z "${MODEL_PATH}" ]] && { echo "!! 缺 actor 模型源" >&2; exit 1; }
echo "MODEL_PATH = ${MODEL_PATH}"

# ---------------------------------------------------------------------------
# 数据：v23 = v2 + v3 现场派生（派生文件不入库，生成命令即此处）
# ---------------------------------------------------------------------------
for f in data/sft_train_v2.jsonl data/sft_train_v3.jsonl; do
    [[ -f "${f}" ]] || { echo "!! 缺 ${f}（需先提交入仓库）" >&2; exit 1; }
done
cat data/sft_train_v2.jsonl data/sft_train_v3.jsonl > data/sft_train_v23.jsonl
wc -l data/sft_train_v2.jsonl data/sft_train_v3.jsonl data/sft_train_v23.jsonl

# v23 快速体检：episode 结构、v2 格式可解析率、max_len 截断率
python - <<'EOF'
import json, sys
sys.path.insert(0, ".")
from agents.parsing import parse_decision, parse_interaction

n_turn = ok_d = n_d = ok_i = n_i = 0
lens = []
for line in open("data/sft_train_v23.jsonl"):
    ep = json.loads(line)
    for t in ep.get("turns", []):
        n_turn += 1
        lens.append(len(t["system"]) + len(t["user"]) + len(t["response"]))
        if t.get("role_name") == "controller":
            n_d += 1; ok_d += parse_decision(t["response"]) in ("continue", "stop")
        else:
            n_i += 1; ok_i += parse_interaction(t["response"])[0] in ("none", "request", "challenge")
trunc = sum(1 for l in lens if l / 2.2 > 1024) / max(len(lens), 1)
print(f"v23: {n_turn} turns, controller {ok_d}/{n_d}, worker {ok_i}/{n_i}, "
      f"max_len=1024 预计截断率 {trunc:.1%}")
assert ok_d == n_d and ok_i == n_i, "存在不可解析 turn"
assert trunc < 0.02, f"截断率 {trunc:.1%} 超阈"
EOF

# ---------------------------------------------------------------------------
# 输出目录（持久化挂载，扁平结构）
# ---------------------------------------------------------------------------
SAVE_ROOT="${PRIMUS_SAVE_CHECKPOINT_DIR:-$(pwd)/checkpoints}"
SAVE_DIR="${SAVE_ROOT}/sft_v3"
if [[ -e "${SAVE_DIR}/proposer" ]]; then
    echo "!! ${SAVE_DIR} 存在 role 子目录 —— load_trainable_models 会优先读它导致静默覆盖，请清理" >&2
    exit 1
fi
mkdir -p "${SAVE_DIR}"

export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# 训练（超参对齐训练机上验证过的 sft_v3 配方）
# ---------------------------------------------------------------------------
python train_sft.py \
    exp_name="${EXP_NAME:-sft_v3_primus}" \
    data=math \
    llm.model_path="${MODEL_PATH}" \
    ++data.sft_path=data/sft_train_v23.jsonl \
    +sft.save_dir="${SAVE_DIR}" \
    +sft.max_len=1024 \
    +sft.epochs="${EPOCHS:-3}" \
    +sft.batch="${BATCH:-2}" \
    +sft.lr="${LR:-2e-5}" \
    hydra.run.dir=. \
    "$@"

# ---------------------------------------------------------------------------
# 验收：目录结构 + 权重存在 + （可选）四角色格式可解析率
# ---------------------------------------------------------------------------
ls -la "${SAVE_DIR}"
[[ -f "${SAVE_DIR}/adapter_model.safetensors" ]] || { echo "!! 无 adapter_model.safetensors" >&2; exit 1; }
[[ -d "${SAVE_DIR}/proposer" ]] && { echo "!! 出现 role 子目录（RL 加载优先级冲突）" >&2; exit 1; }

if [[ "${VERIFY:-1}" == "1" ]]; then
    # 走 RL 真实加载路径的格式验收（可解析率 <70% 时 exit 1，熔断后续 RL）
    python verify_sft_format.py --ckpt "${SAVE_DIR}" --model_path "${MODEL_PATH}" \
        || { echo "!! SFT 格式验收未通过，勿用此 ckpt 启 RL" >&2; exit 1; }
fi

echo "== done: SFT checkpoint at ${SAVE_DIR} =="
echo "== 后续 RL 作业引用：PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR 加 sft:${SAVE_DIR} =="
