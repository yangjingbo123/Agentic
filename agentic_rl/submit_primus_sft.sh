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
# v3.2 派生两件事（见 data/prepare_sft.py 的 docstring）：
#   ① M1：把 `<interaction>` 块从开头机械搬到末尾；
#   ② 剔除关键字段解析不出来的 turn —— 实测 44 条，**全在 v2**
#      （proposer 17 缺「最终答案：」/ critic 12 缺「错误分析|无错误」/
#      verifier 15 缺「分数：」），其中绝大多数整条是英文输出。v3 一条不剔。
#      剔除后 2914 → 2870 turn，且 100% 能被 agents/parsing.py 解析。
# 为什么在这里现场派生、而不是把 *_m1.jsonl 提交入库：SFT 数据的 system 字段
# 必须与 llm/prompt_templates.py **逐字节相同**，一旦模板再改而数据文件是死的，
# SFT 教的格式就和 RL 推理时给的格式漂移，且这种漂移不报错、只体现为可解析率
# 缓慢下滑。现场从原始文件派生可以保证两者永远同步。脚本自带逐 turn 校验：
# 只允许「块位置」「system」「整条剔除」三种变化，实质内容改一个字符就 assert
# 失败；该剔除哪些 turn 也在校验时独立重算一遍，多删少删都会当场炸掉。
# ---------------------------------------------------------------------------
for f in data/sft_train_v2.jsonl data/sft_train_v3.jsonl; do
    [[ -f "${f}" ]] || { echo "!! 缺 ${f}（需先提交入仓库）" >&2; exit 1; }
done
python data/prepare_sft.py \
    --inputs data/sft_train_v2.jsonl data/sft_train_v3.jsonl \
    || { echo "!! SFT 派生校验失败，勿继续 SFT" >&2; exit 1; }
cat data/sft_train_v2_m1.jsonl data/sft_train_v3_m1.jsonl > data/sft_train_v23.jsonl
wc -l data/sft_train_v2_m1.jsonl data/sft_train_v3_m1.jsonl data/sft_train_v23.jsonl

# v23 快速体检：episode 结构、v2 格式可解析率、max_len 截断率
#
# M1 的已知副作用（本地实测，chars/2.2 口径，与下面 assert 同一把尺）：
# train_sft.py 的 `full_ids[:max_length]` 砍的是**尾部**，而 M1 把块搬去了尾部。
#   截断 turn：30 → 34（1.03% → 1.17%，仍在 2% 阈内，全是 proposer 26/critic 5/verifier 3）
#   丢 <interaction> 块：5 → 34（+29）      丢「最终答案：」：23 → 22（−1）
# 即：这 34 条 turn 从此不再监督块的生成。判断是可接受的，理由：
#   ① 绝对量 34/2030 = 1.7% 的带块 turn，另外 1996 条监督完好，格式学习靠的是
#      分布主体，不是尾部这几条；
#   ② 这 34 条里有 22 条**连答案标记一起丢**——它们在 M1 之前就已经是坏样本
#      （教模型不输出「最终答案：」），M1 只是把坏的方式从「丢答案」换成
#      「丢答案+丢块」，真正新增的只有 12 条「答案完好、只丢块」；
#   ③ 更干净的做法是把超长 turn 整条剔除（截断样本无论砍哪头都在教「不输出
#      EOS」），但那是与 M1 无关的数据卫生改动，混进来会污染 M1 的归因。
# → 留作后续独立一步：剔除超长 turn，或把 max_len 抬到 1536 后重测。
python - <<'EOF'
import json, re, sys
sys.path.insert(0, ".")
from agents.parsing import (has_answer_label, parse_decision, parse_reasoning,
                            parse_score)

# 每个角色只查**下游真正会读**的那个字段。
# 注意不要用 parse_interaction 来查 worker：它解析失败时返回 "none"，而 "none"
# 是合法取值，assert 恒真——v3.1 的体检就是这么假绿的。同理 parse_decision 有
# "解析失败保守 continue" 的兜底，必须显式确认标签在，不能只看返回值合法。
_CRIT = re.compile(r"错误分析[：:]|无错误|无错")


def key_field_ok(role, resp):
    if role == "controller":
        return "decision:" in resp and parse_decision(resp) in ("continue", "stop")
    if role == "proposer":
        return has_answer_label(resp) and bool(parse_reasoning(resp)[1])
    if role == "critic":
        return bool(_CRIT.search(resp))
    if role == "verifier":
        return parse_score(resp) is not None
    raise SystemExit(f"未知角色：{role}")


n_turn = 0
lens = []
ok = {}
tot = {}
for line in open("data/sft_train_v23.jsonl"):
    ep = json.loads(line)
    for t in ep.get("turns", []):
        n_turn += 1
        lens.append(len(t["system"]) + len(t["user"]) + len(t["response"]))
        role = t["role_name"]
        tot[role] = tot.get(role, 0) + 1
        ok[role] = ok.get(role, 0) + key_field_ok(role, t["response"])
trunc = sum(1 for l in lens if l / 2.2 > 1024) / max(len(lens), 1)
print(f"v23: {n_turn} turns, max_len=1024 预计截断率 {trunc:.1%}")
for role in sorted(tot):
    print(f"  {role:11s} 关键字段 {ok[role]}/{tot[role]}")
# 要求 100%：prepare_sft.py 已经把解析不出来的整条剔除了，这里若还有漏网，说明
# 两处的判据脱了钩（例如只改了一边的标签），必须停下来查，不能放宽阈值。
bad = {r: (ok[r], tot[r]) for r in tot if ok[r] != tot[r]}
assert not bad, f"存在关键字段解析不出来的 turn：{bad}（prepare_sft.py 的判据与此处脱钩了？）"
assert trunc < 0.02, f"截断率 {trunc:.1%} 超阈"
EOF

# ---------------------------------------------------------------------------
# 输出目录（持久化挂载，扁平结构）
# ---------------------------------------------------------------------------
SAVE_ROOT="${PRIMUS_SAVE_CHECKPOINT_DIR:-$(pwd)/checkpoints}"
# v3.2：M1 的 SFT 产物写**新目录**，绝不覆盖 sft_v3。理由有两条：① sft_v3 是
# 当前 RL 的 KL 参考快照，覆盖它等于悄悄改掉正在跑的实验的参考分布；② M1 若
# 不奏效需要能一键回退对照。用 SFT_TAG 覆盖目录名。
SAVE_DIR="${SAVE_ROOT}/${SFT_TAG:-sft_v3_m1}"
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
