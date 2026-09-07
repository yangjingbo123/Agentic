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
# max_len —— **唯一来源**
# ---------------------------------------------------------------------------
# 下面的体检闸门与 train_sft.py 的 +sft.max_len 都读这一个变量。此前这两处各写
# 一个字面量 1024，天然会漂：改了训练参数而忘了改闸门，闸门就在守一条谁也不用
# 的线，而且不会报错。
# 1024 → 1536 的依据是集群上真 tokenizer 的实测（见下面注释块的表）：超限
# 164 → 8，零梯度 32 → 0。
MAX_LEN=${MAX_LEN:-1536}
echo "MAX_LEN    = ${MAX_LEN}"

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
# ── 关于下面这段历史：它记的数曾经全是**错的**，因为尺子错了 ────────────────
# 直到 v3.2 第八轮，这里的截断率一直用 `chars/2.2 > max_len` 这把代理尺子量。
# 在集群上用**真 tokenizer** 跑 check_sft_template.py（见 submit_primus_probe.sh）
# 之后拿到真值，两者差得很远：
#
#              代理尺 chars/2.2      真 tokenizer
#   @1024        44 (1.54%)           164 (5.73%)   零梯度 32
#   @1536         3 (0.10%)             8 (0.28%)   零梯度  0
#   @2048         —                     0 (0.00%)   零梯度  0
#   （full token 数：中位 426，最长 1797）
#
# 代理尺偏乐观约 3.7 倍，而阈值是 2%——也就是说 1024 下真实截断 5.73% 早已越阈，
# 闸门却因为尺子偏软而放行。这是本项目第三次踩「两把尺子漂移」（前两次：
# evaluate.py 的三把尺子、以及这里的 parse_interaction 返回 "none" 让 assert 恒真）。
# 所以本轮把这把尺子换成真 tokenizer，见下面的 heredoc。
#
# 「零梯度」是比截断率更硬的读数，旧尺子完全没有这个概念：train_sft.py:42 的
#   labels = ([-100]*len(prompt_ids) + full_ids[len(prompt_ids):])[:max_length]
# 配合 :96 的 `if response_mask.sum() == 0: continue`，意味着**prompt 本身就超
# max_len** 的 turn 整条不产生任何梯度——比丢尾巴严格更坏。1024 下有 32 条，
# 按角色 verifier 17 / critic 12 / proposer 3 / controller 0（角色内占比 2.3% /
# 1.8% / 0.5% / 0%）。恰好压在 critic 与 verifier 上，而 v3 观测到的「critic
# 无差别 flag、对最终答案无因果通路」就在这两个角色身上——是否足以解释，未测。
#
# 而剩下 132 条「只丢尾巴」的，丢掉的**恰好是 <interaction> 块**：
# `full_ids[:max_length]` 砍尾部，而 M1 把块搬去了尾部。也就是 132/2864 = 4.6%
# 的 turn 在 SFT 阶段压根没被监督过交互决策——正是 M1 与 Eq (12) 要修的东西，
# 且超限的都是黑板上下文最厚（轮次靠后、答案累积多、flaw 长）、最需要判断该不该
# 求助的那些 turn，不是随机 dropout。
# （这 132 条的**角色分布没量到**：探针只对零梯度那一档做了 byrole，超限那一档
#  漏了，是探针的缺口，回头补。）
#
# → 本轮据此把 max_len 1024 → 1536：一个数字换掉 164 里的 156 条，且零梯度
#   32 → 0。代价：train_sft.py:43 把每条 pad 到 1536，激活显存约 1.5 倍；以及
#   KL 参考快照 sft_v3 是 1024 训的，本轮 sft_v3_m1 与它多一处不同。
#   不上 2048 是因为 1536 已把零梯度清零、超限降到 0.28%，再抬只换那 8 条尾巴，
#   而显存是每一条都要付的。
#
# 以下是换尺子之前留下的历史判断，**其中的数字都出自那把偏软的代理尺**，保留是
# 为了让「当时为什么觉得可接受」这条推理链可追溯，不要再引用它的数字：
#   旧记：截断 30 → 34（M1 引入）→ 44（v2 重放引入）；丢块 5 → 34；
#   丢「最终答案：」23 → 22；当时认为可接受的理由是绝对量小、且 34 条里 22 条
#   在 M1 之前就已经是坏样本。真值出来后这条理由不再成立（164 而非 44）。
#
# v3.2（第四轮，信道审计之后重测）：跨角色截断改为**带可见标记**（`CLIP_MARK`）
# 之后，截断率**一点没动**——代理尺下 1024 仍是 44、1536 仍是 3。意料之中：
# 一个标记约十个字符，不足以把任何一条推过线。这个「加标记不影响截断」的结论与
# 尺子无关，继续有效。
#
# 顺带得到一个新读数：**148/2864（5.2%）的 turn，其 `user` 里带着截断标记**，
# 按角色 critic 79 / verifier 65 / controller 3 / proposer 1。critic 与 verifier
# 占绝对多数，因为它们正是读 `发现问题` 与 `对方内容` 的那两个角色——也就是
# v3「critic 说了等于没说」发生的位置，5.2% 是那条信道被掐的真实频率。
# 附带的好处：这 148 条会**教模型认这个标记**（SFT 里就见过带标记的输入），
# 而不是上线才第一次遇到。
# 量它用 grep 而不是计数器：`to_text` 每个 turn 被调用多次（controller / critic /
# verifier / 修正各一次），在函数里计数得到的是调用次数而非事件数；而落盘 prompt
# 里的标记既能算率，又能指出是哪个 turn。
python - "${MODEL_PATH}" "${MAX_LEN}" <<'EOF'
import json, re, sys
sys.path.insert(0, ".")
from agents.parsing import (has_answer_label, parse_decision, parse_reasoning,
                            parse_score)

model_path, max_len = sys.argv[1], int(sys.argv[2])

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
rows = []          # (role, prompt_tokens, full_tokens)
ok = {}
tot = {}
turns = []
for line in open("data/sft_train_v23.jsonl"):
    ep = json.loads(line)
    for t in ep.get("turns", []):
        n_turn += 1
        turns.append(t)
        role = t["role_name"]
        tot[role] = tot.get(role, 0) + 1
        ok[role] = ok.get(role, 0) + key_field_ok(role, t["response"])
for role in sorted(tot):
    print(f"  {role:11s} 关键字段 {ok[role]}/{tot[role]}")
# 要求 100%：prepare_sft.py 已经把解析不出来的整条剔除了，这里若还有漏网，说明
# 两处的判据脱了钩（例如只改了一边的标签），必须停下来查，不能放宽阈值。
bad = {r: (ok[r], tot[r]) for r in tot if ok[r] != tot[r]}
assert not bad, f"存在关键字段解析不出来的 turn：{bad}（prepare_sft.py 的判据与此处脱钩了？）"

# ── 截断普查：真 tokenizer，不用 chars/2.2 ────────────────────────────────
# 换尺子的理由见上面的注释块：代理尺在 1024 上给 1.54%，真值 5.73%，偏乐观
# 3.7 倍，而阈值是 2%——闸门因此在越阈的情况下放行。
# 下面这两行 apply_chat_template 必须与 train_sft.py:37-38 逐字一致（含
# add_generation_prompt 的取值），否则又是一次两把尺子漂移。**注意也一并不传
# enable_thinking**：不是疏漏，是刻意镜像 train_sft.py 的现状，那处缺陷已由
# check_sft_template.py 量到（监督区从空 think 块开始），修法单独一步；这里
# 若擅自补上，量出来的就不是训练实际吃的那份序列。
from transformers import AutoTokenizer
print(f"  tokenizer = {model_path}（真 token 计数，max_len={max_len}）", flush=True)
tok = AutoTokenizer.from_pretrained(model_path)
for t in turns:
    messages = [
        {"role": "system",    "content": t["system"]},
        {"role": "user",      "content": t["user"]},
        {"role": "assistant", "content": t["response"]},
    ]
    full = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=False)
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False,
                                     add_generation_prompt=True)
    rows.append((t["role_name"],
                 len(tok.encode(prompt, add_special_tokens=False)),
                 len(tok.encode(full,   add_special_tokens=False))))

over = [r for r in rows if r[2] > max_len]
# 零梯度：prompt 本身就吃满 max_len ⇒ train_sft.py:42 切完 labels 全是 -100，
# :96 的 `if response_mask.sum() == 0: continue` 把整条跳过。比丢尾巴更坏，
# 因为丢尾巴至少还监督了前半段。
dead = [r for r in over if r[1] >= min(r[2], max_len)]
trunc = len(over) / max(len(rows), 1)


def byrole(rs):
    d = {}
    for r in rs:
        d[r[0]] = d.get(r[0], 0) + 1
    return d


srt = sorted(r[2] for r in rows)
print(f"v23: {n_turn} turns, full token 中位 {srt[len(srt) // 2]}，最长 {srt[-1]}")
print(f"  max_len={max_len}: 超限 {len(over)} ({trunc:.2%}) {byrole(over)}")
print(f"  其中零梯度 {len(dead)} {byrole(dead)}")
# 零梯度必须为 0：这些 turn 一点梯度都不产生，是纯浪费，且系统性偏向黑板上下文
# 最厚的那些（轮次靠后、答案累积多），不是随机 dropout。1536 下实测为 0。
assert not dead, (
    f"有 {len(dead)} 条 turn 零梯度（prompt 本身超 max_len={max_len}）"
    f"{byrole(dead)} —— 抬 max_len 或剔除这些 turn，别让它们静默白跑")
# 超限（丢尾巴）仍留 2% 阈：砍的是尾部，而 M1 把 <interaction> 块搬去了尾部，
# 所以每一条超限都等于一条「交互决策未被监督」的样本。
assert trunc < 0.02, f"截断率 {trunc:.2%} 超阈（真 token 口径）"
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
    +sft.max_len="${MAX_LEN}" \
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
