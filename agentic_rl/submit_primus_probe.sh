#!/bin/bash
# SFT 模板探针（check_sft_template.py）—— Primus 作业启动脚本
#
# Primus 作业命令（单行）：
#   bash /root/code/med-mul/agentic_rl/submit_primus_probe.sh
#
# ---------------------------------------------------------------------------
# 它回答哪个问题
# ---------------------------------------------------------------------------
# AST 枚举过（不是 grep —— grep -A3 的窗口会漏掉写在第 4 行的关键字，
# training/grpo_trainer.py:237 就这样被误判过一次）：活代码里一共 13 个
# apply_chat_template 调用点，7 处传了 enable_thinking=False，包括 RL 采样的
# agents/agentic_executor.py:106、RL 重算 log-prob 的 grpo_trainer.py:237、
# 本轮验收用的 verify_sft_format.py:84。没传的 6 处里，diag_fix.py:23 是故意的
# （它是那个 A/B 对照的「带 thinking」一侧），diagnose_loss.py:54 /
# diagnose_loss2.py:50 / diagnose_nan.py:46 是一次性排查脚本、不在训练也不在
# 推理路径上。**活路径上唯一不传的就是训练那一步：train_sft.py:37 和 :38。**
#
# train_sft.py:42 是
#   labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids):])[:max_length]
# 也就是「prompt 之后的全部」都被监督。于是只剩两种可能，都是实的：
#
#   ① Qwen3 模板渲染末条 assistant 时也插一个空 think 块 → full_text 是
#      ...assistant\n<think>\n\n</think>\n\n + response，而 prompt_text 停在
#      ...assistant\n，监督区因此从 <think> 开始。SFT 在教模型「先吐一个空块
#      再答题」，可推理时那个块已经在 prompt 里了，模型会吐第二个。这一档更重。
#   ② 模板没插 → 监督区正好等于 response，但推理喂进去的前缀
#      ...assistant\n<think>\n\n</think>\n\n 是 SFT 从没见过的，模型被要求从一个
#      训练时不存在的位置往下续。比 ① 轻，但不是零。
#
# 「毫无后果」需要 SFT prompt 与推理 prompt 逐字节相同，那要求模板在没被告知
# enable_thinking=False 时也主动插块 —— 与这个 flag 的用途矛盾，够不着。
#
# 哪一档成立，只取决于 ${MODEL_PATH}/tokenizer_config.json 里那段 jinja
# （各 Qwen3 快照写法有差异），必须在有 tokenizer 的机器上读，故有此作业。
# 相关事实：训练数据 2864 个 turn 里 response 含 think 标签的是 0 个，所以若
# 走 ①，reasoning_content 必为空，渲染出来正好是 <think>\n\n</think>\n\n。
#
# ---------------------------------------------------------------------------
# 与 submit_primus_measure.sh / submit_primus_sft.sh 的差别
# ---------------------------------------------------------------------------
#   - **不需要 GPU**：只 AutoTokenizer.from_pretrained，不加载权重、不前反向、
#     不起 vLLM。脚本显式 CUDA_VISIBLE_DEVICES=""，给多少卡都不碰，
#     所以也不像那两个脚本那样检查 NUM_ACCELERATORS。
#   - 依赖只要 transformers（+tokenizers），不装 torch / vllm / peft。
#   - 唯一产出是 stdout，故同样强制 tee 到持久化盘。
#   - 退出码：正常 0（探针刻意只读数、不拦作业）。唯一的非 0 是 exit 2，
#     含义是 train_sft.py 那两处调用的**实参形状漂了** —— 形状漂了说明探针在量
#     一个已经不存在的东西，读数无意义，必须让作业红。注意「缺陷被修好」
#     （给 train_sft.py 补上 enable_thinking）**不算**漂：探针会打一条退休提示
#     然后正常退出，不会出现「修 bug 反而挂作业」。
#
# ---------------------------------------------------------------------------
# 模型源与开关
# ---------------------------------------------------------------------------
# 模型源（PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR，格式 key:value;key:value）：
#   actor:<Qwen3-8B HF 权重路径>
#   —— 与 submit_primus_sft.sh 同一套解析逻辑，这是刻意的：保证探针读到的
#   tokenizer 就是 train_sft.py 将要用的那一个，否则量了也不能推论。
#   无则回退 MODEL_PATH_OVERRIDE。
#
#   SKIP_CENSUS=1   只跑 A 段（模板对齐，几秒）。不设则 B 段用真 tokenizer 走完
#                   全部 turn（几十秒），给出 max_len 下截断与「零梯度」的真值，
#                   用来替掉开发机上 chars/2.2 的估算（估算是 44 超限 / 8 零梯度）。
#   BASE_TYPE_EXPECT=skip  跳过基座架构体检。
#   ALLOW_PIP=0     不自动补依赖。
#   TAG=<str>       日志文件名后缀，默认时间戳。

set -xeuo pipefail

cd "$(dirname "$0")" || exit 1
echo "REPO = $(pwd)"

# 明确不占卡：transformers 在某些版本会顺手初始化 CUDA，这里从环境上断掉，
# 免得一个纯 tokenizer 作业白占一张 GPU 的调度额度。
export CUDA_VISIBLE_DEVICES=""

# ---------------------------------------------------------------------------
# 依赖自举（同 measure 脚本，但只要 tokenizer 相关的两个包）
# ---------------------------------------------------------------------------
if [[ "${ALLOW_PIP:-1}" == "1" ]]; then
    PIP_INDEX=${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}
    MISSING=$(python - <<'EOF'
import importlib.util
mods = {"transformers": "transformers", "tokenizers": "tokenizers"}
print(" ".join(p for m, p in mods.items() if importlib.util.find_spec(m) is None))
EOF
)
    if [[ -n "${MISSING// /}" ]]; then
        echo "== 镜像缺依赖: ${MISSING} → pip 安装（index=${PIP_INDEX}） =="
        pip install --no-cache-dir -i "${PIP_INDEX}" ${MISSING}
    fi
fi

# ---------------------------------------------------------------------------
# 模型路径（与 submit_primus_sft.sh:55-62 逐行同构，勿单边改）
# ---------------------------------------------------------------------------
MODEL_PATH=""
IFS=';' read -ra _kv <<< "${PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR:-}"
for item in "${_kv[@]}"; do
    [[ "${item}" == actor:* ]] && MODEL_PATH="${item#actor:}"
done
MODEL_PATH=${MODEL_PATH:-${MODEL_PATH_OVERRIDE:-}}
[[ -z "${MODEL_PATH}" ]] && {
    echo "!! 缺 actor 模型源（PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR 或 MODEL_PATH_OVERRIDE）" >&2
    exit 1
}
echo "MODEL_PATH = ${MODEL_PATH}"

# 基座架构体检（抄 submit_primus_measure.sh:96-110）。平台模型库曾把内部
# qwen3_eum 变体标成 Qwen3-8B。对本探针格外要紧：变体的 chat_template 可能
# 是另一段 jinja，那读回来的结论就推不到 train_sft.py 身上去。
if [[ "${BASE_TYPE_EXPECT:-qwen3}" != "skip" ]]; then
    python - "${MODEL_PATH}" "${BASE_TYPE_EXPECT:-qwen3}" <<'EOF'
import json, sys, pathlib
path, want = sys.argv[1], sys.argv[2]
cfg = pathlib.Path(path) / "config.json"
got = json.loads(cfg.read_text()).get("model_type")
print(f"base model_type = {got} (expect {want})")
if got != want:
    sys.exit(f"!! 基座架构不符：{got} != {want}。chat_template 可能也不是同一段 "
             f"jinja，读数无法推论；换模型源或设 BASE_TYPE_EXPECT={got} 显式放行。")
EOF
fi

# ---------------------------------------------------------------------------
# 数据：v23 = v2 + v3 现场派生，且**每次重建**
#
# 派生文件不入库，生成命令与 submit_primus_sft.sh 完全一致（勿单边改，否则
# 探针量的就不是 SFT 实际吃的那份字节）。刻意不做「已存在就跳过」的缓存：
# 在开发机上踩过一次 —— v23 比它的 _m1 源文件旧 41 分钟，md5 从 9f12c8ba…
# 变成 56809c6d…，拿旧文件量出来的截断数是错的。重建只要几秒，不值得缓存。
# ---------------------------------------------------------------------------
for f in data/sft_train_v2.jsonl data/sft_train_v3.jsonl; do
    [[ -f "${f}" ]] || { echo "!! 缺 ${f}（需先提交入仓库）" >&2; exit 1; }
done
python data/prepare_sft.py \
    --inputs data/sft_train_v2.jsonl data/sft_train_v3.jsonl \
    || { echo "!! SFT 派生校验失败，探针读数会落在未校验的数据上，勿继续" >&2; exit 1; }
cat data/sft_train_v2_m1.jsonl data/sft_train_v3_m1.jsonl > data/sft_train_v23.jsonl
wc -l data/sft_train_v23.jsonl
md5sum data/sft_train_v23.jsonl || md5 data/sft_train_v23.jsonl || true

# ---------------------------------------------------------------------------
# 输出落持久化盘（容器日志会随作业结束消失，读数是唯一产出）
# ---------------------------------------------------------------------------
SAVE_ROOT="${PRIMUS_SAVE_CHECKPOINT_DIR:-$(pwd)}"
PROBE_DIR="${SAVE_ROOT}/probe"
mkdir -p "${PROBE_DIR}"
TAG=${TAG:-$(date +%Y%m%d_%H%M%S)}
LOG="${PROBE_DIR}/template_probe_${TAG}.log"

EXTRA=()
[[ "${SKIP_CENSUS:-0}" == "1" ]] && EXTRA+=(--skip-census)

echo "== 日志 → ${LOG} =="
# ${EXTRA[@]+...} 这个写法是给老 bash 的：set -u 下直接展开空数组会报
# unbound variable。
python check_sft_template.py \
    --model_path "${MODEL_PATH}" \
    --data data/sft_train_v23.jsonl \
    ${EXTRA[@]+"${EXTRA[@]}"} "$@" 2>&1 | tee "${LOG}"

echo "== done: ${LOG} =="
