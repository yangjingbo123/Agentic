"""RACA v3 SFT 数据构造：用 gold answer 做特权信息，自动生成三类能力数据。

背景（§18.4 三通道测量，2026-08-20）：
  - verifier 分辨力 Δ=+0.012（橡皮图章）——SFT 数据里几乎没有负样本
  - critic flag_rate=0.12——没学过"该不该报错"的先验
  - 修正 acc 0.125 < 裸重采样 0.188——批评不具体 + 锚定效应
三个都是能力缺口而非 RL 问题（三次 int_rate 坍塌均为负期望下的理性收敛）。

三类数据（全部零人工标注，gold 只在构造期使用、不进训练输入）：
  1. verifier 判别对：采样解按 gold 判对错，50/50 配平，分数标签 1.0/0.0
  2. critic 检错：错解 + 特权 prompt（喂 gold）反推具体错误 → 标准 critic
     格式 target；对解配"无错误"，50/50 配平
  3. 修正三元组：错解 + 特权批评 → target 为 gold 引导重生成并验证过的正确解

防泄漏：
  - 特权批评经 _leaks_gold 过滤（批评正文不得出现"答案是/应为 gold"式泄漏）
  - 修正 target 由 parse 后重建为规范格式，丢弃提及"提示/已知正确答案"的推理
  - 数据源 = math_train_rl.jsonl（训练池内部复用，math_test 不受影响，
    eval 完整性不变）

用法（训练机）：
  python generate_sft_v3.py --checkpoint checkpoints/sft_v2 --vllm_gpu 1
  cat data/sft_train_v2.jsonl data/sft_train_v3.jsonl > data/sft_train_v23.jsonl
  # 然后 train_sft 指向 sft_train_v23.jsonl，输出到全新扁平目录 checkpoints/sft_v3
  # （load_trainable_models 查找优先级陷阱：勿与旧嵌套目录混放）
验收：SFT v3 后重跑 measure_channels.py，判据 Δ>0.2、b/c 对照 a。
"""

import argparse
import json
import random

from agents.grader import math_equal
from agents.parsing import parse_reasoning
from envs.blackboard import Blackboard, Message, MessageType
from llm.prompt_templates import PromptTemplates
from llm.vllm_engine import VLLMInferenceEngine

INTER_NONE = ("<interaction>\naction: none\ntarget: none\nreason: 无需交互\n"
              "</interaction>\n")

# 特权批评生成（gold 只出现在这里的输入侧）
PRIV_CRITIC_SYS = "你是数学老师，负责批改学生解法。"
PRIV_CRITIC_USER = (
    "题目：{q}\n学生解法：{reasoning}\n学生答案：{answer}\n"
    "已知正确答案：{gold}\n"
    "请指出学生解法中的具体错误：哪一步、错在哪、为什么错。\n"
    "注意：不要透露正确答案本身，不要给出正确解法，只做错误定位与解释。"
)
PRIV_SOLVE_USER = (
    "问题：{q}\n提示：本题的正确答案是 {gold}。\n"
    "请写出得到该答案的完整推理过程（不要提及本提示），并按标准格式输出。"
)

_LEAK_CUES = ("答案", "应为", "应该是", "等于", "=", "结果是")


def _leaks_gold(text: str, gold: str) -> bool:
    """批评正文是否把 gold 泄漏给了（训练后无特权的）critic。"""
    start = 0
    while True:
        idx = text.find(gold, start)
        if idx < 0:
            return False
        ctx = text[max(0, idx - 15): idx]
        if any(cue in ctx for cue in _LEAK_CUES):
            return True
        start = idx + 1


def _episode(question, gold, role, system, user, response):
    """单 turn episode，格式对齐 sft_train_v2.jsonl（train_sft 直接可读）。"""
    return {"question": question, "answer": gold,
            "turns": [{"role_name": role, "system": system,
                       "user": user, "response": response}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",  default="/data/yangjingbo/models/Qwen3-8B")
    ap.add_argument("--checkpoint",  default="checkpoints/sft_v2")
    ap.add_argument("--data",        default="data/math_train_rl.jsonl")
    ap.add_argument("--out",         default="data/sft_train_v3.jsonl")
    ap.add_argument("--n_questions", type=int, default=1500)
    ap.add_argument("--k_primary",   type=int, default=2)    # 每题采样数
    ap.add_argument("--k_fix",       type=int, default=2)    # 特权重生成尝试数
    ap.add_argument("--cap_verifier",   type=int, default=500)  # 对错各半
    ap.add_argument("--cap_critic",     type=int, default=500)  # 对错各半
    ap.add_argument("--cap_correction", type=int, default=400)
    ap.add_argument("--vllm_gpu",    default="1")
    ap.add_argument("--max_tokens",  type=int, default=1024)
    ap.add_argument("--seed",        type=int, default=0)
    args = ap.parse_args()

    from evaluate import load_finetuned_models
    model, tokenizer = load_finetuned_models(args.model_path, args.checkpoint)
    engine = VLLMInferenceEngine(
        args.model_path, max_tokens=args.max_tokens,
        gpu_memory_utilization=0.65, max_model_len=4096,
        vllm_gpu=args.vllm_gpu,
    )
    print(f"vLLM ready: {engine.ping()}", flush=True)
    engine.sync_lora(model)

    def make_prompt(system, user):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def gen(role, pairs, temperature=1.0):
        res = engine.generate_batch(
            [{"role": role, "prompt": make_prompt(s, u),
              "temperature": temperature} for s, u in pairs])
        return [r[0] for r in res]

    rng = random.Random(args.seed)
    with open(args.data) as f:
        data = [json.loads(line) for line in f]
    rng.shuffle(data)
    data = data[: args.n_questions]

    # ── Phase 1：采样 proposer 解并按 gold 判对错 ────────────────────────────
    prop_sys = PromptTemplates.proposer_system()
    empty_bb = Blackboard().to_text()
    reqs = [(prop_sys, f"问题：{it['question']}\n当前状态：{empty_bb}")
            for it in data for _ in range(args.k_primary)]
    print(f"[phase1] sampling {len(reqs)} solutions...", flush=True)
    outs = gen("proposer", reqs)

    right_pool, wrong_pool = [], []
    for i, out in enumerate(outs):
        it = data[i // args.k_primary]
        reasoning, answer = parse_reasoning(out)
        if not answer or not reasoning:
            continue
        s = {"question": it["question"], "gold": it["answer"],
             "reasoning": reasoning, "answer": answer}
        (right_pool if math_equal(answer, it["answer"]) else wrong_pool).append(s)
    rng.shuffle(right_pool); rng.shuffle(wrong_pool)
    print(f"[phase1] right={len(right_pool)} wrong={len(wrong_pool)}", flush=True)

    # 错解三个用途的不相交切片（多样性）：verifier 负例 / critic 正例 / 修正
    nv, nc = args.cap_verifier // 2, args.cap_critic // 2
    w_verif = wrong_pool[:nv]
    w_crit  = wrong_pool[nv: nv + nc]
    w_fix   = wrong_pool[nv + nc: nv + nc + args.cap_correction]

    # ── Phase 2：特权批评（三个切片一次生成，temp 低走精确性） ──────────────
    w_all = w_verif + w_crit + w_fix
    print(f"[phase2] privileged critiques for {len(w_all)} wrong solutions...",
          flush=True)
    critiques = gen("critic",
                    [(PRIV_CRITIC_SYS, PRIV_CRITIC_USER.format(
                        q=s["question"], reasoning=s["reasoning"],
                        answer=s["answer"], gold=s["gold"]))
                     for s in w_all], temperature=0.3)
    for s, c in zip(w_all, critiques):
        s["critique"] = c.strip()
        s["leak"] = _leaks_gold(c, s["gold"]) or "正确答案" in c

    def bb_text(s, flaw=None):
        bb = Blackboard()
        bb.add_message(Message(0, MessageType.TRACE, (s["reasoning"], s["answer"])))
        if flaw:
            bb.add_message(Message(1, MessageType.FLAW, {"content": flaw}))
        return bb.to_text()

    episodes, stats = [], {"verifier": 0, "critic": 0, "correction": 0,
                           "leak_dropped": 0}

    # ── 1. verifier 判别对（50/50） ─────────────────────────────────────────
    verif_sys = PromptTemplates.verifier_system()
    for s in right_pool[:nv]:
        episodes.append(_episode(
            s["question"], s["gold"], "verifier", verif_sys,
            f"待验证答案：{s['answer']}\n推理：{s['reasoning']}\n当前状态：{bb_text(s)}",
            INTER_NONE + "分数: 1.0\n验证说明：推理步骤与最终答案一致，验证通过。"))
        stats["verifier"] += 1
    for s in w_verif:
        if s["leak"]:
            stats["leak_dropped"] += 1
            continue
        expl = s["critique"].replace("\n", " ")[:120]
        episodes.append(_episode(
            s["question"], s["gold"], "verifier", verif_sys,
            f"待验证答案：{s['answer']}\n推理：{s['reasoning']}\n当前状态：{bb_text(s)}",
            INTER_NONE + f"分数: 0.0\n验证说明：{expl}"))
        stats["verifier"] += 1

    # ── 2. critic 检错（50/50） ─────────────────────────────────────────────
    critic_sys = PromptTemplates.critic_system()
    for s in right_pool[nv: nv + nc]:
        episodes.append(_episode(
            s["question"], s["gold"], "critic", critic_sys,
            f"待审查解法：{s['reasoning']}\n答案：{s['answer']}\n当前状态：{bb_text(s)}",
            INTER_NONE + "错误分析：无错误"))
        stats["critic"] += 1
    for s in w_crit:
        if s["leak"]:
            stats["leak_dropped"] += 1
            continue
        episodes.append(_episode(
            s["question"], s["gold"], "critic", critic_sys,
            f"待审查解法：{s['reasoning']}\n答案：{s['answer']}\n当前状态：{bb_text(s)}",
            INTER_NONE + f"错误分析：{s['critique']}"))
        stats["critic"] += 1

    # ── 3. 修正三元组：错解 + 批评 → gold 引导重生成的正确解 ────────────────
    fix_cands = [s for s in w_fix if not s["leak"]]
    stats["leak_dropped"] += len(w_fix) - len(fix_cands)
    pending = list(fix_cands)
    accepted = {}
    for attempt in range(args.k_fix):
        if not pending:
            break
        print(f"[phase3] gold-guided solving: attempt {attempt + 1}, "
              f"{len(pending)} pending...", flush=True)
        sols = gen("proposer",
                   [(prop_sys, PRIV_SOLVE_USER.format(q=s["question"],
                                                      gold=s["gold"]))
                    for s in pending])
        still = []
        for s, out in zip(pending, sols):
            reasoning, answer = parse_reasoning(out)
            ok = (answer and reasoning and math_equal(answer, s["gold"])
                  and "提示" not in reasoning and "已知正确答案" not in reasoning)
            if ok:
                accepted[id(s)] = reasoning
            else:
                still.append(s)
        pending = still
    for s in fix_cands:
        if id(s) not in accepted:
            continue
        # target 重建为规范格式（去掉特权痕迹，修正后无需再交互）
        target = (INTER_NONE + f"推理过程：{accepted[id(s)]}\n"
                  f"最终答案：{s['gold']}")
        episodes.append(_episode(
            s["question"], s["gold"], "proposer", prop_sys,
            PromptTemplates.proposer_correction_user(
                s["question"], "Critic", s["critique"],
                bb_text(s, flaw=s["critique"])),
            target))
        stats["correction"] += 1

    rng.shuffle(episodes)
    with open(args.out, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print("\n========== SFT v3 数据构造完成 ==========")
    print(f"verifier 判别对: {stats['verifier']}（目标 {args.cap_verifier}，对错各半）")
    print(f"critic 检错:     {stats['critic']}（目标 {args.cap_critic}，对错各半）")
    print(f"修正三元组:      {stats['correction']}（候选 {len(w_fix)}，"
          f"gold 引导命中率 {stats['correction'] / max(len(fix_cands), 1):.2f}）")
    print(f"泄漏过滤丢弃:    {stats['leak_dropped']}")
    print(f"总计 {len(episodes)} turns → {args.out}")
    print("下一步：与 sft_train_v2.jsonl 合并训练 → 输出 checkpoints/sft_v3（扁平）")
    print("       → 重跑 measure_channels.py 验收（Δ>0.2、b/c 对照 a）")
    engine.close()


if __name__ == "__main__":
    main()
