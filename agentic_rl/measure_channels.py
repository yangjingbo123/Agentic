"""RACA v2.3 §18 第 0 步：三通道离线测量（决定交互该怎么给 acc 做贡献）。

在 primary 答错的样本上对比三条错误恢复路径的正确率：
  a) resample    裸重采样（基线——免费竞争者，交互必须超过它才有存在价值）
  b) correction  critic 审查 → proposer 修正（通道③：修正票质量）
  c) next_round  黑板带 FLAW 的下一轮 primary（通道②：批评条件化重答）
并测 verifier 校准（通道①加权投票的前提）：
  d) 对/错答案的平均分数差 + 校准误差 mean|score − 正确性|

判读：
  - b ≤ a 且 c ≤ a：critic 通道不产生超额价值 → 修正票不该进投票 /
    critic 反馈质量要回 SFT 修；交互对 acc 的贡献只能靠通道①（加权投票）
  - d 的分数差 ≤ 0：verifier 无分辨力 → 加权投票也不成立，先修 verifier SFT

prompt 构造与 agentic_executor 逐字节一致（复用 PromptTemplates/Blackboard），
保证测出来的就是训练环境里的那条链。

用法（训练机）：
  python measure_channels.py --checkpoint checkpoints/sft_v2 \
      --model_path /data/yangjingbo/models/Qwen3-8B --vllm_gpu 1 --n 300
"""

import argparse
import json
import random

from agents.grader import math_equal
from agents.parsing import ROLE_NAMES, critic_found_errors, parse_reasoning, parse_score
from envs.blackboard import Blackboard, Message, MessageType
from llm.prompt_templates import PromptTemplates
from llm.vllm_engine import VLLMInferenceEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/yangjingbo/models/Qwen3-8B")
    ap.add_argument("--checkpoint", default="checkpoints/sft_v2")
    ap.add_argument("--data",       default="data/math_train_rl.jsonl")
    ap.add_argument("--n",          type=int, default=300)
    ap.add_argument("--max_wrong",  type=int, default=200)
    ap.add_argument("--vllm_gpu",   default="1")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--rpc_timeout_s", type=int, default=1800)
    ap.add_argument("--gen_chunk",     type=int, default=256)   # 单次 RPC 请求数
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    from evaluate import load_finetuned_models
    model, tokenizer = load_finetuned_models(args.model_path, args.checkpoint)

    engine = VLLMInferenceEngine(
        args.model_path, max_tokens=args.max_tokens,
        gpu_memory_utilization=0.65, max_model_len=4096,
        vllm_gpu=args.vllm_gpu, rpc_timeout_s=args.rpc_timeout_s,
    )
    print(f"vLLM ready: {engine.ping()}", flush=True)
    engine.sync_lora(model)
    print("LoRA synced.", flush=True)

    def make_prompt(system, user):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def gen(role, pairs, temperature=1.0):
        """pairs: [(system, user)] → [text]，按角色挂对应 LoRA。
        分块发送，避免单个巨型 RPC 撞 rpc_timeout。"""
        outs = []
        for i in range(0, len(pairs), args.gen_chunk):
            part = pairs[i: i + args.gen_chunk]
            res = engine.generate_batch(
                [{"role": role, "prompt": make_prompt(s, u),
                  "temperature": temperature} for s, u in part])
            outs.extend(r[0] for r in res)
            if len(pairs) > args.gen_chunk:
                print(f"  [gen:{role}] {min(i + args.gen_chunk, len(pairs))}"
                      f"/{len(pairs)}", flush=True)
        return outs

    rng = random.Random(args.seed)
    with open(args.data) as f:
        data = [json.loads(line) for line in f]
    rng.shuffle(data)
    data = data[: args.n]
    print(f"Loaded {len(data)} items from {args.data}", flush=True)

    # ── Phase 1：primary 生成（与 executor 第一轮完全一致，黑板为空） ────────
    prop_sys = PromptTemplates.proposer_system()
    empty_bb = Blackboard().to_text()
    primary_out = gen("proposer",
                      [(prop_sys, f"问题：{it['question']}\n当前状态：{empty_bb}")
                       for it in data])

    samples = []   # {question, gold, reasoning, answer, correct}
    for it, out in zip(data, primary_out):
        reasoning, answer = parse_reasoning(out)
        samples.append({
            "question": it["question"], "gold": it["answer"],
            "reasoning": reasoning, "answer": answer,
            "correct": bool(answer) and math_equal(answer, it["answer"]),
        })
    n_right = sum(s["correct"] for s in samples)
    print(f"[phase1] primary acc = {n_right}/{len(samples)} "
          f"({n_right / len(samples):.3f})", flush=True)

    wrong = [s for s in samples if not s["correct"]][: args.max_wrong]
    print(f"[phase1] wrong subset = {len(wrong)}", flush=True)

    def bb_with_trace(s, flaw=None):
        bb = Blackboard()
        bb.add_message(Message(0, MessageType.TRACE, (s["reasoning"], s["answer"])))
        if flaw is not None:
            bb.add_message(Message(1, MessageType.FLAW, {"content": flaw}))
        return bb

    def acc_of(outs, subset):
        hits = 0
        for s, out in zip(subset, outs):
            _, ans = parse_reasoning(out)
            hits += bool(ans) and math_equal(ans, s["gold"])
        return hits / max(len(subset), 1)

    # ── a) resample：同一 primary prompt 重采样（基线） ─────────────────────
    acc_a = acc_of(gen("proposer",
                       [(prop_sys, f"问题：{s['question']}\n当前状态：{empty_bb}")
                        for s in wrong]), wrong)

    # ── b) correction：critic 审查 → proposer 修正（与 executor 修正链一致）─
    critic_out = gen("critic",
                     [(PromptTemplates.critic_system(),
                       f"待审查解法：{s['reasoning']}\n答案：{s['answer']}\n"
                       f"当前状态：{bb_with_trace(s).to_text()}")
                      for s in wrong])
    flag_rate = sum(critic_found_errors(o) for o in critic_out) / max(len(wrong), 1)
    corr_pairs = []
    for s, co in zip(wrong, critic_out):
        bb = bb_with_trace(s, flaw=co if critic_found_errors(co) else None)
        corr_pairs.append((prop_sys, PromptTemplates.proposer_correction_user(
            s["question"], ROLE_NAMES.get("critic", "critic"), co, bb.to_text())))
    acc_b = acc_of(gen("proposer", corr_pairs), wrong)

    # ── c) next_round：黑板带 FLAW 的下一轮 primary ─────────────────────────
    acc_c = acc_of(gen("proposer",
                       [(prop_sys,
                         f"问题：{s['question']}\n"
                         f"当前状态：{bb_with_trace(s, flaw=co).to_text()}")
                        for s, co in zip(wrong, critic_out)]), wrong)

    # ── d) verifier 校准（全部样本，含答对的） ──────────────────────────────
    verif_out = gen("verifier",
                    [(PromptTemplates.verifier_system(),
                      f"待验证答案：{s['answer']}\n推理：{s['reasoning']}\n"
                      f"当前状态：{bb_with_trace(s).to_text()}")
                     for s in samples])
    sc_right, sc_wrong, cal_err, n_parsed = [], [], [], 0
    for s, vo in zip(samples, verif_out):
        score = parse_score(vo)
        if score is None:
            continue
        n_parsed += 1
        (sc_right if s["correct"] else sc_wrong).append(score)
        cal_err.append(abs(score - float(s["correct"])))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

    # ── 报告 ────────────────────────────────────────────────────────────────
    print("\n========== 三通道测量结果 ==========")
    print(f"样本：n={len(samples)}  primary_acc={n_right / len(samples):.3f}  "
          f"wrong={len(wrong)}")
    print(f"a) resample   （基线）      acc = {acc_a:.3f}")
    print(f"b) correction （通道③）    acc = {acc_b:.3f}  "
          f"Δ={acc_b - acc_a:+.3f}  (critic flag_rate={flag_rate:.2f})")
    print(f"c) next_round （通道②）    acc = {acc_c:.3f}  Δ={acc_c - acc_a:+.3f}")
    print(f"d) verifier 校准（通道①前提，parse {n_parsed}/{len(samples)}）")
    print(f"   score|对 = {mean(sc_right):.3f}   score|错 = {mean(sc_wrong):.3f}   "
          f"分辨力Δ = {mean(sc_right) - mean(sc_wrong):+.3f}   "
          f"校准误差 = {mean(cal_err):.3f}")
    print("====================================")
    print("判读：b/c 显著 > a → critic 通道有超额价值，修正票留在投票池；")
    print("      b/c ≤ a → 修正票退出投票（只做 r_int 因果信用），critic 回 SFT；")
    print("      分辨力Δ ≤ 0 → 加权投票不成立，verifier 先回 SFT。")
    engine.close()


if __name__ == "__main__":
    main()
