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

v3.2 新增两个臂（§21，定向回答两个已定位的问题）：
  e) critic 精度 + 替换语义 Δacc
     现状是硬错配：r_int/eff 按 p_end（修正后正确性）发奖，而投票把修正
     票 exclude 掉 —— 奖励在为不进结果的行为付费。替换语义让修正覆盖被
     flag 的那一票，收益符号完全由 critic 精度决定。旧实现只在答错子集上
     跑 critic，因此测不到假阳性、算不出精度。
  f) 全覆盖 vs 稀疏覆盖加权投票
     判决 §20.2 的覆盖率假说：离线全覆盖测得 Δ=+0.312，但线上 d_vote 两轮
     都≈0。全覆盖下若仍≈0，覆盖率解释就可以埋了。

prompt 构造与 agentic_executor 逐字节一致（复用 PromptTemplates/Blackboard，
e/f 臂的计票直接调 AgenticExecutor._majority_vote），
保证测出来的就是训练环境里的那条链。

用法（训练机）：
  python measure_channels.py --checkpoint checkpoints/sft_v3 \
      --model_path /data/yangjingbo/models/Qwen3-8B --vllm_gpu 1 --n 300
只跑旧三通道（不花 e/f 的额外采样）：加 --skip_ef
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
    ap.add_argument("--vllm_use_v1",   default="0", choices=["0", "1", "auto"],
                    help="0=V0(默认) 1=V1(vLLM≥0.10 必选) auto=交给 vLLM")
    ap.add_argument("--seed",       type=int, default=0)
    # ── v3.2 e/f 臂 ──
    ap.add_argument("--skip_ef", action="store_true",
                    help="只跑旧 a/b/c/d 四臂，跳过 e（critic 精度）/f（全覆盖投票）")
    ap.add_argument("--vote_k", type=int, default=4,
                    help="f 臂每题的投票池大小（第 1 票复用 phase1，故新增 k-1 次采样）")
    ap.add_argument("--sparse_cov", type=float, default=0.32,
                    help="f 臂稀疏对照的覆盖率，默认 0.32 = 训练时实测值")
    # ── v3.3：采样条件对齐。e 臂的收益符号取决于基础错误率 p——
    # precision = TPR·p / (TPR·p + FPR·(1−p))，p 一降 precision 就塌。
    # 训练 rollout 是 temperature=1.0 + math_train_rl 全难度（p≈0.54），
    # 而上报的 eval 是 greedy + math_test Level-5（p≈0.15），两者不是同一个
    # 结论。要在哪个条件下用这个通道，就得在哪个条件下测。
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="采样温度。1.0=训练 rollout 条件；0.0=eval 条件（executor "
                         "eval_mode 走 greedy）")
    ap.add_argument("--level", default="",
                    help="只保留该难度，如 'Level 5'。对齐 train.py 的 eval 子集")
    ap.add_argument("--no_shuffle", action="store_true",
                    help="不打乱、取前 n 条。配合 --level 可复现 train.py 的固定 eval 子集")
    args = ap.parse_args()

    from evaluate import load_finetuned_models
    model, tokenizer = load_finetuned_models(args.model_path, args.checkpoint)

    engine = VLLMInferenceEngine(
        args.model_path, max_tokens=args.max_tokens,
        gpu_memory_utilization=0.65, max_model_len=4096,
        vllm_gpu=args.vllm_gpu, rpc_timeout_s=args.rpc_timeout_s,
        vllm_use_v1=args.vllm_use_v1,
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

    def gen(role, pairs, temperature=None):
        """pairs: [(system, user)] → [text]，按角色挂对应 LoRA。
        分块发送，避免单个巨型 RPC 撞 rpc_timeout。"""
        temperature = args.temperature if temperature is None else temperature
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
    if args.level:
        data = [it for it in data if it.get("level") == args.level]
    if not args.no_shuffle:
        rng.shuffle(data)
    data = data[: args.n]
    print(f"Loaded {len(data)} items from {args.data}"
          f"{f' [{args.level}]' if args.level else ''}"
          f"  temperature={args.temperature}", flush=True)

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

    wrong_idx = [j for j, s in enumerate(samples) if not s["correct"]][: args.max_wrong]
    wrong = [samples[j] for j in wrong_idx]
    print(f"[phase1] wrong subset = {len(wrong)}", flush=True)

    def bb_with_trace(s, flaw=None):
        bb = Blackboard()
        bb.add_message(Message(0, MessageType.TRACE, (s["reasoning"], s["answer"])))
        if flaw is not None:
            bb.add_message(Message(1, MessageType.FLAW, {"content": flaw}))
        return bb

    def hits_of(outs, subset):
        """逐样本正确性。e 臂要按真阳/假阳分组算，拿不到均值就够。"""
        hits = []
        for s, out in zip(subset, outs):
            _, ans = parse_reasoning(out)
            hits.append(bool(ans) and math_equal(ans, s["gold"]))
        return hits

    def acc_of(outs, subset):
        h = hits_of(outs, subset)
        return sum(h) / max(len(h), 1)

    # ── a) resample：同一 primary prompt 重采样（基线） ─────────────────────
    acc_a = acc_of(gen("proposer",
                       [(prop_sys, f"问题：{s['question']}\n当前状态：{empty_bb}")
                        for s in wrong]), wrong)

    # ── critic 审查：跑全量样本（含答对的） ─────────────────────────────
    # v3.2：旧实现只审查 wrong 子集，于是只能看到真阳/漏检，测不到假阳性
    # → 算不出 precision。而替换语义的收益符号完全由 precision 决定（见 e 臂），
    # 所以答对样本上的 critic 输出是必需数据，不是额外开销。
    # b/c 臂从中切片，prompt 只依赖单样本，因此与旧实现结果等价。
    critic_all = gen("critic",
                     [(PromptTemplates.critic_system(),
                       f"待审查解法：{s['reasoning']}\n答案：{s['answer']}\n"
                       f"当前状态：{bb_with_trace(s).to_text()}")
                      for s in samples])
    flagged_all = [critic_found_errors(o) for o in critic_all]

    # ── b) correction：critic 审查 → proposer 修正（与 executor 修正链一致）─
    critic_out = [critic_all[j] for j in wrong_idx]
    flag_rate = sum(flagged_all[j] for j in wrong_idx) / max(len(wrong_idx), 1)
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

    # ── e) critic 精度 + 替换语义 Δacc（v3.2） ──────────────────────────
    # 现状：critic flag → 机械触发修正 → 修正票被 exclude 出投票池，而
    # r_int 却按 p_end（修正后正确性）发奖。替换语义把「修正覆盖被 flag
    # 的那一票」，使奖励与结果对齐。单票期望正确性的变化：
    #     Δ/样本 = [n_TP·b_TP − n_FP·(1−b_FP)] / n
    # 真阳换掉一张必错票（净赚 b_TP），假阳换掉一张对票（净亏 1−b_FP）。
    # 注意这测的是被替换那张票的期望正确性变化，不是 episode 级多数投票的
    # 变化——多数投票会稀释单票翻转，所以幅度是上界，但符号一致。
    e = {}
    if not args.skip_ef:
        idx_flag = [j for j, fl in enumerate(flagged_all) if fl]
        tp_idx = [j for j in idx_flag if not samples[j]["correct"]]
        fp_idx = [j for j in idx_flag if samples[j]["correct"]]
        n_tp, n_fp = len(tp_idx), len(fp_idx)
        n_fn = sum(1 for j, fl in enumerate(flagged_all)
                   if not fl and not samples[j]["correct"])
        n_tn = len(samples) - n_tp - n_fp - n_fn

        # 被 flag 样本上的修正（prompt 与 executor 修正链一致：critic 标了就修）
        sub_hit = {}
        if idx_flag:
            flag_subset = [samples[j] for j in idx_flag]
            sub_out = gen("proposer", [
                (prop_sys, PromptTemplates.proposer_correction_user(
                    samples[j]["question"], ROLE_NAMES.get("critic", "critic"),
                    critic_all[j],
                    bb_with_trace(samples[j], flaw=critic_all[j]).to_text()))
                for j in idx_flag])
            sub_hit = dict(zip(idx_flag, hits_of(sub_out, flag_subset)))

        b_tp = (sum(sub_hit[j] for j in tp_idx) / n_tp) if n_tp else float("nan")
        b_fp = (sum(sub_hit[j] for j in fp_idx) / n_fp) if n_fp else float("nan")
        gain = sum(sub_hit[j] for j in tp_idx)            # 错→对，净 +1 票
        loss = n_fp - sum(sub_hit[j] for j in fp_idx)     # 对→错，净 −1 票
        denom = b_tp + 1.0 - b_fp
        # 敏感度/特异度是检测器自身的性质，不随基础错误率 p 变；precision 会。
        # 所以把盈亏平衡改写成对 p 的条件，才能从一个分布外推到另一个：
        #   p·TPR·b_TP > (1−p)·FPR·(1−b_FP)
        # 解出的 p_break 就是“策略错得多于这个比例时，插手才划得来”。
        # 推推就知道：策略越准，p 越小，这个通道自动越不划算——自限的。
        tpr = n_tp / max(n_tp + n_fn, 1)
        fpr = n_fp / max(n_fp + n_tn, 1)
        num = tpr * b_tp if n_tp else 0.0
        odds = (fpr * (1.0 - b_fp) / num) if num > 0 else float("inf")
        e = {
            "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn, "n_tn": n_tn,
            "prec": n_tp / max(n_tp + n_fp, 1),
            "rec":  n_tp / max(n_tp + n_fn, 1),
            "flag_rate_all": len(idx_flag) / max(len(samples), 1),
            "b_tp": b_tp, "b_fp": b_fp, "gain": gain, "loss": loss,
            "d_sub": (gain - loss) / max(len(samples), 1),
            # 盈亏平衡所需精度（b_tp/b_fp 固定时）：prec·b_tp = (1−prec)·(1−b_fp)
            "prec_break": ((1.0 - b_fp) / denom) if denom > 0 else float("nan"),
            "tpr": tpr, "fpr": fpr,
            "p_now": (n_tp + n_fn) / max(len(samples), 1),
            "p_break": odds / (1.0 + odds) if odds != float("inf") else 1.0,
        }

    # ── f) 全覆盖 vs 稀疏覆盖加权投票（判决 §20.2 覆盖率假说） ───────────
    # 单个 primary 没法测投票，所以这里每题凑 vote_k 个独立解法当投票池。
    # 全覆盖：给每个不同答案都打分；稀疏：用**同一批分数**随机遮蔽到
    # sparse_cov。只变覆盖率、不变分数，因此两者之差就是覆盖率的因果量。
    #   全覆盖 d_vote ≈ 0  → 覆盖率假说被排除，Δ 本身在 on-policy 分布上不成立
    #   全覆盖 > 0 而稀疏 ≈ 0 → 覆盖率确认，verify_all_answers 值得进生产
    # 计票直接调 executor 的 _majority_vote：显式传 mode 时 `mode or self.vote_mode`
    # 短路，不触碰 self，因此可以 self=None 调用，保证与生产同一套语义。
    # 温度 0 时这个臂无意义：k 票用的是同一个空黑板 prompt，greedy 下会采出
    # k 份完全相同的答案，投票池退化成单票，两种计票必然同分。
    arm_f = {}
    if not args.skip_ef and args.temperature == 0.0:
        print("\n[f 臂已跳过] temperature=0 下 k 票同 prompt 会采出同一答案，"
              "投票池退化。加权投票的判决用 temperature=1.0 那次的结果。", flush=True)
    if not args.skip_ef and args.temperature > 0.0:
        from agents.agentic_executor import AgenticExecutor
        vote = lambda bb, m: AgenticExecutor._majority_vote(None, bb, None, m)

        pools = [[(s["reasoning"], s["answer"])] for s in samples]
        for _r in range(max(args.vote_k - 1, 0)):
            outs = gen("proposer",
                       [(prop_sys, f"问题：{s['question']}\n当前状态：{empty_bb}")
                        for s in samples])
            for p, out in zip(pools, outs):
                p.append(parse_reasoning(out))

        def build_bbs():
            bbs = []
            for pool in pools:
                bb = Blackboard()
                for rsn, ans in pool:
                    bb.add_message(Message(0, MessageType.TRACE, (rsn, ans)))
                bbs.append(bb)
            return bbs

        # 全覆盖打分：每个不同答案一次。reasoning 取首个产出它的解法，黑板含
        # 全部 trace——与 executor 调 verifier 时的上下文形态一致。
        base_bbs = build_bbs()
        score_req = []
        for i_s, (pool, bb) in enumerate(zip(pools, base_bbs)):
            first = {}
            for rsn, ans in pool:
                if ans and ans not in first:
                    first[ans] = rsn
            for ans, rsn in first.items():
                score_req.append((i_s, ans, PromptTemplates.verifier_system(),
                                  f"待验证答案：{ans}\n推理：{rsn}\n"
                                  f"当前状态：{bb.to_text()}"))
        vs_out = gen("verifier", [(s, u) for _, _, s, u in score_req])
        scored = []
        for (i_s, ans, _s, _u), vo in zip(score_req, vs_out):
            sc = parse_score(vo)
            scored.append((i_s, ans, sc if sc is not None else 0.5, sc is not None))

        def with_scores(keep):
            bbs = build_bbs()
            r = random.Random(args.seed + 7)
            n_sc = 0
            for i_s, ans, sc, _ok in scored:
                if keep >= 1.0 or r.random() < keep:
                    bbs[i_s].add_message(Message(2, MessageType.SCORE, (ans, sc)))
                    n_sc += 1
            return bbs, n_sc

        def acc_vote(bbs, mode):
            return sum(math_equal(vote(bb, mode), s["gold"])
                       for bb, s in zip(bbs, samples)) / max(len(samples), 1)

        bbs_full, n_full = with_scores(1.0)
        bbs_sp,   n_sp   = with_scores(args.sparse_cov)
        arm_f = {
            "k": args.vote_k,
            "n_distinct": len(scored) / max(len(samples), 1),
            "parse_ok": sum(1 for *_x, ok in scored if ok) / max(len(scored), 1),
            "uni":  acc_vote(bbs_full, "uniform"),     # 等权，与分数无关
            "full": acc_vote(bbs_full, "weighted"),
            "sp":   acc_vote(bbs_sp,   "weighted"),
            "cov_full": n_full / max(len(scored), 1),
            "cov_sp":   n_sp / max(len(scored), 1),
        }

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

    if e:
        print("\n========== e) critic 精度 + 替换语义（v3.2）==========")
        print(f"混淆矩阵（n={len(samples)}，flag_rate={e['flag_rate_all']:.2f}）")
        print(f"              primary错      primary对")
        print(f"  flag        TP={e['n_tp']:<4d}        FP={e['n_fp']:<4d}")
        print(f"  不 flag     FN={e['n_fn']:<4d}        TN={e['n_tn']:<4d}")
        print(f"  precision = {e['prec']:.3f}   recall = {e['rec']:.3f}")
        print(f"被 flag 样本上的修正正确率：b_TP={e['b_tp']:.3f}（真阳）  "
              f"b_FP={e['b_fp']:.3f}（假阳）")
        print(f"替换收益：真阳救回 {e['gain']} 票  假阳弄坏 {e['loss']} 票  "
              f"→ Δ_sub = {e['d_sub']:+.4f} /样本")
        print(f"盈亏平衡所需 precision = {e['prec_break']:.3f}（实测 {e['prec']:.3f}）")
        print(f"检测器本身（不随基础错误率变）：TPR={e['tpr']:.3f}  FPR={e['fpr']:.3f}")
        print(f"→ 只有当策略错误率 p > {e['p_break']:.3f} 时插手才划得来（本次 p={e['p_now']:.3f}）")
        print("⚠ precision 随 p 变：precision = TPR·p/(TPR·p+FPR·(1−p))。本次是训练")
        print("  rollout 条件（temperature=1.0）；上报的 eval 是 greedy + Level-5，p 更")
        print("  小，同一个检测器在那里的 precision 会低很多。要在 eval 上用，先拿")
        print("  --temperature 0 --level 'Level 5' --no_shuffle 重测一次。")
        print("判读：Δ_sub > 0 且 p > p_break → 修正采用替换语义进投票池；")
        print("      Δ_sub ≤ 0 → critic 精度不够，改走另一条：把 r_int 改成按投票的")
        print("      实际变化计分。acc 不会涨，但不再为不进结果的行为付奖励。")
        print("注：Δ_sub 是单票期望变化，是 episode 级 Δacc 的上界（多数投票会稀释）。")

    if arm_f:
        d_full = arm_f["full"] - arm_f["uni"]
        d_sp   = arm_f["sp"]   - arm_f["uni"]
        print("\n========== f) 全覆盖 vs 稀疏覆盖加权投票（v3.2）==========")
        print(f"投票池 k={arm_f['k']}  平均不同答案数={arm_f['n_distinct']:.2f}  "
              f"score 可解析率={arm_f['parse_ok']:.2f}")
        print(f"  uniform                       acc = {arm_f['uni']:.3f}")
        print(f"  weighted 全覆盖 (cov={arm_f['cov_full']:.2f})  "
              f"acc = {arm_f['full']:.3f}   d_vote = {d_full:+.4f}")
        print(f"  weighted 稀疏   (cov={arm_f['cov_sp']:.2f})  "
              f"acc = {arm_f['sp']:.3f}   d_vote = {d_sp:+.4f}")
        print(f"  覆盖率的因果量 = {d_full - d_sp:+.4f}")
        print("判读：全覆盖 d_vote ≈ 0 → 覆盖率假说被排除，§20.2 改写：Δ=+0.312 本身")
        print("      就不能外推到 on-policy 分布（符号对了但使不上力）；")
        print("      全覆盖 > 0 而稀疏 ≈ 0 → 覆盖率确认，上 verify_all_answers。")
    engine.close()


if __name__ == "__main__":
    main()
