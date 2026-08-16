"""RACA v2 执行器：controller 仅终止决策，交互是唯一路由机制。

流程（§2.1）：每轮
  1. σ 机械推导 → 2. Controller continue/stop（stop 闸门：需黑板存在 verifier 分数）
  3. Proposer 固定起点：解题 + 交互决策 {none|request|challenge}×{critic|verifier}
  4. 交互链（≤ max_hops 跳响应）：响应方以标准角色格式回应、追加写入（不覆盖）；
     critic 可 request proposer 修正，修正输出作为新 trace 进入多数投票
  5. ε 强制注入：proposer 未发起时以 ε_t 概率强制一次 critic 审查（冷启动保护）

奖励计算委托给 agents/raca_rewards.compute_turn_data（纯 Python，可单测）。
"""

import random
from collections import Counter

from agents.grader import math_equal   # re-export：evaluate.py 从此模块导入
from agents.parsing import (
    ROLE_NAMES,
    critic_found_errors,
    parse_decision,
    parse_interaction,
    parse_reasoning,
    parse_score,
)
from agents.raca_rewards import compute_turn_data
from envs.blackboard import Blackboard, Message, MessageType
from llm.prompt_templates import PromptTemplates


class AgenticExecutor:
    def __init__(self, model, tokenizer, config, vllm_engine=None, eval_mode: bool = False):
        self.model = model              # v2 rollout 不再直接用训练模型前向，仅保留引用
        self.tokenizer = tokenizer
        self.cfg = dict(config)
        self.max_tokens = config.get("max_tokens", 512)
        self.max_rounds = config.get("max_rounds", 4)
        self.max_hops = config.get("max_hops", 2)        # 0 = 消融A：禁用交互
        self.stop_gate = config.get("stop_gate", True)   # stop 需存在 verifier 分数
        self.vllm_engine = vllm_engine
        # eval_mode：greedy 解码 + 不做 ε 强制注入（评测学到的策略本身）
        self.eval_mode = eval_mode
        self.temperature = 0.0 if eval_mode else 1.0
        self._rng = random.Random()

    # ── batch entry point（N episodes 并行，逐 turn-slot 批量调 vLLM） ───────
    def run_episodes_batch(self, questions: list, correct_answers: list,
                           eps_force: float = 0.0) -> list:
        """运行 N 个 episode。eps_force：本批次强制注入 critic 的概率（训练时
        由 train.py 按线性衰减 schedule 传入；eval 默认 0）。"""
        if self.vllm_engine is None:
            raise RuntimeError("RACA v2 rollout 需要 vLLM 引擎（no_vllm 路径已移除）")
        if self.eval_mode:
            eps_force = 0.0

        n = len(questions)
        blackboards    = [Blackboard() for _ in range(n)]
        all_messages   = [[] for _ in range(n)]
        turn_ids_list  = [[] for _ in range(n)]
        log_probs_list = [[] for _ in range(n)]
        turn_counters  = [0] * n
        round_records  = [[] for _ in range(n)]
        stop_ctrl_tids = [None] * n
        stop_sigmas    = [None] * n
        active         = list(range(n))

        def next_tid(i):
            tid = turn_counters[i]
            turn_counters[i] += 1
            return tid

        def make_prompt(system, user):
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )

        def record(i, role, system, user, result, tid):
            text, turn_lps, token_ids = result
            n_resp = len(token_ids)
            aligned = list(turn_lps) + [0.0] * (n_resp - len(turn_lps))
            log_probs_list[i].extend(aligned[:n_resp])
            turn_ids_list[i].extend([tid] * n_resp)
            all_messages[i].append({
                "role_name":    role,
                "turn_id":      tid,
                "system":       system,
                "user":         user,
                "response":     text,
                "response_ids": token_ids,
            })
            return text

        def responder_prompt(i, target, initiator, action, reason, init_out, forced):
            """响应方 prompt：标准角色格式 + 请求上下文（forced 无请求上下文）。"""
            bb = blackboards[i]
            last = bb.traces[-1] if bb.traces else ("", "")
            if target == "critic":
                sys = PromptTemplates.critic_system()
                usr = f"待审查解法：{last[0]}\n答案：{last[1]}\n当前状态：{bb.to_text()}"
            elif target == "verifier":
                sys = PromptTemplates.verifier_system()
                usr = f"待验证答案：{last[1]}\n推理：{last[0]}\n当前状态：{bb.to_text()}"
            else:  # proposer 修正
                sys = PromptTemplates.proposer_system()
                usr = PromptTemplates.proposer_correction_user(
                    questions[i], ROLE_NAMES.get(initiator, initiator),
                    init_out, bb.to_text())
            if not forced and target != "proposer":
                usr += PromptTemplates.request_context(
                    ROLE_NAMES.get(initiator, initiator), action, reason, init_out)
            return sys, usr, (last[1] if bb.traces else "")

        for _ in range(self.max_rounds):
            if not active:
                break

            # ── σ 机械推导（进入本轮前的黑板状态） ────────────────────────
            sigma_map = {i: blackboards[i].derive_sigma() for i in active}

            # ── 1. controller（batch）：continue / stop ────────────────────
            ctrl_info = []
            for i in active:
                sys = PromptTemplates.controller_system()
                usr = f"问题：{questions[i]}\n当前状态：{blackboards[i].to_text()}"
                ctrl_info.append((i, next_tid(i), sys, usr))

            ctrl_res = self.vllm_engine.generate_batch(
                [{"role": "controller", "prompt": make_prompt(s, u),
                  "temperature": self.temperature} for _, _, s, u in ctrl_info]
            )

            still_active = []
            ep_st = {}   # 本轮活跃 episode 的 round 状态
            for (i, tid, sys, usr), res in zip(ctrl_info, ctrl_res):
                text = record(i, "controller", sys, usr, res, tid)
                gate_blocked = False
                if parse_decision(text) == "stop":
                    # stop 闸门：黑板存在 verifier 分数才允许终止（§2.1 约束）
                    if (not self.stop_gate) or blackboards[i].scores:
                        stop_ctrl_tids[i] = tid
                        stop_sigmas[i] = sigma_map[i]
                        continue
                    gate_blocked = True   # 闸门拦截 → 强制 continue
                still_active.append(i)
                ep_st[i] = {
                    "sigma":            sigma_map[i],
                    "ctrl_tid":         tid,
                    "gate_blocked":     gate_blocked,
                    "critic_turns":     [],
                    "verifier_turns":   [],
                    "correction_turns": [],
                    "corrected_answer": None,
                    "pending":          [],
                }
            active = still_active
            if not ep_st:
                continue

            # ── 2. proposer 固定起点（batch） ──────────────────────────────
            prop_info = []
            for i in active:
                sys = PromptTemplates.proposer_system()
                usr = f"问题：{questions[i]}\n当前状态：{blackboards[i].to_text()}"
                prop_info.append((i, next_tid(i), sys, usr))

            prop_res = self.vllm_engine.generate_batch(
                [{"role": "proposer", "prompt": make_prompt(s, u),
                  "temperature": self.temperature} for _, _, s, u in prop_info]
            )

            for (i, tid, sys, usr), res in zip(prop_info, prop_res):
                out = record(i, "proposer", sys, usr, res, tid)
                reasoning, answer = parse_reasoning(out)
                blackboards[i].add_message(Message(0, MessageType.TRACE, (reasoning, answer)))

                st = ep_st[i]
                st["primary_tid"] = tid
                st["primary_answer"] = answer
                # 格式健康：是否输出了「最终答案：」字段（而非靠抽末尾数字兜底）。
                # 格式崩了 reward 再高也是假的，所以进入监控指标。
                st["primary_parsed"] = "最终答案：" in out and bool(answer)

                action, target, reason = parse_interaction(out)
                u = action != "none" and self.max_hops > 0
                forced = False
                if u:
                    blackboards[i].add_message(Message(
                        0, MessageType.INTERACTION,
                        {"from": "proposer", "action": action,
                         "target": target, "reason": reason}))
                elif self.max_hops > 0 and self._rng.random() < eps_force:
                    # ε 强制注入 critic 审查（冷启动保护，§2.4）
                    forced, action, target, reason = True, "request", "critic", ""
                st["u"] = u
                st["forced"] = forced
                st["target"] = target if u else None
                if u or forced:
                    st["pending"].append(("proposer", out, action, target, reason, forced))

            # ── 3. 交互链（≤ max_hops 跳，跨 episode 按深度批量） ──────────
            for _hop in range(self.max_hops):
                batch_req = []
                for i, st in ep_st.items():
                    if not st["pending"]:
                        continue
                    initiator, init_out, action, target, reason, forced = st["pending"].pop(0)
                    if target == initiator or target not in ROLE_NAMES:
                        continue
                    sys, usr, reviewed = responder_prompt(
                        i, target, initiator, action, reason, init_out, forced)
                    batch_req.append((i, next_tid(i), target, sys, usr, reviewed))
                if not batch_req:
                    break

                res_all = self.vllm_engine.generate_batch(
                    [{"role": t, "prompt": make_prompt(s, u),
                      "temperature": self.temperature} for _, _, t, s, u, _ in batch_req]
                )
                for (i, tid, target, sys, usr, reviewed), res in zip(batch_req, res_all):
                    out = record(i, target, sys, usr, res, tid)
                    st = ep_st[i]
                    bb = blackboards[i]

                    if target == "critic":
                        flagged = critic_found_errors(out)
                        if flagged:
                            bb.add_message(Message(1, MessageType.FLAW, {"content": out}))
                        st["critic_turns"].append({
                            "tid": tid, "flagged": flagged,
                            "reviewed_answer": reviewed,
                            "correction_followed": False,
                        })
                    elif target == "verifier":
                        score = parse_score(out)
                        bb.add_message(Message(
                            2, MessageType.SCORE,
                            (reviewed, score if score is not None else 0.5)))
                        st["verifier_turns"].append({
                            "tid": tid, "score": score, "reviewed_answer": reviewed,
                        })
                    else:  # proposer 修正：新 trace 进入多数投票（因果通路）
                        reasoning, answer = parse_reasoning(out)
                        bb.add_message(Message(0, MessageType.TRACE, (reasoning, answer)))
                        st["correction_turns"].append({"tid": tid, "answer": answer})
                        st["corrected_answer"] = answer
                        # 本轮内此前的 critic flag 得到了修正响应 → 因果窗口在本轮
                        for ct in st["critic_turns"]:
                            ct["correction_followed"] = True

                    # 响应方可再发起下一跳（受 hop 预算约束）
                    a2, t2, r2 = parse_interaction(out)
                    if a2 != "none" and t2 != target:
                        bb.add_message(Message(
                            list(ROLE_NAMES).index(target), MessageType.INTERACTION,
                            {"from": target, "action": a2, "target": t2, "reason": r2}))
                        st["pending"].append((target, out, a2, t2, r2, False))

            # ── 4. round record 落盘 ───────────────────────────────────────
            for i, st in ep_st.items():
                round_records[i].append({
                    "sigma":            st["sigma"],
                    "ctrl_tid":         st["ctrl_tid"],
                    "gate_blocked":     st["gate_blocked"],
                    "primary_tid":      st["primary_tid"],
                    "primary_answer":   st["primary_answer"],
                    "primary_parsed":   st["primary_parsed"],
                    "corrected_answer": st["corrected_answer"],
                    "u":                st["u"],
                    "forced":           st["forced"],
                    "target":           st["target"],
                    "critic_turns":     st["critic_turns"],
                    "verifier_turns":   st["verifier_turns"],
                    "correction_turns": st["correction_turns"],
                })

        # ── finalise ─────────────────────────────────────────────────────────
        results = []
        for i in range(n):
            final_answer = self._majority_vote(blackboards[i])
            is_correct = math_equal(final_answer, correct_answers[i])
            turn_data, round_meta = compute_turn_data(
                round_records[i], correct_answers[i], is_correct,
                self.max_rounds, self.cfg,
                stop_ctrl_tid=stop_ctrl_tids[i],
                stop_sigma=stop_sigmas[i] or "verify",
            )
            results.append({
                "messages":        all_messages[i],
                "turn_ids":        turn_ids_list[i],
                "log_probs":       log_probs_list[i],
                "raca_turn_data":  turn_data,
                "raca_round_meta": round_meta,
                "final_answer":    final_answer,
                "is_correct":      is_correct,
                "stopped":         stop_ctrl_tids[i] is not None,
            })
        return results

    # ── 单条入口（batch 路径的薄包装，评测脚本兼容） ─────────────────────────
    def run_episode(self, question: str, correct_answer: str) -> dict:
        return self.run_episodes_batch([question], [correct_answer])[0]

    # ── evaluate.py 兼容 ─────────────────────────────────────────────────────
    def _critic_found_errors(self, critic_output: str) -> bool:
        return critic_found_errors(critic_output)

    def _majority_vote(self, blackboard: Blackboard) -> str:
        if not blackboard.traces:
            return ""
        return Counter(ans for _, ans in blackboard.traces).most_common(1)[0][0]
