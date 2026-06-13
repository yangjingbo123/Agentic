import re
from collections import Counter

import torch
import torch.nn.functional as F

from envs.blackboard import Blackboard, Message, MessageType
from llm.prompt_templates import PromptTemplates


def normalize_answer(s: str) -> str:
    return re.sub(r"[^0-9.\-]", "", s.strip())


ROLE_SYSTEM = {
    "proposer": PromptTemplates.proposer_system,
    "critic":   PromptTemplates.critic_system,
    "verifier": PromptTemplates.verifier_system,
}
ROLE_NAMES = {"proposer": "Proposer", "critic": "Critic", "verifier": "Verifier"}


class AgenticExecutor:
    def __init__(self, model, tokenizer, config, vllm_engine=None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = config.get("max_tokens", 512)
        self.max_interactions = config.get("max_interactions", 3)
        self.max_rounds = config.get("max_rounds", 3)
        self.vllm_engine = vllm_engine

    # ── batch entry point (N episodes in parallel, batched per turn-slot) ───
    def run_episodes_batch(self, questions: list, correct_answers: list) -> list:
        """Run N episodes, batching all vLLM calls within each turn-slot across episodes."""
        n = len(questions)
        blackboards     = [Blackboard() for _ in range(n)]
        all_messages    = [[] for _ in range(n)]
        turn_ids_list   = [[] for _ in range(n)]
        log_probs_list  = [[] for _ in range(n)]
        seq_input_ids_l = [[] for _ in range(n)]
        seq_step_ids_l  = [[] for _ in range(n)]
        turn_counters   = [0] * n
        round_records   = [[] for _ in range(n)]
        active          = list(range(n))

        def next_tid(i):
            tid = turn_counters[i]
            turn_counters[i] += 1
            return tid

        def record(i, role, system, user, result, tid):
            text, turn_lps, token_ids = result
            n_resp = len(token_ids)
            aligned = list(turn_lps) + [0.0] * (n_resp - len(turn_lps))
            log_probs_list[i].extend(aligned[:n_resp])
            turn_ids_list[i].extend([tid] * n_resp)
            prompt_ids = self.tokenizer.encode(
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                ),
                add_special_tokens=False,
            )
            seq_input_ids_l[i].extend(prompt_ids)
            seq_step_ids_l[i].extend([-1] * len(prompt_ids))
            seq_input_ids_l[i].extend(token_ids)
            seq_step_ids_l[i].extend([tid] * n_resp)
            all_messages[i].append({
                "role_name":    role,
                "turn_id":      tid,
                "system":       system,
                "user":         user,
                "response":     text,
                "response_ids": token_ids,
            })

        def make_prompt(system, user):
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )

        def role_system_user(i, role, question, meta_plan):
            bb = blackboards[i]
            sys = ROLE_SYSTEM[role]()
            ctx = bb.to_text()
            if role == "proposer":
                usr = f"meta-plan：{meta_plan}\n问题：{question}\n当前状态：{ctx}"
            elif role == "critic":
                last = bb.traces[-1] if bb.traces else ("", "")
                usr = f"待审查解法：{last[0]}\n答案：{last[1]}\n当前状态：{ctx}"
            else:
                last = bb.traces[-1] if bb.traces else ("", "")
                usr = f"待验证答案：{last[1]}\n推理：{last[0]}\n当前状态：{ctx}"
            return sys, usr

        for _ in range(self.max_rounds):
            if not active:
                break

            # ── controller (batch) ───────────────────────────────────────────
            ctrl_info = []  # (i, tid, system, user)
            for i in active:
                sys = PromptTemplates.controller_system()
                usr = f"问题：{questions[i]}\n当前状态：{blackboards[i].to_text()}"
                ctrl_info.append((i, next_tid(i), sys, usr))

            ctrl_res = self.vllm_engine.generate_batch(
                [{"role": "controller", "prompt": make_prompt(s, u)} for _, _, s, u in ctrl_info]
            )

            still_active = []
            start_info = []  # (i, tid, role, system, user, meta_plan)
            for (i, tid, sys, usr), res in zip(ctrl_info, ctrl_res):
                record(i, "controller", sys, usr, res, tid)
                meta_plan = res[0]
                if self._parse_strategy(meta_plan) == "stop":
                    continue
                still_active.append(i)
                focus      = self._parse_focus(meta_plan)
                start_role = focus if focus != "balanced" else "proposer"
                s, u = role_system_user(i, start_role, questions[i], meta_plan)
                start_info.append((i, next_tid(i), start_role, s, u, meta_plan))

            active = still_active

            # ── start-role (batch) ───────────────────────────────────────────
            if not start_info:
                continue

            start_res = self.vllm_engine.generate_batch(
                [{"role": role, "prompt": make_prompt(s, u)}
                 for _, _, role, s, u, _ in start_info]
            )

            for (i, tid, role, sys, usr, meta_plan), res in zip(start_info, start_res):
                record(i, role, sys, usr, res, tid)
                response = res[0]
                self._write_to_blackboard(blackboards[i], role, response)
                round_turn_ids = [(tid, role)]
                role_outputs   = {role: response}

                # interaction turns (episode-serial; rare path)
                pending = [(role, response)]
                interaction_count = 0
                while pending and interaction_count < self.max_interactions:
                    initiator, init_out = pending.pop(0)
                    action, target, reason = self._parse_interaction(init_out)
                    if action != "none" and target != "none":
                        blackboards[i].add_message(Message(
                            agent_id=list(ROLE_NAMES.keys()).index(initiator),
                            msg_type=MessageType.INTERACTION,
                            content={"from": initiator, "action": action,
                                     "target": target, "reason": reason},
                        ))
                    if target in ROLE_NAMES and action not in ("none", "support"):
                        itid  = next_tid(i)
                        isys  = PromptTemplates.interaction_response_system(
                            ROLE_NAMES[target], action, init_out)
                        iusr  = f"黑板状态：{blackboards[i].to_text()}\n问题：{questions[i]}"
                        ires  = self.vllm_engine.generate_batch(
                            [{"role": target, "prompt": make_prompt(isys, iusr)}])[0]
                        record(i, target, isys, iusr, ires, itid)
                        round_turn_ids.append((itid, target))
                        resp_out = ires[0]
                        role_outputs[target] = resp_out
                        self._write_to_blackboard(blackboards[i], target, resp_out)
                        pending.append((target, resp_out))
                        interaction_count += 1

                if "proposer" not in role_outputs:
                    ptid = next_tid(i)
                    ps, pu = role_system_user(i, "proposer", questions[i], meta_plan)
                    pres = self.vllm_engine.generate_batch(
                        [{"role": "proposer", "prompt": make_prompt(ps, pu)}])[0]
                    record(i, "proposer", ps, pu, pres, ptid)
                    round_turn_ids.append((ptid, "proposer"))
                    prop_out = pres[0]
                    role_outputs["proposer"] = prop_out
                    self._write_to_blackboard(blackboards[i], "proposer", prop_out)

                _, prop_answer = self._parse_reasoning(role_outputs.get("proposer", ""))
                critic_flagged = "critic" in role_outputs and "无错误" not in role_outputs["critic"]
                verifier_score = (self._parse_score(role_outputs["verifier"])
                                  if "verifier" in role_outputs else None)
                round_records[i].append({
                    "proposer_answer": prop_answer,
                    "critic_flagged":  critic_flagged,
                    "verifier_score":  verifier_score,
                    "turn_ids":        round_turn_ids,
                })

        # ── finalise ─────────────────────────────────────────────────────────
        results = []
        for i in range(n):
            final_answer = self._majority_vote(blackboards[i])
            is_correct   = normalize_answer(final_answer) == normalize_answer(correct_answers[i])
            n_turns = max(turn_counters[i], 1)
            rewards = [0.0] * n_turns

            last_prop_tid = None
            for rnd in round_records[i]:
                for tid, role in rnd["turn_ids"]:
                    if role == "proposer" and tid < n_turns:
                        last_prop_tid = tid
            rewards[last_prop_tid if last_prop_tid is not None else n_turns - 1] += (
                1.0 if is_correct else -0.5)

            for rnd in round_records[i]:
                prop_correct = (normalize_answer(rnd["proposer_answer"]) ==
                                normalize_answer(correct_answers[i]))
                for tid, role in rnd["turn_ids"]:
                    if tid >= n_turns:
                        continue
                    if role == "critic" and rnd["critic_flagged"]:
                        rewards[tid] += 0.3 if is_correct else -0.1
                    elif role == "verifier" and rnd["verifier_score"] is not None:
                        if (prop_correct and rnd["verifier_score"] >= 0.5 or
                                not prop_correct and rnd["verifier_score"] < 0.5):
                            rewards[tid] += 0.1

            results.append({
                "messages":      all_messages[i],
                "turn_ids":      turn_ids_list[i],
                "log_probs":     log_probs_list[i],
                "seq_input_ids": seq_input_ids_l[i],
                "seq_step_ids":  seq_step_ids_l[i],
                "rewards":       rewards,
                "final_answer":  final_answer,
                "is_correct":    is_correct,
            })
        return results

    # ── single episode (kept for compatibility / non-vllm path) ─────────────
    def run_episode(self, question: str, correct_answer: str) -> dict:
        blackboard = Blackboard()
        all_messages = []
        turn_ids = []
        log_probs = []
        seq_input_ids = []
        seq_step_ids  = []
        turn_counter = [0]

        def next_turn_id():
            tid = turn_counter[0]
            turn_counter[0] += 1
            return tid

        round_records = []

        for _ in range(self.max_rounds):
            meta_plan = self._run_turn(
                role="controller",
                system=PromptTemplates.controller_system(),
                user=f"问题：{question}\n当前状态：{blackboard.to_text()}",
                all_messages=all_messages, turn_id=next_turn_id(),
                turn_ids=turn_ids, log_probs=log_probs,
                seq_input_ids=seq_input_ids, seq_step_ids=seq_step_ids,
            )
            if self._parse_strategy(meta_plan) == "stop":
                break

            focus = self._parse_focus(meta_plan)
            start_role = focus if focus != "balanced" else "proposer"
            role_outputs = {}
            round_turn_ids = []

            def tracked_next_turn_id(role):
                tid = next_turn_id()
                round_turn_ids.append((tid, role))
                return tid

            role_out = self._run_role(
                role=start_role, question=question, meta_plan=meta_plan,
                blackboard=blackboard, all_messages=all_messages,
                turn_id=tracked_next_turn_id(start_role), turn_ids=turn_ids,
                log_probs=log_probs, seq_input_ids=seq_input_ids, seq_step_ids=seq_step_ids,
            )
            role_outputs[start_role] = role_out
            self._write_to_blackboard(blackboard, start_role, role_out)

            pending = [(start_role, role_out)]
            interaction_count = 0
            while pending and interaction_count < self.max_interactions:
                initiator, initiator_out = pending.pop(0)
                action, target, reason = self._parse_interaction(initiator_out)
                if action != "none" and target != "none":
                    blackboard.add_message(Message(
                        agent_id=list(ROLE_NAMES.keys()).index(initiator),
                        msg_type=MessageType.INTERACTION,
                        content={"from": initiator, "action": action,
                                 "target": target, "reason": reason},
                    ))
                if target in ROLE_NAMES and action not in ("none", "support"):
                    resp_out = self._run_turn(
                        role=target,
                        system=PromptTemplates.interaction_response_system(
                            ROLE_NAMES[target], action, initiator_out),
                        user=f"黑板状态：{blackboard.to_text()}\n问题：{question}",
                        all_messages=all_messages, turn_id=tracked_next_turn_id(target),
                        turn_ids=turn_ids, log_probs=log_probs,
                        seq_input_ids=seq_input_ids, seq_step_ids=seq_step_ids,
                    )
                    role_outputs[target] = resp_out
                    self._write_to_blackboard(blackboard, target, resp_out)
                    pending.append((target, resp_out))
                    interaction_count += 1

            if "proposer" not in role_outputs:
                prop_out = self._run_role(
                    role="proposer", question=question, meta_plan=meta_plan,
                    blackboard=blackboard, all_messages=all_messages,
                    turn_id=tracked_next_turn_id("proposer"), turn_ids=turn_ids,
                    log_probs=log_probs, seq_input_ids=seq_input_ids, seq_step_ids=seq_step_ids,
                )
                role_outputs["proposer"] = prop_out
                self._write_to_blackboard(blackboard, "proposer", prop_out)

            _, prop_answer = self._parse_reasoning(role_outputs.get("proposer", ""))
            critic_flagged = "critic" in role_outputs and "无错误" not in role_outputs["critic"]
            verifier_score = (self._parse_score(role_outputs["verifier"])
                              if "verifier" in role_outputs else None)
            round_records.append({
                "proposer_answer": prop_answer,
                "critic_flagged":  critic_flagged,
                "verifier_score":  verifier_score,
                "turn_ids":        list(round_turn_ids),
            })

        final_answer = self._majority_vote(blackboard)
        is_correct = normalize_answer(final_answer) == normalize_answer(correct_answer)
        n_turns = turn_counter[0]
        if n_turns == 0:
            n_turns = 1
        rewards = [0.0] * n_turns

        last_proposer_tid = None
        for rnd in round_records:
            for tid, role in rnd["turn_ids"]:
                if role == "proposer" and tid < n_turns:
                    last_proposer_tid = tid
        rewards[last_proposer_tid if last_proposer_tid is not None else -1] += (
            1.0 if is_correct else -0.5)

        for rnd in round_records:
            prop_correct = normalize_answer(rnd["proposer_answer"]) == normalize_answer(correct_answer)
            for tid, role in rnd["turn_ids"]:
                if tid >= n_turns:
                    continue
                if role == "critic" and rnd["critic_flagged"]:
                    rewards[tid] += 0.3 if is_correct else -0.1
                elif role == "verifier" and rnd["verifier_score"] is not None:
                    correct_high = prop_correct and rnd["verifier_score"] >= 0.5
                    correct_low  = not prop_correct and rnd["verifier_score"] < 0.5
                    if correct_high or correct_low:
                        rewards[tid] += 0.1

        return {
            "messages":      all_messages,
            "turn_ids":      turn_ids,
            "log_probs":     log_probs,
            "seq_input_ids": seq_input_ids,
            "seq_step_ids":  seq_step_ids,
            "rewards":       rewards,
            "final_answer":  final_answer,
            "is_correct":    is_correct,
        }

    def _run_role(self, role, question, meta_plan, blackboard,
                  all_messages, turn_id, turn_ids, log_probs,
                  seq_input_ids=None, seq_step_ids=None) -> str:
        context = blackboard.to_text()
        if role == "proposer":
            user = f"meta-plan：{meta_plan}\n问题：{question}\n当前状态：{context}"
        elif role == "critic":
            last_trace = blackboard.traces[-1] if blackboard.traces else ("", "")
            user = f"待审查解法：{last_trace[0]}\n答案：{last_trace[1]}\n当前状态：{context}"
        else:
            last_trace = blackboard.traces[-1] if blackboard.traces else ("", "")
            user = f"待验证答案：{last_trace[1]}\n推理：{last_trace[0]}\n当前状态：{context}"
        return self._run_turn(
            role=role, system=ROLE_SYSTEM[role](), user=user,
            all_messages=all_messages, turn_id=turn_id,
            turn_ids=turn_ids, log_probs=log_probs,
            seq_input_ids=seq_input_ids, seq_step_ids=seq_step_ids,
        )

    def _run_turn(self, role, system, user, all_messages, turn_id, turn_ids, log_probs,
                  seq_input_ids=None, seq_step_ids=None) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]

        if self.vllm_engine is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            response, turn_lps, token_ids = self.vllm_engine.generate(role, prompt)
            resp_token_ids = token_ids
            n_resp = len(resp_token_ids)
            aligned_lps = list(turn_lps) + [0.0] * (n_resp - len(turn_lps))
            log_probs.extend(aligned_lps[:n_resp])
            turn_ids.extend([turn_id] * n_resp)
            if seq_input_ids is not None:
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                seq_input_ids.extend(prompt_ids)
                seq_step_ids.extend([-1] * len(prompt_ids))
                seq_input_ids.extend(resp_token_ids)
                seq_step_ids.extend([turn_id] * len(resp_token_ids))
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True, return_dict=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids, max_new_tokens=self.max_tokens,
                    return_dict_in_generate=True, output_scores=True,
                )
            new_tokens = outputs.sequences[0][input_ids.shape[1]:]
            for token_id, scores in zip(new_tokens, outputs.scores):
                log_probs.append(F.log_softmax(scores[0], dim=-1)[token_id].item())
                turn_ids.append(turn_id)
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            resp_token_ids = new_tokens.tolist()
            if seq_input_ids is not None:
                prompt_ids_list = input_ids[0].tolist()
                seq_input_ids.extend(prompt_ids_list)
                seq_step_ids.extend([-1] * len(prompt_ids_list))
                seq_input_ids.extend(resp_token_ids)
                seq_step_ids.extend([turn_id] * len(resp_token_ids))

        all_messages.append({
            "role_name":    role,
            "turn_id":      turn_id,
            "system":       system,
            "user":         user,
            "response":     response,
            "response_ids": resp_token_ids,
        })
        return response

    def _write_to_blackboard(self, blackboard: Blackboard, role: str, output: str):
        if role == "proposer":
            reasoning, answer = self._parse_reasoning(output)
            blackboard.add_message(Message(0, MessageType.TRACE, (reasoning, answer)))
        elif role == "critic":
            if "无错误" not in output:
                blackboard.add_message(Message(1, MessageType.FLAW, {"content": output}))
        elif role == "verifier":
            answer = blackboard.traces[-1][1] if blackboard.traces else ""
            blackboard.add_message(Message(2, MessageType.SCORE, (answer, self._parse_score(output))))

    def _parse_strategy(self, meta_plan: str) -> str:
        m = re.search(r"strategy:\s*(explore|refine|verify|stop)", meta_plan)
        return m.group(1) if m else "explore"

    def _parse_focus(self, meta_plan: str) -> str:
        m = re.search(r"focus:\s*(proposer|critic|verifier|balanced)", meta_plan)
        return m.group(1) if m else "balanced"

    def _parse_interaction(self, text: str) -> tuple:
        block = re.search(r"<interaction>(.*?)</interaction>", text, re.S)
        if not block:
            return "none", "none", ""
        content = block.group(1)
        action = re.search(r"action:\s*(\S+)", content)
        target = re.search(r"target:\s*(\S+)", content)
        reason = re.search(r"reason:\s*(.+)", content)
        target_str = (target.group(1) if target else "none")
        target_str = target_str if target_str in ROLE_NAMES else "none"
        return (action.group(1) if action else "none"), target_str, \
               (reason.group(1).strip() if reason else "")

    def _parse_reasoning(self, text: str) -> tuple:
        reasoning = re.search(r"推理过程：(.+?)(?=最终答案：|<|$)", text, re.S)
        answer    = re.search(r"最终答案：(.+)", text)
        if not answer:
            nums = re.findall(r"-?\d+\.?\d*", text)
            ans_str = nums[-1] if nums else ""
        else:
            ans_str = answer.group(1).strip()
        return (reasoning.group(1).strip() if reasoning else text, ans_str)

    def _parse_score(self, text: str) -> float:
        m = re.search(r"分数:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text)
        if not m:
            return 0.5
        return max(0.0, min(1.0, float(m.group(1))))

    def _majority_vote(self, blackboard: Blackboard) -> str:
        if not blackboard.traces:
            return ""
        return Counter(ans for _, ans in blackboard.traces).most_common(1)[0][0]
