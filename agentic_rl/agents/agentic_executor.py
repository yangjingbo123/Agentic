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
    def __init__(self, models: dict, tokenizer, config, vllm_engine=None):
        self.models = models
        self.tokenizer = tokenizer
        self.max_tokens = config.get("max_tokens", 512)
        self.max_interactions = config.get("max_interactions", 3)
        self.max_rounds = config.get("max_rounds", 3)
        self.vllm_engine = vllm_engine

    def run_episode(self, question: str, correct_answer: str) -> dict:
        blackboard = Blackboard()
        all_messages = []   # list of (role_name, system, user, response)
        turn_ids = []
        log_probs = []
        turn_counter = [0]

        def next_turn_id():
            tid = turn_counter[0]
            turn_counter[0] += 1
            return tid

        # ── 低层：多轮协作，Controller 每轮重新调度 ──────────────────────
        round_records = []
        max_rounds = self.max_rounds  # 全局上限，防止无限循环

        for _ in range(max_rounds):
            # Controller 每轮开头读取最新黑板，动态决策
            meta_plan = self._run_turn(
                role="controller",
                system=PromptTemplates.controller_system(),
                user=f"问题：{question}\n当前状态：{blackboard.to_text()}",
                all_messages=all_messages, turn_id=next_turn_id(),
                turn_ids=turn_ids, log_probs=log_probs,
            )
            strategy = self._parse_strategy(meta_plan)
            if strategy == "stop":
                break  # Controller 主动决定停止

            focus = self._parse_focus(meta_plan)
            start_role = focus if focus != "balanced" else "proposer"
            role_outputs: dict[str, str] = {}
            round_turn_ids = []

            def tracked_next_turn_id():
                tid = next_turn_id()
                round_turn_ids.append(tid)
                return tid

            role_out = self._run_role(
                role=start_role, question=question, meta_plan=meta_plan,
                blackboard=blackboard, all_messages=all_messages,
                turn_id=tracked_next_turn_id(), turn_ids=turn_ids, log_probs=log_probs,
            )
            role_outputs[start_role] = role_out
            self._write_to_blackboard(blackboard, start_role, role_out)

            # 交互路由
            interaction_count = 0
            pending = [(start_role, role_out)]
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
                            ROLE_NAMES[target], action, initiator_out
                        ),
                        user=f"黑板状态：{blackboard.to_text()}\n问题：{question}",
                        all_messages=all_messages, turn_id=tracked_next_turn_id(),
                        turn_ids=turn_ids, log_probs=log_probs,
                    )
                    role_outputs[target] = resp_out
                    self._write_to_blackboard(blackboard, target, resp_out)
                    pending.append((target, resp_out))
                    interaction_count += 1

            if "proposer" not in role_outputs:
                prop_out = self._run_role(
                    role="proposer", question=question, meta_plan=meta_plan,
                    blackboard=blackboard, all_messages=all_messages,
                    turn_id=tracked_next_turn_id(), turn_ids=turn_ids, log_probs=log_probs,
                )
                role_outputs["proposer"] = prop_out
                self._write_to_blackboard(blackboard, "proposer", prop_out)

            # 记录本round的中间信息
            _, prop_answer = self._parse_reasoning(role_outputs.get("proposer", ""))
            critic_flagged = "critic" in role_outputs and "无错误" not in role_outputs["critic"]
            round_records.append({
                "proposer_answer": prop_answer,
                "critic_flagged":  critic_flagged,
                "turn_ids":        list(round_turn_ids),
            })

        final_answer = self._majority_vote(blackboard)
        is_correct = normalize_answer(final_answer) == normalize_answer(correct_answer)

        n_turns = turn_counter[0]
        rewards = [0.0] * n_turns

        # ── 中间奖励 ──────────────────────────────────────────────────────
        n_proposer_turns = len(round_records)  # 每round至少一个Proposer turn

        for record in round_records:
            prop_correct = normalize_answer(record["proposer_answer"]) == normalize_answer(correct_answer)

            # Proposer校准奖励：episode内总和cap at 0.3（归一化）
            if record["turn_ids"] and prop_correct:
                rewards[record["turn_ids"][0]] += 0.3 / n_proposer_turns

            # Critic挑错时机校准奖励（不评价flaw语义质量，只评价挑错时机）
            if record["turn_ids"]:
                critic_tid = record["turn_ids"][1] if len(record["turn_ids"]) > 1 else record["turn_ids"][0]
                if record["critic_flagged"]:
                    rewards[critic_tid] += 0.2 if not prop_correct else -0.1

        # 终局奖励（主信号，只分配给最后一个turn）
        rewards[-1] += 1.0 if is_correct else 0.0

        return {
            "messages":     all_messages,
            "turn_ids":     turn_ids,
            "log_probs":    log_probs,
            "rewards":      rewards,
            "final_answer": final_answer,
            "is_correct":   is_correct,
        }

    def _run_role(self, role, question, meta_plan, blackboard,
                  all_messages, turn_id, turn_ids, log_probs) -> str:
        context = blackboard.to_text()
        if role == "proposer":
            user = f"meta-plan：{meta_plan}\n问题：{question}\n当前状态：{context}"
        elif role == "critic":
            last_trace = blackboard.traces[-1] if blackboard.traces else ("", "")
            user = f"待审查解法：{last_trace[0]}\n答案：{last_trace[1]}\n当前状态：{context}"
        else:  # verifier
            last_trace = blackboard.traces[-1] if blackboard.traces else ("", "")
            user = f"待验证答案：{last_trace[1]}\n推理：{last_trace[0]}\n当前状态：{context}"

        return self._run_turn(
            role=role, system=ROLE_SYSTEM[role](), user=user,
            all_messages=all_messages, turn_id=turn_id,
            turn_ids=turn_ids, log_probs=log_probs,
        )

    def _run_turn(self, role, system, user, all_messages, turn_id, turn_ids, log_probs) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]

        if self.vllm_engine is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response, turn_lps = self.vllm_engine.generate(role, prompt)
            log_probs.extend(turn_lps)
            turn_ids.extend([turn_id] * len(turn_lps))
        else:
            model = self.models[role]
            if hasattr(model, 'set_role'):
                model.set_role(role)
            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True,
                return_dict=True,
            ).to(model.device)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = model.generate(
                    input_ids, max_new_tokens=self.max_tokens,
                    return_dict_in_generate=True, output_scores=True,
                )
            new_tokens = outputs.sequences[0][input_ids.shape[1]:]
            for token_id, scores in zip(new_tokens, outputs.scores):
                lp = F.log_softmax(scores[0], dim=-1)[token_id].item()
                log_probs.append(lp)
                turn_ids.append(turn_id)
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        all_messages.append({
            "role_name": role,
            "system":    system,
            "user":      user,
            "response":  response,
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

    def _parse_interaction(self, text: str) -> tuple[str, str, str]:
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

    def _parse_reasoning(self, text: str) -> tuple[str, str]:
        reasoning = re.search(r"推理过程：(.+?)(?=最终答案：|<|$)", text, re.S)
        answer    = re.search(r"最终答案：(.+)", text)
        return (reasoning.group(1).strip() if reasoning else text,
                answer.group(1).strip()    if answer    else "")

    def _parse_score(self, text: str) -> float:
        m = re.search(r"分数:\s*([0-9.]+)", text)
        return float(m.group(1)) if m else 0.5

    def _majority_vote(self, blackboard: Blackboard) -> str:
        if not blackboard.traces:
            return ""
        return Counter(ans for _, ans in blackboard.traces).most_common(1)[0][0]
