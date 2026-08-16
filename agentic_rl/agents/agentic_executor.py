import re
from collections import Counter

import torch
import torch.nn.functional as F

from agents.grader import math_equal   # 分层严格判等（Hendrycks 归一化，无数字兜底）
from envs.blackboard import Blackboard, Message, MessageType
from llm.prompt_templates import PromptTemplates


ROLE_SYSTEM = {
    "proposer": PromptTemplates.proposer_system,
    "critic":   PromptTemplates.critic_system,
    "verifier": PromptTemplates.verifier_system,
}
ROLE_NAMES = {"proposer": "Proposer", "critic": "Critic", "verifier": "Verifier"}


class AgenticExecutor:
    def __init__(self, model, tokenizer, config, vllm_engine=None, eval_mode: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = config.get("max_tokens", 512)
        self.max_interactions = config.get("max_interactions", 3)
        self.max_rounds = config.get("max_rounds", 3)
        self.vllm_engine = vllm_engine
        # eval_mode: use greedy decoding (temperature=0) for deterministic evaluation;
        # training rollouts use temperature=1.0 for exploration.
        self.eval_mode = eval_mode
        self.temperature = 0.0 if eval_mode else 1.0
        # RACA controller reward hyperparameters
        self.ctrl_alpha = config.get("ctrl_alpha", 0.3)
        self.ctrl_beta  = config.get("ctrl_beta",  0.2)
        # ctrl_gamma penalises unused rounds when the answer is wrong, mirroring
        # ctrl_alpha's bonus for unused rounds when it is right. Without it,
        # stopping early is weakly dominant at every confidence level.
        self.ctrl_gamma = config.get("ctrl_gamma", 0.3)

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
        # Controller turn that ended each episode via strategy:stop (None if the
        # episode ran out of max_rounds instead).
        stop_ctrl_tids  = [None] * n
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
                [{"role": "controller", "prompt": make_prompt(s, u),
                  "temperature": self.temperature} for _, _, s, u in ctrl_info]
            )

            still_active = []
            start_info = []  # (i, tid, role, system, user, meta_plan)
            # ctrl_tid_map: episode_idx → controller turn_id for this round
            ctrl_tid_map = {}
            for (i, tid, sys, usr), res in zip(ctrl_info, ctrl_res):
                record(i, "controller", sys, usr, res, tid)
                ctrl_tid_map[i] = tid          # store controller tid before filtering
                meta_plan = res[0]
                if self._parse_strategy(meta_plan) == "stop":
                    # Remember the turn that ended the episode. Stop turns never
                    # reach round_records, so without this they get no reward
                    # entry, no advantage, and therefore no gradient — the single
                    # most consequential controller action went untrained.
                    stop_ctrl_tids[i] = tid
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
                [{"role": role, "prompt": make_prompt(s, u),
                  "temperature": self.temperature}
                 for _, _, role, s, u, _ in start_info]
            )

            # initialise per-episode round state
            # round_turn_ids starts with (ctrl_tid, "controller") so the controller
            # turn is included in round_records and gets a RACA reward signal.
            ep_st = {}
            for (i, tid, role, sys, usr, meta_plan), res in zip(start_info, start_res):
                record(i, role, sys, usr, res, tid)
                response = res[0]
                self._write_to_blackboard(blackboards[i], role, response)
                ep_st[i] = {
                    "role_outputs":   {role: response},
                    "round_turn_ids": [(ctrl_tid_map[i], "controller"), (tid, role)],
                    "meta_plan":      meta_plan,
                    "pending":        [(role, response)],
                }

            # interaction turns — batched across all episodes at each depth
            for _ in range(self.max_interactions):
                batch_req = []
                for i, st in ep_st.items():
                    if not st["pending"]:
                        continue
                    initiator, init_out = st["pending"].pop(0)
                    action, target, reason = self._parse_interaction(init_out)
                    if action != "none" and target != "none":
                        blackboards[i].add_message(Message(
                            agent_id=list(ROLE_NAMES.keys()).index(initiator),
                            msg_type=MessageType.INTERACTION,
                            content={"from": initiator, "action": action,
                                     "target": target, "reason": reason},
                        ))
                    if target in ROLE_NAMES and action not in ("none", "support"):
                        itid = next_tid(i)
                        isys = PromptTemplates.interaction_response_system(
                            ROLE_NAMES[target], action, init_out)
                        iusr = f"黑板状态：{blackboards[i].to_text()}\n问题：{questions[i]}"
                        batch_req.append((i, itid, target, isys, iusr))
                if not batch_req:
                    break
                ires_all = self.vllm_engine.generate_batch(
                    [{"role": t, "prompt": make_prompt(s, u),
                      "temperature": self.temperature} for _, _, t, s, u in batch_req]
                )
                for (i, itid, target, isys, iusr), ires in zip(batch_req, ires_all):
                    record(i, target, isys, iusr, ires, itid)
                    resp_out = ires[0]
                    ep_st[i]["role_outputs"][target] = resp_out
                    ep_st[i]["round_turn_ids"].append((itid, target))
                    self._write_to_blackboard(blackboards[i], target, resp_out)
                    ep_st[i]["pending"].append((target, resp_out))

            # proposer fallback — batched across all episodes that need it
            prop_req = []
            for i, st in ep_st.items():
                if "proposer" not in st["role_outputs"]:
                    ptid = next_tid(i)
                    ps, pu = role_system_user(i, "proposer", questions[i], st["meta_plan"])
                    prop_req.append((i, ptid, ps, pu))
            if prop_req:
                prop_res_all = self.vllm_engine.generate_batch(
                    [{"role": "proposer", "prompt": make_prompt(s, u),
                      "temperature": self.temperature} for _, _, s, u in prop_req]
                )
                for (i, ptid, ps, pu), pres in zip(prop_req, prop_res_all):
                    record(i, "proposer", ps, pu, pres, ptid)
                    prop_out = pres[0]
                    ep_st[i]["role_outputs"]["proposer"] = prop_out
                    ep_st[i]["round_turn_ids"].append((ptid, "proposer"))
                    self._write_to_blackboard(blackboards[i], "proposer", prop_out)

            # commit round records
            for i, st in ep_st.items():
                ro = st["role_outputs"]
                _, prop_answer = self._parse_reasoning(ro.get("proposer", ""))
                critic_flagged = "critic" in ro and self._critic_found_errors(ro["critic"])
                verifier_score = (self._parse_score(ro["verifier"])
                                  if "verifier" in ro else None)
                round_records[i].append({
                    "proposer_answer": prop_answer,
                    "critic_flagged":  critic_flagged,
                    "verifier_score":  verifier_score,
                    "strategy":        self._parse_strategy(st["meta_plan"]),
                    "turn_ids":        st["round_turn_ids"],
                })

        # ── finalise ─────────────────────────────────────────────────────────
        results = []
        for i in range(n):
            final_answer = self._majority_vote(blackboards[i])
            is_correct   = math_equal(final_answer, correct_answers[i])

            raca_turn_data = self._compute_raca_turn_data(
                round_records[i], correct_answers[i], is_correct,
                self.max_rounds, self.ctrl_alpha, self.ctrl_beta,
                self.ctrl_gamma, stop_ctrl_tid=stop_ctrl_tids[i],
            )

            results.append({
                "messages":       all_messages[i],
                "turn_ids":       turn_ids_list[i],
                "log_probs":      log_probs_list[i],
                "seq_input_ids":  seq_input_ids_l[i],
                "seq_step_ids":   seq_step_ids_l[i],
                "raca_turn_data": raca_turn_data,
                "final_answer":   final_answer,
                "is_correct":     is_correct,
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
        stop_ctrl_tid = None

        for _ in range(self.max_rounds):
            ctrl_tid = next_turn_id()
            meta_plan = self._run_turn(
                role="controller",
                system=PromptTemplates.controller_system(),
                user=f"问题：{question}\n当前状态：{blackboard.to_text()}",
                all_messages=all_messages, turn_id=ctrl_tid,
                turn_ids=turn_ids, log_probs=log_probs,
                seq_input_ids=seq_input_ids, seq_step_ids=seq_step_ids,
            )
            if self._parse_strategy(meta_plan) == "stop":
                stop_ctrl_tid = ctrl_tid
                break

            focus = self._parse_focus(meta_plan)
            start_role = focus if focus != "balanced" else "proposer"
            role_outputs = {}
            # Initialise with controller so it appears in round_records and gets RACA reward.
            round_turn_ids = [(ctrl_tid, "controller")]

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
            critic_flagged = "critic" in role_outputs and self._critic_found_errors(role_outputs["critic"])
            verifier_score = (self._parse_score(role_outputs["verifier"])
                              if "verifier" in role_outputs else None)
            round_records.append({
                "proposer_answer": prop_answer,
                "critic_flagged":  critic_flagged,
                "verifier_score":  verifier_score,
                "strategy":        self._parse_strategy(meta_plan),
                "turn_ids":        list(round_turn_ids),
            })

        final_answer = self._majority_vote(blackboard)
        is_correct = math_equal(final_answer, correct_answer)

        raca_turn_data = self._compute_raca_turn_data(
            round_records, correct_answer, is_correct,
            self.max_rounds, self.ctrl_alpha, self.ctrl_beta,
            self.ctrl_gamma, stop_ctrl_tid=stop_ctrl_tid,
        )

        return {
            "messages":       all_messages,
            "turn_ids":       turn_ids,
            "log_probs":      log_probs,
            "seq_input_ids":  seq_input_ids,
            "seq_step_ids":   seq_step_ids,
            "raca_turn_data": raca_turn_data,
            "final_answer":   final_answer,
            "is_correct":     is_correct,
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
            response, turn_lps, token_ids = self.vllm_engine.generate(
                role, prompt, temperature=self.temperature)
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
            if self._critic_found_errors(output):
                blackboard.add_message(Message(1, MessageType.FLAW, {"content": output}))
        elif role == "verifier":
            answer = blackboard.traces[-1][1] if blackboard.traces else ""
            blackboard.add_message(Message(2, MessageType.SCORE, (answer, self._parse_score(output))))

    def _compute_raca_turn_data(
        self,
        round_records: list,
        correct_answer: str,
        is_correct: bool,
        max_rounds: int,
        alpha: float,
        beta: float,
        gamma: float = 0.3,
        stop_ctrl_tid: int | None = None,
    ) -> dict:
        """Compute RACA per-turn reward data (Phase 2 of the algorithm).

        Returns a dict mapping turn_id → {"role": str, "round": int, "reward": float}.
        The trainer reads this to build anchor groups and compute advantages.
        """
        turn_data: dict = {}

        # Pre-compute per-round proposer correctness for critic's causal reward.
        prop_correct_per_round: list = []
        for rnd in round_records:
            pc = math_equal(rnd["proposer_answer"], correct_answer)
            prop_correct_per_round.append(pc)

        last_ctrl_tid = None

        for rnd_idx, rnd in enumerate(round_records):
            p_t      = prop_correct_per_round[rnd_idx]
            strategy = rnd.get("strategy", "explore")   # controller strategy for this round
            # p_{t+1}: next round's proposer correctness, or episode outcome for last round
            if rnd_idx + 1 < len(round_records):
                p_t1 = prop_correct_per_round[rnd_idx + 1]
            else:
                p_t1 = is_correct

            for tid, role in rnd["turn_ids"]:
                if role == "proposer":
                    reward = 1.0 if p_t else 0.0

                elif role == "critic":
                    f = rnd["critic_flagged"]
                    if f and not p_t:       # 真阳性：正确识别错误
                        reward = 0.3 * float(p_t1) + 0.1 * (1.0 - float(p_t1))
                    elif f and p_t:         # 假阳性：干扰了正确解
                        reward = -0.2
                    elif not f and p_t:     # 真阴性：正确放行
                        reward = 0.1
                    else:                   # 漏检：未发现错误
                        reward = 0.0

                elif role == "verifier":
                    vs = rnd["verifier_score"]
                    if vs is not None:
                        reward = 1.0 - abs(vs - float(p_t))
                    else:
                        reward = 0.0

                elif role == "controller":
                    reward = 0.0            # placeholder; last ctrl turn updated below
                    last_ctrl_tid = tid

                else:
                    reward = 0.0

                turn_data[tid] = {"role": role, "round": rnd_idx, "strategy": strategy, "reward": reward}

        # ── Controller outcome reward ────────────────────────────────────────
        # Exactly one controller turn carries the episode signal (every other
        # controller turn keeps its 0.0 placeholder), so the trainer's per-episode
        # sum over controller rewards stays equal to this single value.
        #
        # The turn that ends the episode must own the consequence of ending it:
        # prefer the explicit "stop" turn, and fall back to the last working turn
        # for episodes that ended by exhausting max_rounds. Previously the stop
        # turn never entered round_records, so it got no reward entry at all while
        # the unused-budget bonus landed on an earlier "continue" turn. An episode
        # that stopped on round 0 produced entirely empty turn data and was then
        # dropped from the batch, so immediate-stop rollouts trained on nothing.
        t_stop      = len(round_records)
        remaining   = (max_rounds - t_stop) / max(max_rounds, 1)
        outcome_tid = stop_ctrl_tid if stop_ctrl_tid is not None else last_ctrl_tid

        if outcome_tid is not None:
            # The unused-round term is symmetric: saved rounds are a bonus when
            # the answer is right and an equal-magnitude penalty when it is wrong.
            # Its coefficient is (p*alpha - (1-p)*gamma), so with gamma == alpha
            # early stopping only pays off once the answer is more likely right
            # than wrong. The old one-sided form added the bonus but left the
            # wrong-answer cost flat, making "stop now" weakly dominant at every
            # confidence level — which drove avg_turns to the floor.
            correct = float(is_correct)
            ctrl_reward = (
                correct
                + alpha * correct * remaining
                - beta  * (1.0 - correct)
                - gamma * (1.0 - correct) * remaining
            )
            entry = turn_data.get(outcome_tid)
            if entry is None:
                # Stop turns are absent from round_records — create their entry.
                turn_data[outcome_tid] = {
                    "role":     "controller",
                    "round":    t_stop,
                    "strategy": "stop",
                    "reward":   ctrl_reward,
                }
            else:
                entry["reward"] = ctrl_reward

        return turn_data

    def _critic_found_errors(self, critic_output: str) -> bool:
        """Robustly detect whether the critic found errors.

        Old check: '"无错误" not in output' — too brittle because interaction
        responses overwrite role_outputs["critic"] with a response that doesn't
        follow the critic format and never contains "无错误", causing
        critic_flagged to be almost always True.

        New logic: parse the '错误分析' section specifically. If the section
        says '无错误'/'无错'/'正确', return False. If it has other content,
        return True. If the section is missing (e.g. interaction response),
        conservatively return False.
        """
        # Fast path: if '无错误' appears anywhere, the critic approved.
        if "无错误" in critic_output or "无错" in critic_output:
            return False
        # Try to find the '错误分析' section
        err_match = re.search(r"错误分析[：:]\s*(.+?)(?=<|$)", critic_output, re.S)
        if err_match:
            err_text = err_match.group(1).strip()
            # Non-empty error analysis that doesn't say 'no error' → errors found
            return bool(err_text) and "无错误" not in err_text and "无错" not in err_text
        # No '错误分析' section — likely an interaction response, not a review.
        # Conservative: don't flag.
        return False

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
