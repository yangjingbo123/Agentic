"""
Blackboard-based multi-role rollout using vLLM with per-role LoRA adapters.

Episode structure per round:
  Controller -> Proposer -> Critic -> Verifier

Each role uses its own LoRA adapter via vLLM LoRARequest.
The Blackboard accumulates traces/flaws/scores across rounds and provides
context to every role at each turn.

Reward computation is intentionally left to the external reward_fn so that
this class only handles trajectory collection (separation of concerns).
"""

import re
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams

from verl import DataProto

# LoRARequest may not be available in older vllm versions
try:
    from vllm.lora.request import LoRARequest
    _LORA_AVAILABLE = True
except ImportError:
    _LORA_AVAILABLE = False
    LoRARequest = None

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
from envs.blackboard import Blackboard, Message, MessageType


class VLLMBlackboardRollout:
    """
    Runs the blackboard multi-agent episode loop and returns a DataProto
    whose tensor keys match the format expected by merge_roles_data():
      {role}_input_ids, {role}_attention_mask, {role}_labels,
      {role}_step_ids, {role}_position_ids, {role}_num_gen_tokens,
      {role}_turn_level_reward   (zeros here; filled by reward_fn)
      {role}_turn_level_shaped_reward (zeros here; filled by reward_fn)
    Plus non-tensor keys: num_turns, history, finish_reason, response, final_answer.
    """

    _ADAPTER_ID_BASE = 1

    def __init__(
        self,
        model_path: str,
        config,
        tokenizer: PreTrainedTokenizer,
        model_hf_config,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.max_seq_len = config.prompt_length + config.response_length
        self.max_num_turns = config.max_num_turns

        # Build per-role LoRARequest objects
        # config.lora_adapter_paths: {role_name: "/path/to/adapter"}
        self._lora_requests: Dict[str, Optional[object]] = {}
        adapter_paths = dict(config.get("lora_adapter_paths", {}))
        if adapter_paths and not _LORA_AVAILABLE:
            raise RuntimeError(
                "LoRARequest not available in this vllm version. "
                "Upgrade to vllm>=0.4.0 for LoRA support."
            )
        for idx, (role, path) in enumerate(adapter_paths.items()):
            self._lora_requests[role] = LoRARequest(
                lora_name=role,
                lora_int_id=self._ADAPTER_ID_BASE + idx,
                lora_path=path,
            )

        tp = config.get("tensor_model_parallel_size", 1)
        self.inference_engine = LLM(
            model=model_path,
            enable_lora=bool(adapter_paths),
            max_lora_rank=config.get("lora_rank", 16),
            tensor_parallel_size=tp,
            dtype=config.dtype,
            enforce_eager=config.get("enforce_eager", False),
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            max_model_len=self.max_seq_len,
            disable_log_stats=config.get("disable_log_stats", True),
            enable_prefix_caching=True,
            seed=config.get("seed", 0),
            enable_chunked_prefill=config.get("enable_chunked_prefill", False),
            max_num_batched_tokens=config.get("max_num_batched_tokens", None),
            max_num_seqs=config.get("max_num_seqs", 256),
        )
        self.inference_engine.sleep(level=1)

        self.sampling_params = SamplingParams(
            n=1,
            max_tokens=config.response_length,
            temperature=config.get("temperature", 1.0),
            top_p=config.get("top_p", 1.0),
            detokenize=True,
        )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def multi_turn_generate_sequences(
        self,
        prompts: DataProto,
        tokenizer: PreTrainedTokenizer,
        max_num_turns: int,
        finish_flag: Optional[str],
        agent_roles: List[str],
        system_prompts: Dict[str, str],
    ) -> DataProto:
        questions = list(prompts.non_tensor_batch["question"])
        episodes = self._run_episode_batch(
            questions=questions,
            agent_roles=agent_roles,
            system_prompts=system_prompts,
            max_num_turns=max_num_turns,
        )
        return self._pack_to_dataproto(episodes, agent_roles)

    # -------------------------------------------------------------------------
    # Blackboard episode loop
    # -------------------------------------------------------------------------

    def _run_episode_batch(
        self,
        questions: List[str],
        agent_roles: List[str],
        system_prompts: Dict[str, str],
        max_num_turns: int,
    ) -> List[dict]:
        bsz = len(questions)
        blackboards = [Blackboard() for _ in range(bsz)]
        episodes: List[dict] = [{"messages": [], "final_answer": ""} for _ in range(bsz)]
        turn_counter = [0] * bsz

        # How many rounds fit in max_num_turns given the number of roles
        rounds = max(1, max_num_turns // len(agent_roles))

        for _round in range(rounds):
            # Controller
            if "controller" in agent_roles:
                prompts_c, tids_c = self._build_prompts(
                    "controller", questions, blackboards,
                    system_prompts.get("controller", ""), turn_counter)
                outputs_c = self._generate("controller", prompts_c)
                self._record("controller", episodes, prompts_c, outputs_c, tids_c)
                turn_counter = [t + 1 for t in turn_counter]

            # Proposer
            if "proposer" in agent_roles:
                prompts_p, tids_p = self._build_prompts(
                    "proposer", questions, blackboards,
                    system_prompts.get("proposer", ""), turn_counter)
                outputs_p = self._generate("proposer", prompts_p)
                for i, (out, bb) in enumerate(zip(outputs_p, blackboards)):
                    reasoning, answer = self._parse_proposer(out)
                    bb.add_message(Message(1, MessageType.TRACE, (reasoning, answer)))
                self._record("proposer", episodes, prompts_p, outputs_p, tids_p)
                turn_counter = [t + 1 for t in turn_counter]

            # Critic
            if "critic" in agent_roles:
                prompts_cr, tids_cr = self._build_prompts(
                    "critic", questions, blackboards,
                    system_prompts.get("critic", ""), turn_counter)
                outputs_cr = self._generate("critic", prompts_cr)
                for i, (out, bb) in enumerate(zip(outputs_cr, blackboards)):
                    if self._critic_flagged(out):
                        bb.add_message(Message(2, MessageType.FLAW, {"content": out}))
                self._record("critic", episodes, prompts_cr, outputs_cr, tids_cr)
                turn_counter = [t + 1 for t in turn_counter]

            # Verifier
            if "verifier" in agent_roles:
                prompts_v, tids_v = self._build_prompts(
                    "verifier", questions, blackboards,
                    system_prompts.get("verifier", ""), turn_counter)
                outputs_v = self._generate("verifier", prompts_v)
                for i, (out, bb) in enumerate(zip(outputs_v, blackboards)):
                    ans, score = self._parse_verifier(out)
                    bb.add_message(Message(3, MessageType.SCORE, (ans, score)))
                self._record("verifier", episodes, prompts_v, outputs_v, tids_v)
                turn_counter = [t + 1 for t in turn_counter]

        # Majority-vote final answer per episode
        for ep, bb in zip(episodes, blackboards):
            ep["final_answer"] = self._majority_vote(bb)
            ep["blackboard_text"] = bb.to_text()

        return episodes

    # -------------------------------------------------------------------------
    # vLLM generation
    # -------------------------------------------------------------------------

    def _generate(self, role: str, prompts: List[str]) -> List[str]:
        lora_req = self._lora_requests.get(role)
        outputs = self.inference_engine.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
            lora_request=lora_req,
            use_tqdm=False,
        )
        return [o.outputs[0].text for o in outputs]

    def _build_prompts(
        self,
        role: str,
        questions: List[str],
        blackboards: List[Blackboard],
        system_prompt: str,
        turn_counter: List[int],
    ) -> Tuple[List[str], List[int]]:
        prompts, tids = [], []
        for i, (q, bb) in enumerate(zip(questions, blackboards)):
            user_content = f"Question: {q}\n\nBlackboard:\n{bb.to_text()}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            tids.append(turn_counter[i])
        return prompts, tids

    def _record(
        self,
        role: str,
        episodes: List[dict],
        prompts: List[str],
        outputs: List[str],
        tids: List[int],
    ):
        for i, (prompt, resp, tid) in enumerate(zip(prompts, outputs, tids)):
            resp_ids = self.tokenizer.encode(resp, add_special_tokens=False)
            episodes[i]["messages"].append({
                "role_name": role,
                "turn_id": tid,
                "prompt": prompt,
                "response": resp,
                "response_ids": resp_ids,
            })

    # -------------------------------------------------------------------------
    # Parsing helpers
    # -------------------------------------------------------------------------

    def _parse_proposer(self, text: str) -> Tuple[str, str]:
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        answer = m.group(1).strip() if m else ""
        reasoning = text[: text.find("<answer>")] if m else text
        return reasoning.strip(), answer

    def _critic_flagged(self, text: str) -> bool:
        return bool(re.search(r"<flaw>|错误|error|mistake", text, re.IGNORECASE))

    def _parse_verifier(self, text: str) -> Tuple[str, float]:
        m_score = re.search(r"<score>([\d.]+)", text)  # don't require closing tag
        m_ans = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        try:
            score = float(m_score.group(1)) if m_score else 0.5
        except (ValueError, AttributeError):
            score = 0.5
        answer = m_ans.group(1).strip() if m_ans else ""
        return answer, score

    def _majority_vote(self, bb: Blackboard) -> str:
        if not bb.traces:
            return ""
        counts: Dict[str, int] = defaultdict(int)
        for _, ans in bb.traces:
            if ans:
                counts[ans] += 1
        return max(counts, key=counts.get) if counts else ""

    # -------------------------------------------------------------------------
    # DataProto packing
    # -------------------------------------------------------------------------

    def _pack_to_dataproto(
        self, episodes: List[dict], agent_roles: List[str]
    ) -> DataProto:
        bsz = len(episodes)
        L = self.max_seq_len
        T = self.max_num_turns

        # Pre-allocate per-role tensors
        rt: Dict[str, dict] = {}
        for role in agent_roles:
            rt[role] = {
                "input_ids":        torch.full((bsz, L), self.pad_token_id, dtype=torch.long),
                "attention_mask":   torch.zeros(bsz, L, dtype=torch.long),
                "labels":           torch.full((bsz, L), -100, dtype=torch.long),
                "step_ids":         torch.full((bsz, L), -100, dtype=torch.long),
                "position_ids":     torch.zeros(bsz, L, dtype=torch.long),
                "num_gen_tokens":   torch.zeros(bsz, T, dtype=torch.long),
                "turn_finished":    torch.zeros(bsz, dtype=torch.long),  # 0=finished normally
                # reward placeholders — filled later by reward_fn
                "turn_level_reward":        torch.zeros(bsz, T),
                "turn_level_shaped_reward": torch.zeros(bsz, T),
            }

        num_turns_arr, history_arr, final_ans_arr = [], [], []

        for i, ep in enumerate(episodes):
            by_role: Dict[str, List[dict]] = defaultdict(list)
            for msg in ep["messages"]:
                by_role[msg["role_name"]].append(msg)

            for role in agent_roles:
                msgs = by_role.get(role, [])
                d = rt[role]
                flat_ids, flat_step_ids = [], []
                for t_idx, msg in enumerate(msgs):
                    prompt_ids = self.tokenizer.encode(
                        msg["prompt"], add_special_tokens=False)
                    resp_ids = msg["response_ids"]
                    flat_ids.extend(prompt_ids)
                    flat_step_ids.extend([-100] * len(prompt_ids))
                    flat_ids.extend(resp_ids)
                    flat_step_ids.extend([msg["turn_id"]] * len(resp_ids))
                    if t_idx < T:
                        d["num_gen_tokens"][i, t_idx] = len(resp_ids)

                seq_len = min(len(flat_ids), L)
                if seq_len == 0:
                    continue
                ids_t = torch.tensor(flat_ids[:seq_len], dtype=torch.long)
                step_t = torch.tensor(flat_step_ids[:seq_len], dtype=torch.long)
                d["input_ids"][i, :seq_len] = ids_t
                d["attention_mask"][i, :seq_len] = 1
                d["step_ids"][i, :seq_len] = step_t
                d["labels"][i, :seq_len] = torch.where(
                    step_t >= 0, ids_t, torch.full_like(ids_t, -100))
                d["position_ids"][i, :seq_len] = torch.arange(seq_len)

            num_turns_arr.append(len(ep["messages"]))
            # strip response_ids from history to avoid slow serialization
            history_arr.append([{k: v for k, v in m.items() if k != "response_ids"} for m in ep["messages"]])
            final_ans_arr.append(ep.get("final_answer", ""))

        # Assemble flat tensor dict (merge_roles_data expects {role}_* keys)
        tensor_batch: dict = {}
        for role in agent_roles:
            for key, val in rt[role].items():
                tensor_batch[f"{role}_{key}"] = val

        non_tensor_batch = {
            "num_turns":    np.array(num_turns_arr, dtype=object),
            "history":      np.array(history_arr, dtype=object),
            "finish_reason": np.array(["stop"] * bsz, dtype=object),
            "response":     np.array(final_ans_arr, dtype=object),
            # final_answer used by reward_fn to score the episode
            "final_answer": np.array(final_ans_arr, dtype=object),
        }

        result = DataProto.from_dict(tensor_batch, non_tensors=non_tensor_batch)
        result.meta_info['agent_roles'] = agent_roles
        return result
