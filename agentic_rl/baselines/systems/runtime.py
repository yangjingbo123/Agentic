"""Small batched runtimes for system-level baselines."""

from collections import Counter

from agents.agentic_executor import math_equal
from agents.parsing import parse_reasoning, parse_score
from baselines.systems import prompts


def render(tokenizer, system, user):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def request(role, prompt, temperature):
    return {"role": role, "prompt": prompt, "temperature": temperature}


def unpack(result):
    text, log_probs, token_ids = result
    return text, list(log_probs), list(token_ids)


def message(role, prompt, text, lps, ids):
    return {"role_name": role, "turn_id": 0, "prompt_text": prompt,
            "response": text, "response_ids": ids, "log_probs": lps}


def majority_answer(answers):
    """Majority vote under the project's mathematical-equivalence grader."""
    valid = [a for a in answers if a]
    if not valid:
        return ""
    representatives = []
    counts = []
    for answer in valid:
        matched = None
        for idx, representative in enumerate(representatives):
            if math_equal(answer, representative):
                matched = idx
                break
        if matched is None:
            representatives.append(answer)
            counts.append(1)
        else:
            counts[matched] += 1
    best = max(counts)
    # Stable tie break: first sampled equivalence class.
    return representatives[counts.index(best)]


class SingleAgentExecutor:
    """One proposer trajectory; compatible with the baseline PPO trainer."""

    def __init__(self, model, tokenizer, config, vllm_engine=None, eval_mode=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = dict(config or {})
        self.vllm_engine = vllm_engine
        self.eval_mode = eval_mode
        self.temperature = 0.0 if eval_mode else float(self.cfg.get("temperature", 1.0))
        self.n_prompt_clipped = self.n_self_target = 0
        self.n_credit_split_failed = self.n_credit_boundary_tokens = 0
        self.n_credit_decode_fallback = self.n_logprob_mismatch = 0
        self.n_gate_unlocked = 0
        self.n_credit_split_failures = {}
        self.n_primary_format = {}
        self.n_hop_depth = {}

    def run_episodes_batch(self, questions, correct_answers, eps_force=0.0):
        if self.vllm_engine is None:
            raise RuntimeError("single-agent baseline requires vLLM")
        prompts_ = [render(self.tokenizer, prompts.SINGLE_SYSTEM,
                           prompts.single_user(q)) for q in questions]
        outputs = self.vllm_engine.generate_batch([
            request("proposer", p, self.temperature) for p in prompts_])
        episodes = []
        for prompt, output, gold in zip(prompts_, outputs, correct_answers):
            text, lps, ids = unpack(output)
            _reasoning, answer = parse_reasoning(text)
            correct = math_equal(answer, gold)
            msg = message("proposer", prompt, text, lps, ids)
            episodes.append({
                "messages": [msg], "turn_ids": [0] * len(ids),
                "log_probs": lps,
                "raca_turn_data": {0: {"role": "proposer", "round": 0,
                    "sigma": "single", "is_response": False,
                    "reward": float(correct), "r_prop": float(correct)}},
                "raca_round_meta": [], "baseline_gold": gold,
                "final_answer": answer, "is_correct": correct,
                "is_correct_uniform": correct,
                "is_correct_weighted": correct, "stopped": True,
                "n_votes": 1, "n_distinct": int(bool(answer)),
                "vote_margin": 1.0 if answer else 0.0,
            })
        return episodes

    def run_episode(self, question, correct_answer):
        return self.run_episodes_batch([question], [correct_answer])[0]

    def absorb_counters(self, _others):
        return None


class SystemEvaluator:
    METHODS = {"sft_cot", "self_consistency", "self_refine", "fixed_four_role"}

    def __init__(self, tokenizer, engine, method, n_samples=8, temperature=0.0):
        if method not in self.METHODS:
            raise ValueError("unknown system baseline %r" % method)
        self.tokenizer = tokenizer
        self.engine = engine
        self.method = method
        self.n_samples = max(1, int(n_samples))
        self.temperature = float(temperature)

    def _generate(self, role, systems, users, temperature=None):
        ps = [render(self.tokenizer, s, u) for s, u in zip(systems, users)]
        temp = self.temperature if temperature is None else temperature
        outs = self.engine.generate_batch([request(role, p, temp) for p in ps])
        rows = []
        for p, out in zip(ps, outs):
            text, lps, ids = unpack(out)
            rows.append({"role": role, "prompt": p, "text": text,
                         "log_probs": lps, "token_ids": ids})
        return rows

    def run_batch(self, questions, golds):
        if self.method == "sft_cot":
            return self._single(questions, golds)
        if self.method == "self_consistency":
            return self._self_consistency(questions, golds)
        if self.method == "self_refine":
            return self._self_refine(questions, golds)
        return self._fixed_four_role(questions, golds)

    def _episode(self, gold, answer, calls, extra=None):
        row = {"final_answer": answer, "is_correct": math_equal(answer, gold),
               "calls": calls, "n_calls": len(calls),
               "generated_tokens": sum(len(x["token_ids"]) for x in calls)}
        row["prompt_tokens"] = sum(len(self.tokenizer.encode(
            x["prompt"], add_special_tokens=False)) for x in calls)
        if extra:
            row.update(extra)
        return row

    def _single(self, questions, golds):
        rows = self._generate("proposer", [prompts.SINGLE_SYSTEM] * len(questions),
                              [prompts.single_user(q) for q in questions], 0.0)
        return [self._episode(g, parse_reasoning(r["text"])[1], [r])
                for g, r in zip(golds, rows)]

    def _self_consistency(self, questions, golds):
        expanded_q = [q for q in questions for _ in range(self.n_samples)]
        rows = self._generate("proposer", [prompts.SINGLE_SYSTEM] * len(expanded_q),
                              [prompts.single_user(q) for q in expanded_q],
                              max(self.temperature, 0.7))
        result = []
        for i, gold in enumerate(golds):
            calls = rows[i * self.n_samples:(i + 1) * self.n_samples]
            answers = [parse_reasoning(x["text"])[1] for x in calls]
            result.append(self._episode(gold, majority_answer(answers), calls,
                                        {"candidate_answers": answers}))
        return result

    def _self_refine(self, questions, golds):
        initial = self._generate("proposer", [prompts.SINGLE_SYSTEM] * len(questions),
                                 [prompts.single_user(q) for q in questions], 0.0)
        feedback = self._generate(
            "proposer", [prompts.FEEDBACK_SYSTEM] * len(questions),
            [prompts.feedback_user(q, r["text"]) for q, r in zip(questions, initial)], 0.0)
        refined = self._generate(
            "proposer", [prompts.REFINE_SYSTEM] * len(questions),
            [prompts.refine_user(q, a["text"], f["text"])
             for q, a, f in zip(questions, initial, feedback)], 0.0)
        return [self._episode(g, parse_reasoning(r["text"])[1], [a, f, r],
                              {"initial_answer": parse_reasoning(a["text"])[1]})
                for g, a, f, r in zip(golds, initial, feedback, refined)]

    def _fixed_four_role(self, questions, golds):
        initial = self._generate("proposer", [prompts.SINGLE_SYSTEM] * len(questions),
                                 [prompts.single_user(q) for q in questions], 0.0)
        coordination = self._generate(
            "controller", [prompts.COORDINATOR_SYSTEM] * len(questions),
            [prompts.coordinator_user(q, a["text"])
             for q, a in zip(questions, initial)], 0.0)
        critique = self._generate(
            "critic", [prompts.CRITIC_SYSTEM] * len(questions),
            [prompts.critic_user(q, a["text"]) for q, a in zip(questions, initial)], 0.0)
        corrected = self._generate(
            "proposer", [prompts.CORRECTION_SYSTEM] * len(questions),
            [prompts.correction_user(q, a["text"], c["text"])
             for q, a, c in zip(questions, initial, critique)], 0.0)
        verified = self._generate(
            "verifier", [prompts.VERIFIER_SYSTEM] * len(questions),
            [prompts.verifier_user(q, c["text"])
             for q, c in zip(questions, corrected)], 0.0)
        result = []
        for gold, a, ctrl, c, r, v in zip(
                golds, initial, coordination, critique, corrected, verified):
            initial_answer = parse_reasoning(a["text"])[1]
            corrected_answer = parse_reasoning(r["text"])[1]
            answer = corrected_answer or initial_answer
            result.append(self._episode(gold, answer, [a, ctrl, c, r, v], {
                "initial_answer": initial_answer,
                "verifier_score": parse_score(v["text"]),
            }))
        return result

class FixedPipelineExecutor:
    """Training-compatible fixed P->Controller->Critic->P-revise->Verifier pipeline."""

    def __init__(self, model, tokenizer, config, vllm_engine=None, eval_mode=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = dict(config or {})
        self.vllm_engine = vllm_engine
        self.eval_mode = eval_mode
        self.temperature = 0.0 if eval_mode else float(self.cfg.get("temperature", 1.0))
        self.n_prompt_clipped = self.n_self_target = 0
        self.n_credit_split_failed = self.n_credit_boundary_tokens = 0
        self.n_credit_decode_fallback = self.n_logprob_mismatch = 0
        self.n_gate_unlocked = 0
        self.n_credit_split_failures = {}
        self.n_primary_format = {}
        self.n_hop_depth = {}

    def _stage(self, role, systems, users):
        prompts_ = [render(self.tokenizer, s, u) for s, u in zip(systems, users)]
        outputs = self.vllm_engine.generate_batch([
            request(role, p, self.temperature) for p in prompts_])
        rows = []
        for prompt, output in zip(prompts_, outputs):
            text, lps, ids = unpack(output)
            rows.append({"role": role, "prompt": prompt, "text": text,
                         "log_probs": lps, "token_ids": ids})
        return rows

    def run_episodes_batch(self, questions, correct_answers, eps_force=0.0):
        if self.vllm_engine is None:
            raise RuntimeError("fixed-pipeline baseline requires vLLM")
        n = len(questions)
        initial = self._stage("proposer", [prompts.SINGLE_SYSTEM] * n,
                              [prompts.single_user(q) for q in questions])
        controllers = self._stage(
            "controller", [prompts.COORDINATOR_SYSTEM] * n,
            [prompts.coordinator_user(q, a["text"])
             for q, a in zip(questions, initial)])
        critiques = self._stage(
            "critic", [prompts.CRITIC_SYSTEM] * n,
            [prompts.critic_user(q, a["text"])
             for q, a in zip(questions, initial)])
        corrections = self._stage(
            "proposer", [prompts.CORRECTION_SYSTEM] * n,
            [prompts.correction_user(q, a["text"], c["text"])
             for q, a, c in zip(questions, initial, critiques)])
        verifiers = self._stage(
            "verifier", [prompts.VERIFIER_SYSTEM] * n,
            [prompts.verifier_user(q, c["text"])
             for q, c in zip(questions, corrections)])

        episodes = []
        for gold, stages in zip(correct_answers, zip(
                initial, controllers, critiques, corrections, verifiers)):
            messages = []
            flat_lps = []
            flat_tids = []
            turn_data = {}
            for tid, row in enumerate(stages):
                role = row["role"]
                ids = row["token_ids"]
                messages.append({"role_name": role, "turn_id": tid,
                    "prompt_text": row["prompt"], "response": row["text"],
                    "response_ids": ids, "logprob_aligned":
                        len(ids) == len(row["log_probs"])})
                flat_lps.extend(row["log_probs"])
                flat_tids.extend([tid] * len(ids))
                turn_data[tid] = {"role": role, "round": 0,
                                  "sigma": "fixed", "is_response": tid > 0,
                                  "reward": 0.0}
            initial_answer = parse_reasoning(stages[0]["text"])[1]
            corrected_answer = parse_reasoning(stages[3]["text"])[1]
            answer = corrected_answer or initial_answer
            correct = math_equal(answer, gold)
            episodes.append({"messages": messages, "turn_ids": flat_tids,
                "log_probs": flat_lps, "raca_turn_data": turn_data,
                "raca_round_meta": [], "baseline_gold": gold,
                "final_answer": answer, "is_correct": correct,
                "is_correct_uniform": correct, "is_correct_weighted": correct,
                "stopped": True, "n_votes": 1,
                "n_distinct": int(bool(answer)),
                "vote_margin": 1.0 if answer else 0.0})
        return episodes

    def run_episode(self, question, correct_answer):
        return self.run_episodes_batch([question], [correct_answer])[0]

    def absorb_counters(self, _others):
        return None
