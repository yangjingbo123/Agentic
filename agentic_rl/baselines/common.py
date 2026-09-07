"""Shared, dependency-light helpers for baseline experiments."""

import hashlib
import math
from collections import defaultdict


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mu = sum(float(v) for v in values) / len(values)
    var = sum((float(v) - mu) ** 2 for v in values) / len(values)
    return mu, math.sqrt(var)


def zscores(values, delta=1e-4):
    """Population-standardized values, or ``None`` for a degenerate group."""
    mu, std = mean_std(values)
    if len(values) < 2 or std <= delta:
        return None
    return [(float(v) - mu) / std for v in values]


def trainable_turn_ids(episode):
    """Return generated turn ids in message order, without duplicates."""
    seen = set()
    result = []
    for msg in episode.get("messages", []):
        tid = msg.get("turn_id")
        if tid is None or tid in seen or not msg.get("response_ids", []):
            continue
        seen.add(tid)
        result.append(tid)
    return result


def message_by_turn(episode):
    return {msg.get("turn_id"): msg for msg in episode.get("messages", [])
            if msg.get("turn_id") is not None}


def turn_metadata(episode, tid):
    """Read structural metadata without consuming any RACA reward value."""
    data = episode.get("raca_turn_data", {}).get(tid, {})
    msg = message_by_turn(episode).get(tid, {})
    return {
        "role": data.get("role", msg.get("role_name", "proposer")),
        "round": int(data.get("round", 0)),
        "sigma": data.get("sigma", "unknown"),
        "is_response": bool(data.get("is_response", False)),
    }


def role_occurrences(episode):
    """Map turn id to a per-role occurrence index in generation order."""
    counters = defaultdict(int)
    result = {}
    for msg in episode.get("messages", []):
        tid = msg.get("turn_id")
        if tid is None or tid in result:
            continue
        role = msg.get("role_name", "proposer")
        result[tid] = counters[role]
        counters[role] += 1
    return result


def prompt_digest(message):
    prompt = message.get("prompt_text")
    if prompt is None:
        prompt = "\n".join((message.get("system", ""), message.get("user", "")))
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def broadcast_scalar(episode, value):
    return {tid: float(value) for tid in trainable_turn_ids(episode)}


def aggregate_cost(episodes, tokenizer=None):
    """Reproducible per-problem inference-cost counters."""
    n = max(len(episodes), 1)
    calls = 0
    generated = 0
    prompt_tokens = 0
    turns = 0
    for ep in episodes:
        seen = set()
        for msg in ep.get("messages", []):
            tid = msg.get("turn_id")
            if tid in seen:
                continue
            seen.add(tid)
            calls += 1
            turns += 1
            generated += len(msg.get("response_ids", []))
            if tokenizer is not None:
                prompt = msg.get("prompt_text")
                if prompt is not None:
                    prompt_tokens += len(tokenizer.encode(
                        prompt, add_special_tokens=False))
    return {
        "llm_calls_per_problem": calls / n,
        "agent_turns_per_problem": turns / n,
        "generated_tokens_per_problem": generated / n,
        "prompt_tokens_per_problem": prompt_tokens / n,
    }
