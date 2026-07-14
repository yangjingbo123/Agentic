"""
Reward function for the 4-role blackboard system in the veRL pipeline.

Called by rema.ReMARewardManager as:
    compute_score(data_source, response, ground_truth, extra_info) -> float

Returns a binary accuracy score (0.0 or 1.0).
Per-role turn-level rewards and shaped rewards are computed by rema.py itself.
"""

import re


def _normalize(s: str) -> str:
    return re.sub(r"[^0-9.\-/]", "", s.strip())


def compute_reward(data_source: str, response: str, ground_truth: str, extra_info=None) -> float:
    """
    Binary math correctness scorer.
    Returns 1.0 if response matches ground_truth, else 0.0.
    """
    pred = _normalize(str(response))
    gt   = _normalize(str(ground_truth))
    if not pred or not gt:
        return 0.0
    return 1.0 if pred == gt else 0.0
