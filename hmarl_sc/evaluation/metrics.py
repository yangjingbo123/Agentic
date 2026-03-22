"""评估指标计算"""
import numpy as np
from typing import List, Dict


def compute_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """计算准确率"""
    correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == gt)
    return correct / len(predictions) if predictions else 0.0


def compute_average_cost(episodes: List[Dict]) -> float:
    """计算平均token成本"""
    total_cost = sum(ep.get("total_cost", 0) for ep in episodes)
    return total_cost / len(episodes) if episodes else 0.0


def compute_k_equiv(episodes: List[Dict], single_cot_cost: int = 400) -> float:
    """计算等价SC采样次数"""
    avg_cost = compute_average_cost(episodes)
    return avg_cost / single_cot_cost


def compute_verification_coverage(episodes: List[Dict]) -> float:
    """计算验证覆盖率"""
    total_answers = 0
    verified_answers = 0

    for ep in episodes:
        distinct_answers = set(trace["answer"] for trace in ep.get("traces", []))
        verified = set(score["answer"] for score in ep.get("scores", []))
        total_answers += len(distinct_answers)
        verified_answers += len(verified)

    return verified_answers / total_answers if total_answers > 0 else 0.0


def compute_metrics(episodes: List[Dict], ground_truth: List[str]) -> Dict:
    """计算所有指标"""
    predictions = [ep.get("final_answer", "") for ep in episodes]

    return {
        "accuracy": compute_accuracy(predictions, ground_truth),
        "avg_cost": compute_average_cost(episodes),
        "k_equiv": compute_k_equiv(episodes),
        "verification_coverage": compute_verification_coverage(episodes),
    }
