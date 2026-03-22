"""高层SMDP环境"""
import numpy as np
from typing import Dict, Any, Tuple


class HighLevelEnv:
    """高层Controller环境"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_rounds = config.get("max_rounds", 8)
        self.current_round = 0
        self.question = None
        self.blackboard = None
        self.total_cost = 0

    def reset(self, question: str, blackboard) -> np.ndarray:
        """重置环境"""
        self.current_round = 0
        self.question = question
        self.blackboard = blackboard
        self.total_cost = 0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """构建观测
        obs_dim = 15 (统计特征) + 256 (问题embedding)
        """
        # 统计特征 (15维)
        num_traces = len(self.blackboard.traces)
        num_distinct = len(self.blackboard.get_distinct_answers())
        num_verified = len(self.blackboard.scores)
        num_flaws = len(self.blackboard.flaws)

        stats = np.array([
            self.current_round / self.max_rounds,  # 归一化轮次
            num_traces / 20.0,  # 归一化轨迹数
            num_distinct / 10.0,  # 归一化答案数
            num_verified / 10.0,
            num_flaws / 10.0,
            self.total_cost / 4000.0,  # 归一化成本
            # 其他统计特征...
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ])

        # 问题embedding (256维) - 简化版本
        question_emb = np.random.randn(256)  # TODO: 使用真实embedding

        return np.concatenate([stats, question_emb])

    def step(self, action: Tuple[str, str, str]) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行高层动作
        Args:
            action: (goal, focus, budget)
        Returns:
            obs, reward, done, info
        """
        goal, focus, budget = action
        self.current_round += 1

        # STOP动作
        if goal == "STOP":
            final_answer = self._aggregate_answer(focus)
            correct = self._check_answer(final_answer)
            reward = 1.0 if correct else 0.0
            done = True
            info = {"correct": correct, "answer": final_answer}
            return self._get_observation(), reward, done, info

        # 其他动作继续
        done = self.current_round >= self.max_rounds
        reward = 0.0  # 中间奖励
        info = {"goal": goal, "focus": focus, "budget": budget}

        return self._get_observation(), reward, done, info

    def _aggregate_answer(self, method: str) -> str:
        """聚合答案"""
        if method == "majority":
            answers = [ans for _, ans in self.blackboard.traces]
            return max(set(answers), key=answers.count) if answers else ""
        return ""

    def _check_answer(self, answer: str) -> bool:
        """检查答案正确性"""
        # TODO: 实现真实的答案检查
        return False
