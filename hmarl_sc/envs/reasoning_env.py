"""主环境 - 整合高层和低层"""
from envs.high_level_env import HighLevelEnv
from envs.low_level_env import LowLevelEnv
from envs.blackboard import Blackboard
from llm.llm_interface import LLMInterface
from typing import Dict, Any


class ReasoningEnv:
    """HMARL-SC主环境"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 初始化LLM
        self.llm = LLMInterface(config.get("llm", {}))

        # 初始化高层和低层环境
        self.high_env = HighLevelEnv(config.get("high_level", {}))
        self.low_env = LowLevelEnv(config.get("low_level", {}), self.llm)

        # 共享信息板
        self.blackboard = Blackboard()

        # 当前层级
        self.current_level = "high"  # "high" or "low"
        self.current_question = None

    def reset(self, question: str):
        """重置环境"""
        self.current_question = question
        self.blackboard.clear()
        self.current_level = "high"

        obs = self.high_env.reset(question, self.blackboard)
        return {"controller": obs}

    def step(self, action_dict: Dict):
        """执行一步
        Args:
            action_dict: {"controller": action} 或 {"proposer": ..., "critic": ..., "verifier": ...}
        """
        if self.current_level == "high":
            return self._step_high(action_dict["controller"])
        else:
            return self._step_low(action_dict)

    def _step_high(self, action):
        """执行高层步骤"""
        obs, reward, done, info = self.high_env.step(action)

        # 如果是STOP,直接返回
        if done:
            return {"controller": obs}, {"controller": reward}, done, info

        # 否则切换到低层
        goal, focus, budget = action
        self.current_level = "low"
        low_obs = self.low_env.reset(goal, budget, self.current_question, self.blackboard)

        return low_obs, {k: 0.0 for k in low_obs}, False, info

    def _step_low(self, action_dict):
        """执行低层步骤"""
        obs_dict, reward_dict, done, info = self.low_env.step(action_dict)

        # 低层结束,切换回高层
        if done:
            self.current_level = "high"
            high_obs = self.high_env._get_observation()
            return {"controller": high_obs}, {"controller": 0.0}, False, info

        return obs_dict, reward_dict, False, info
