"""低层多agent环境"""
import numpy as np
from typing import Dict, Any, Tuple
from envs.blackboard import Blackboard, Message, MessageType
from agents.agent_executor import AgentExecutor


class LowLevelEnv:
    """低层三agent环境"""

    def __init__(self, config: Dict[str, Any], llm_interface):
        self.config = config
        self.llm = llm_interface
        self.executor = AgentExecutor(llm_interface, config)
        self.k_max = {"light": 3, "standard": 5, "heavy": 8}
        self.current_step = 0
        self.current_goal = None
        self.current_budget = None
        self.current_question = None
        self.blackboard = None

    def reset(self, goal: str, budget: str, question: str, blackboard: Blackboard) -> Dict[str, np.ndarray]:
        """重置低层环境"""
        self.current_step = 0
        self.current_goal = goal
        self.current_budget = budget
        self.current_question = question
        self.blackboard = blackboard
        return self._get_observations()

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取三个agent的观测"""
        # 简化版本: 每个agent观测512维
        base_obs = np.random.randn(512)
        return {
            "proposer": base_obs,
            "critic": base_obs,
            "verifier": base_obs,
        }

    def step(self, actions: Dict[str, Tuple]) -> Tuple[Dict, Dict, bool, Dict]:
        """执行低层动作
        Args:
            actions: {"proposer": (work, comm, target), ...}
        Returns:
            obs_dict, reward_dict, done, info
        """
        self.current_step += 1

        # 执行各agent动作
        for agent_id, (work_action, comm_action, target) in actions.items():
            self._execute_action(agent_id, work_action, comm_action, target)

        # 检查是否结束
        max_steps = self.k_max[self.current_budget]
        done = self.current_step >= max_steps

        obs_dict = self._get_observations()
        reward_dict = {"proposer": 0.0, "critic": 0.0, "verifier": 0.0}
        info = {"step": self.current_step}

        return obs_dict, reward_dict, done, info

    def _execute_action(self, agent_id: str, work_action: str,
                       comm_action: str, target: int):
        """执行单个agent的动作"""
        # Proposer动作
        if agent_id == "proposer":
            result = self.executor.execute_proposer(work_action, self.current_question, self.blackboard)
            if result and comm_action == "submit-trace":
                reasoning, answer = result
                msg = Message(sender=0, msg_type=MessageType.TRACE, content=(reasoning, answer))
                self.blackboard.add_message(msg)

        # Critic动作
        elif agent_id == "critic":
            result = self.executor.execute_critic(work_action, self.current_question, self.blackboard)
            if result and comm_action == "submit-flaw":
                msg = Message(sender=1, msg_type=MessageType.FLAW, content=result)
                self.blackboard.add_message(msg)

        # Verifier动作
        elif agent_id == "verifier":
            result = self.executor.execute_verifier(work_action, self.current_question, self.blackboard)
            if result and comm_action == "submit-score":
                answer, score = result
                msg = Message(sender=2, msg_type=MessageType.SCORE, content=(answer, score))
                self.blackboard.add_message(msg)
