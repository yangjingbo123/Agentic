"""Agent执行逻辑 - 将LLM调用集成到agent动作执行"""
import re
from typing import Dict, Any, Optional, Tuple
from llm.prompt_templates import PromptTemplates
from envs.blackboard import Blackboard, Message, MessageType


class AgentExecutor:
    """Agent执行器 - 负责将抽象动作转化为LLM调用"""

    def __init__(self, llm, config: Dict[str, Any]):
        self.llm = llm
        self.config = config
        self.t_explore = config.get("temperature_explore", 0.7)
        self.t_diagnose = config.get("temperature_diagnose", 0.3)
        self.max_tokens = config.get("max_tokens", 512)

    def execute_proposer(self, work_action: str, question: str,
                         blackboard: Blackboard) -> Optional[Tuple[str, str]]:
        """执行Proposer动作,返回(reasoning, answer)或None"""
        if work_action == "generate":
            prompt = PromptTemplates.generate(question)
            output = self.llm.generate(prompt, temperature=self.t_explore,
                                       max_tokens=self.max_tokens)
            return self._parse_reasoning(output)

        elif work_action == "generate-diverse":
            existing = [ans for _, ans in blackboard.traces]
            prompt = PromptTemplates.generate_diverse(question, existing)
            output = self.llm.generate(prompt, temperature=self.t_explore + 0.1,
                                       max_tokens=self.max_tokens)
            return self._parse_reasoning(output)

        elif work_action == "refine":
            if not blackboard.traces:
                return None
            draft_reasoning, _ = blackboard.traces[-1]
            prompt = PromptTemplates.refine(question, draft_reasoning)
            output = self.llm.generate(prompt, temperature=self.t_diagnose,
                                       max_tokens=self.max_tokens)
            return self._parse_reasoning(output)

        return None  # work-idle

    def execute_critic(self, work_action: str, question: str,
                       blackboard: Blackboard) -> Optional[Dict]:
        """执行Critic动作,返回flaw信息或None"""
        if not blackboard.traces:
            return None

        target_reasoning, target_answer = blackboard.traces[-1]

        if work_action == "critique-logic":
            prompt = PromptTemplates.critique_logic(question, target_reasoning, target_answer)
            output = self.llm.generate(prompt, temperature=self.t_diagnose,
                                       max_tokens=256)
            if "无错误" not in output:
                return {"type": "logic", "content": output, "target": target_answer}

        elif work_action == "find-counterexample":
            prompt = PromptTemplates.find_counterexample(question, target_reasoning, target_answer)
            output = self.llm.generate(prompt, temperature=self.t_explore,
                                       max_tokens=256)
            if "无反例" not in output:
                return {"type": "counterexample", "content": output, "target": target_answer}

        return None  # work-idle or no flaw found

    def execute_verifier(self, work_action: str, question: str,
                         blackboard: Blackboard) -> Optional[Tuple[str, float]]:
        """执行Verifier动作,返回(answer, score)或None"""
        if not blackboard.traces:
            return None

        target_reasoning, target_answer = blackboard.traces[-1]

        if work_action == "quick-verify":
            prompt = PromptTemplates.quick_verify(question, target_answer)
            output = self.llm.generate(prompt, temperature=self.t_diagnose,
                                       max_tokens=128)
            score = self._parse_score(output)
            return (target_answer, score)

        elif work_action == "step-verify":
            prompt = PromptTemplates.step_verify(question, target_reasoning, target_answer)
            output = self.llm.generate(prompt, temperature=self.t_diagnose,
                                       max_tokens=256)
            score = self._parse_score(output)
            return (target_answer, score)

        return None  # work-idle

    def _parse_reasoning(self, output: str) -> Tuple[str, str]:
        """从LLM输出中解析推理链和答案"""
        reasoning = ""
        answer = ""

        lines = output.strip().split("\n")
        for line in lines:
            if "推理过程：" in line or "推理过程:" in line:
                reasoning = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif "最终答案：" in line or "最终答案:" in line:
                answer = line.split("：", 1)[-1].split(":", 1)[-1].strip()

        # 如果解析失败,提取数字作为答案
        if not answer:
            numbers = re.findall(r'\d+\.?\d*', output)
            answer = numbers[-1] if numbers else output[-20:].strip()

        if not reasoning:
            reasoning = output

        return reasoning, answer

    def _parse_score(self, output: str) -> float:
        """从LLM输出中解析置信度分数"""
        # 匹配"分数: 0.8"或"分数：0.85"格式
        match = re.search(r'分数[：:]\s*([0-9.]+)', output)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # 匹配任意0-1之间的小数
        numbers = re.findall(r'0\.[0-9]+|1\.0', output)
        if numbers:
            return float(numbers[0])

        return 0.5  # 默认中性分数
