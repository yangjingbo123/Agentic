"""Prompt模板 - 各类推理任务的提示词"""


class PromptTemplates:
    """Prompt模板集合"""

    @staticmethod
    def generate(question: str) -> str:
        """生成推理链"""
        return f"""请解决以下数学问题，给出详细的推理步骤。

问题：{question}

请按以下格式回答：
推理过程：[详细的推理步骤]
最终答案：[数值答案]"""

    @staticmethod
    def generate_diverse(question: str, existing_answers: list) -> str:
        """生成多样化推理链"""
        existing = ", ".join(existing_answers) if existing_answers else "无"
        return f"""请解决以下数学问题，尝试使用不同的方法。

问题：{question}
已有答案：{existing}

请尝试不同的解题思路，给出新的推理过程。
推理过程：[详细的推理步骤]
最终答案：[数值答案]"""

    @staticmethod
    def refine(question: str, draft_reasoning: str) -> str:
        """改进推理链"""
        return f"""请改进以下推理过程，使其更清晰准确。

问题：{question}
初稿推理：{draft_reasoning}

请给出改进后的推理过程：
推理过程：[改进后的推理步骤]
最终答案：[数值答案]"""

    @staticmethod
    def critique_logic(question: str, reasoning: str, answer: str) -> str:
        """批判推理逻辑"""
        return f"""请检查以下推理过程是否存在逻辑错误。

问题：{question}
推理过程：{reasoning}
答案：{answer}

请指出可能的逻辑错误或不严谨之处。如果没有问题，回答"无错误"。"""

    @staticmethod
    def find_counterexample(question: str, reasoning: str, answer: str) -> str:
        """寻找反例"""
        return f"""请尝试找出以下推理的反例或错误。

问题：{question}
推理过程：{reasoning}
答案：{answer}

如果能找到反例或错误，请说明。如果推理正确，回答"无反例"。"""

    @staticmethod
    def quick_verify(question: str, answer: str) -> str:
        """快速验证"""
        return f"""请快速验证答案是否合理。

问题：{question}
答案：{answer}

请给出0-1之间的置信度分数，并简要说明理由。
格式：分数: [0-1的数值]"""

    @staticmethod
    def step_verify(question: str, reasoning: str, answer: str) -> str:
        """逐步验证"""
        return f"""请逐步验证以下推理过程。

问题：{question}
推理过程：{reasoning}
答案：{answer}

请检查每一步是否正确，给出0-1之间的置信度分数。
格式：分数: [0-1的数值]"""
