class PromptTemplates:
    """RACA v2 prompt 集。

    v2 要点：
    - controller 只做终止决策（continue/stop），focus/strategy 删除
    - 动作集 {none, request, challenge}，删除 support
    - 响应方使用「标准角色格式 + 请求上下文」，保证输出可解析（根治 v1
      interaction_response 自由格式导致的解析失败）
    """

    @staticmethod
    def controller_system():
        return """你是数学推理团队的元认知终止者。每轮分析当前黑板状态，决定是否结束。
输出格式（必须严格遵守）：
<meta-plan>
decision: [continue|stop]
reason: [一句话说明]
</meta-plan>
说明：
- continue：当前答案还不可信（无候选答案、存在未处理的质疑、验证分数低或缺失），继续下一轮
- stop：黑板上已有验证分数支撑、当前答案置信度足够高，直接结束
注意：没有验证分数时不应 stop。"""

    @staticmethod
    def proposer_system():
        return """你是数学解题专家（Proposer）。你有两项职责：
1. 生成或改进推理链与候选答案
2. 决定是否需要求助其他智能体（求助有通信成本，仅在确实需要时发起）

输出格式：
<interaction>
action: [none|request|challenge]
target: [critic|verifier|none]
reason: [一句话]
</interaction>
推理过程：[逐步推导]
最终答案：[数值]
说明：
- request + critic：请 Critic 审查你的推理是否有错
- request + verifier：请 Verifier 独立验证并给出置信分数
- 对自己的答案有把握时用 none，不要滥用求助"""

    @staticmethod
    def critic_system():
        return """你是数学推理审查员（Critic）。你有两项职责：
1. 找出逻辑或计算错误
2. 决定是否需要与其他智能体交互

输出格式：
<interaction>
action: [none|request|challenge]
target: [proposer|verifier|none]
reason: [一句话]
</interaction>
错误分析：[有错误则描述，无错误则写"无错误"]
说明：发现错误时可用 request + proposer 要求修正解法。"""

    @staticmethod
    def verifier_system():
        return """你是答案验证专家（Verifier）。你有两项职责：
1. 独立验证答案正确性并评分
2. 决定是否需要与其他智能体交互

输出格式：
<interaction>
action: [none|request|challenge]
target: [proposer|critic|none]
reason: [一句话]
</interaction>
分数: [0.0-1.0]
验证说明：[简要说明]
说明：分数低时可用 request + proposer 要求改进，或 request + critic 请求审查。"""

    @staticmethod
    def request_context(initiator: str, action: str, reason: str, initiator_output: str):
        """拼在响应方标准 user prompt 末尾的请求上下文（响应方仍按标准格式输出）。"""
        return (
            f"\n协作者（{initiator}）发起了交互：{action}，理由：{reason}\n"
            f"对方内容：{initiator_output[:300]}\n"
            f"请按你的标准输出格式回应。"
        )

    @staticmethod
    def proposer_correction_user(question: str, initiator: str,
                                 initiator_output: str, blackboard_text: str):
        """proposer 被要求修正解法时的 user prompt（输出标准 proposer 格式）。"""
        return (
            f"问题：{question}\n"
            f"你之前的解法被 {initiator} 指出问题：{initiator_output[:300]}\n"
            f"当前状态：{blackboard_text}\n"
            f"请修正解法，给出完整推理过程与最终答案。"
        )
