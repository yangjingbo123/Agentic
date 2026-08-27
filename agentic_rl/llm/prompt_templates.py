class PromptTemplates:
    """RACA v2 prompt 集。

    v2 要点：
    - controller 只做终止决策（continue/stop），focus/strategy 删除
    - 动作集 {none, request, challenge}，删除 support
    - 响应方使用「标准角色格式 + 请求上下文」，保证输出可解析（根治 v1
      interaction_response 自由格式导致的解析失败）

    v3.2（M1）：`<interaction>` 块从**开头移到末尾**，三个角色一致。两个理由，
    第一个是决定性的：

    1. **Eq (12) 的良定义性。** $r^{\\rm int}$ 定义在 $(p^{\\rm primary},
       p^{\\rm end})$ 矩阵上——这个形式预设了「是否求助」的决策是在**已知自己
       答案**时做出的。块在开头时决策先于答案生成，等于用策略尚未产生的量去给
       它的动作定价，`sel = corr(u, p_primary) ≈ 0` 是这个结构的必然结果，不是
       调参问题。这条与交互能否见效无关，属于必须修。

    2. **顺带打开一条被挤掉的信道（实测）。** 块长度固定 68 字符（v2+v3 共
       2030 个 turn 全部如此，min=max=68）。而 `Blackboard.to_text` 里
       `发现问题：{flaws[-1]['content'][:80]}` 存的是 critic 的**原始输出**，
       块在开头时这 80 字窗口只剩 12 字装实质内容。分两种情形看（667 个
       critic turn）：65% 的输出是「错误分析：无错误」这类 8 字短句，窗口本来
       就不吃紧；但**真正报了错的 234 个 turn 分析长度中位数 254 字，旧布局只
       送达 11 字，新布局送达 80 字（7.3×）**。这条 `发现问题` 行会进 critic /
       verifier / 修正 / controller 每一个 prompt——也就是说过去 critic 报的错
       传到下游基本只剩「有问题」三个字。`request_context` 与
       `proposer_correction_user` 的 300 字窗口同理由 231 → 254。
       （更彻底的做法是存黑板前显式剥离块，那样短分析也不会混进块尾；留作后续
       独立一步，以免与 M1 混淆 `eff` / `flip` 的归因。）

    注：`parse_interaction` 全文搜索块，不依赖位置；但 `parse_reasoning` 的
    `最终答案：` 正则原先缺 `<` 的 lookahead，已同步修正（见 parsing.py）。
    SFT 数据的 `system` 字段与本文件逐字节相同，改这里必须同步重新生成，
    否则 SFT 与 RL 的 prompt 会漂移。
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
推理过程：[逐步推导]
最终答案：[数值]
<interaction>
action: [none|request|challenge]
target: [critic|verifier|none]
reason: [一句话]
</interaction>
说明：
- 必须先完成推理并写出最终答案，再据此决定是否求助——求助的依据是你对这个答案的把握
- request + critic：请 Critic 审查你的推理是否有错
- request + verifier：请 Verifier 独立验证并给出置信分数
- 对自己的答案有把握时用 none，不要滥用求助"""

    @staticmethod
    def critic_system():
        return """你是数学推理审查员（Critic）。你有两项职责：
1. 找出逻辑或计算错误
2. 决定是否需要与其他智能体交互

输出格式：
错误分析：[有错误则描述，无错误则写"无错误"]
<interaction>
action: [none|request|challenge]
target: [proposer|verifier|none]
reason: [一句话]
</interaction>
说明：先写完错误分析，再据此决定是否交互。发现错误时可用 request + proposer 要求修正解法。"""

    @staticmethod
    def verifier_system():
        return """你是答案验证专家（Verifier）。你有两项职责：
1. 独立验证答案正确性并评分
2. 决定是否需要与其他智能体交互

输出格式：
分数: [0.0-1.0]
验证说明：[简要说明]
<interaction>
action: [none|request|challenge]
target: [proposer|critic|none]
reason: [一句话]
</interaction>
说明：先打分再据此决定是否交互。分数低时可用 request + proposer 要求改进，或 request + critic 请求审查。"""

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
