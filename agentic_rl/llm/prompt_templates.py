from agents.parsing import MAX_CHANNEL_CHARS, clip_text


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
       2030 个 turn 全部如此，min=max=68）。而 `Blackboard.to_text` 的
       `发现问题：` 行当时只给 80 字窗口、且存的是 critic 的**原始输出**（含
       块），块在开头时这 80 字里只剩 12 字装实质内容。分两种情形看（667 个
       critic turn）：65% 的输出是「错误分析：无错误」这类 8 字短句，窗口本来
       就不吃紧；但**真正报了错的 218 个 turn，进信道的那个串（剥块后整段）
       中位 267 字、p75 341、p90 508**，旧布局只送达 11 字。这条 `发现问题`
       行会进 critic / verifier / 修正 / controller 每一个 prompt——也就是说
       过去 critic 报的错传到下游基本只剩「有问题」三个字。
       （注：曾有一版注释写「234 个 turn 中位 254 字」，两把尺子都量不出这两个
       数——「仅『错误分析』段」这把尺量得 n=218 中位 262 p90 503，与整段那把
       差别很小；该串已作废，以本段为准。窗口内真正决定送达量的是**整段**，
       因为黑板存的就是整段。）

       M1 之后另两步已落地，此处一并记：块在**展示副本**上被显式剥离
       （`strip_interaction`，`5d4bac7`），窗口由 80 抬到
       `MAX_CHANNEL_CHARS = 300`（`envs/blackboard.py`）。300 仍有 37%
       （80/218）被截，故第四轮改为**带可见标记**截断（`clip_text`），
       让接收方能分辨「说完了」与「被砍了」；继续加大窗口是另一回事。

    注：`parse_interaction` 全文搜索块，不依赖位置；但 `parse_reasoning` 的
    `最终答案：` 正则原先缺块开标签的 lookahead，已同步修正（见 parsing.py）。
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
        """拼在响应方标准 user prompt 末尾的请求上下文（响应方仍按标准格式输出）。

        `对方内容` 走 `clip_text` 而非硬切片：响应方要**基于这段内容**给判断，
        分不清"对方说完了"和"被砍了"就会对半句话下结论。窗口与黑板 flaw 同源。
        """
        return (
            f"\n协作者（{initiator}）发起了交互：{action}，理由：{reason}\n"
            f"对方内容：{clip_text(initiator_output, MAX_CHANNEL_CHARS)}\n"
            f"请按你的标准输出格式回应。"
        )

    @staticmethod
    def proposer_correction_user(question: str, initiator: str,
                                 initiator_output: str, blackboard_text: str):
        """proposer 被要求修正解法时的 user prompt（输出标准 proposer 格式）。

        同 `request_context`：被指出的问题若在第 300 字处断掉而不留痕迹，
        proposer 会把半句批评当成完整意见去改，这是 v3 `flip/corr` 低的一个
        可疑来源（未证实，但至少先别自己制造这个歧义）。
        """
        return (
            f"问题：{question}\n"
            f"你之前的解法被 {initiator} 指出问题："
            f"{clip_text(initiator_output, MAX_CHANNEL_CHARS)}\n"
            f"当前状态：{blackboard_text}\n"
            f"请修正解法，给出完整推理过程与最终答案。"
        )
