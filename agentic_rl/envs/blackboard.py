from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(Enum):
    TRACE = "trace"
    FLAW = "flaw"
    SCORE = "score"
    INTERACTION = "interaction"


@dataclass
class Message:
    agent_id: int
    msg_type: MessageType
    content: Any


class Blackboard:
    def __init__(self):
        self.traces: list[tuple[str, str]] = []       # (reasoning, answer)
        self.flaws: list[dict] = []                   # {"content": str}
        self.scores: list[tuple[str, float]] = []     # (answer, score)
        self.interactions: list[dict] = []            # {from, action, target, reason}
        # 全局事件序号：用于判断 flag 是否已被后续 trace 处理（RACA v2 机械 σ）
        self._seq = 0
        self._last_trace_seq = -1
        self._last_flaw_seq = -1

    def add_message(self, msg: Message):
        self._seq += 1
        if msg.msg_type == MessageType.TRACE:
            self.traces.append(msg.content)
            self._last_trace_seq = self._seq
        elif msg.msg_type == MessageType.FLAW:
            self.flaws.append(msg.content)
            self._last_flaw_seq = self._seq
        elif msg.msg_type == MessageType.SCORE:
            self.scores.append(msg.content)
        elif msg.msg_type == MessageType.INTERACTION:
            self.interactions.append(msg.content)

    def derive_sigma(self) -> str:
        """RACA v2 机械上下文标签（替代 controller 生成的 strategy）。

        σ = explore  若黑板无 trace
          = refine   若最近一条 critic flag 存在且未被后续 trace 处理
          = verify   若已有候选答案且无未处理的 flag

        确定性推导：跨 rollout、跨训练阶段严格可比，不依赖模型输出解析。
        """
        if not self.traces:
            return "explore"
        if self._last_flaw_seq > self._last_trace_seq:
            return "refine"
        return "verify"

    def get_distinct_answers(self) -> list[str]:
        return list({ans for _, ans in self.traces})

    def get_avg_score(self, answer: str) -> float:
        vals = [s for a, s in self.scores if a == answer]
        return sum(vals) / len(vals) if vals else 0.0

    def to_text(self, include_flaws: bool = True) -> str:
        """include_flaws=False：供下一轮 primary 用（v3 测量：旧错解+批评的
        上下文对重答有锚定伤害，通道②两次测量 Δ=−0.085/−0.055）。"""
        lines = []
        if self.traces:
            lines.append(f"已有{len(self.traces)}个解法，答案：{self.get_distinct_answers()}")
        if self.flaws and include_flaws:
            lines.append(f"发现问题：{self.flaws[-1]['content'][:80]}")
        if self.scores:
            distinct = self.get_distinct_answers()
            if distinct:
                best = max(distinct, key=self.get_avg_score)
                lines.append(f"最高置信答案：{best}（分数{self.get_avg_score(best):.2f}）")
        if self.interactions:
            last = self.interactions[-1]
            lines.append(f"最近交互：{last['from']}→{last['target']}（{last['action']}）")
        return "\n".join(lines) if lines else "尚无信息"
