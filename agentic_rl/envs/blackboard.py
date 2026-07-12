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

    def add_message(self, msg: Message):
        if msg.msg_type == MessageType.TRACE:
            self.traces.append(msg.content)
        elif msg.msg_type == MessageType.FLAW:
            self.flaws.append(msg.content)
        elif msg.msg_type == MessageType.SCORE:
            self.scores.append(msg.content)
        elif msg.msg_type == MessageType.INTERACTION:
            self.interactions.append(msg.content)

    def get_distinct_answers(self) -> list[str]:
        return list({ans for _, ans in self.traces})

    def get_avg_score(self, answer: str) -> float:
        vals = [s for a, s in self.scores if a == answer]
        return sum(vals) / len(vals) if vals else 0.0

    def to_text(self) -> str:
        lines = []
        if self.traces:
            lines.append(f"已有{len(self.traces)}个解法，答案：{self.get_distinct_answers()}")
        if self.flaws:
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
