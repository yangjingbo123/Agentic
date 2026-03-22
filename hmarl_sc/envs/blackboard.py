"""共享信息板实现 - agent间通信的核心组件"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MessageType(Enum):
    """消息类型"""
    TRACE = "trace"              # 推理链
    SCORE = "score"              # 验证分数
    FLAW = "flaw"                # 发现的缺陷
    REQUEST = "request"          # 请求
    CHALLENGE = "challenge"      # 质疑
    ENDORSE = "endorse"          # 支持


@dataclass
class Message:
    """信息板消息"""
    sender: int                  # 发送者ID (0=proposer, 1=critic, 2=verifier)
    msg_type: MessageType
    content: Any
    target: Optional[int] = None # 目标agent ID (None表示广播)
    step: int = 0                # 微步编号


class Blackboard:
    """共享信息板"""

    def __init__(self):
        self.messages: List[Message] = []
        self.traces: List[Tuple[str, str]] = []  # (reasoning_chain, answer)
        self.scores: Dict[str, List[float]] = {}  # answer -> [scores]
        self.flaws: List[Dict] = []

    def add_message(self, msg: Message):
        """添加消息"""
        self.messages.append(msg)

        # 更新对应的数据结构
        if msg.msg_type == MessageType.TRACE:
            self.traces.append(msg.content)
        elif msg.msg_type == MessageType.SCORE:
            answer, score = msg.content
            if answer not in self.scores:
                self.scores[answer] = []
            self.scores[answer].append(score)
        elif msg.msg_type == MessageType.FLAW:
            self.flaws.append(msg.content)

    def get_messages(self, msg_type: Optional[MessageType] = None,
                     target: Optional[int] = None) -> List[Message]:
        """获取消息"""
        msgs = self.messages
        if msg_type:
            msgs = [m for m in msgs if m.msg_type == msg_type]
        if target is not None:
            msgs = [m for m in msgs if m.target is None or m.target == target]
        return msgs

    def get_distinct_answers(self) -> List[str]:
        """获取所有不同的答案"""
        return list(set(ans for _, ans in self.traces))

    def get_answer_count(self, answer: str) -> int:
        """获取某答案出现次数"""
        return sum(1 for _, ans in self.traces if ans == answer)

    def get_avg_score(self, answer: str) -> float:
        """获取某答案的平均验证分数"""
        if answer not in self.scores or not self.scores[answer]:
            return 0.0
        return sum(self.scores[answer]) / len(self.scores[answer])

    def clear(self):
        """清空信息板"""
        self.messages.clear()
        self.traces.clear()
        self.scores.clear()
        self.flaws.clear()
