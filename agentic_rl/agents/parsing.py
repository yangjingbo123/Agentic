"""RACA v2 模型输出解析器（纯正则，零 torch 依赖，便于 CPU 单测）。

v2 变更点：
- controller 只输出 decision: continue|stop（focus/strategy 删除）
- 动作集 {none, request, challenge}，删除 support（不改变状态的噪声动作）
- action 解析用 split(":")[0]，根治 v1 "support:<答案>" 穿透过滤器的 bug
"""

from __future__ import annotations

import re

ROLE_NAMES = {"proposer": "Proposer", "critic": "Critic", "verifier": "Verifier"}
_VALID_ACTIONS = ("request", "challenge")


def parse_decision(meta_plan: str) -> str:
    """controller 输出 → continue | stop（解析失败保守 continue）。"""
    m = re.search(r"decision:\s*(continue|stop)", meta_plan)
    return m.group(1) if m else "continue"


def parse_interaction(text: str) -> tuple[str, str, str]:
    """<interaction> 块 → (action, target, reason)。

    仅 action ∈ {request, challenge} 且 target 为合法角色时视为发起；
    其余（none/support/解析失败）一律 ("none", "none", "")。
    """
    block = re.search(r"<interaction>(.*?)</interaction>", text, re.S)
    if not block:
        return "none", "none", ""
    content = block.group(1)
    action_m = re.search(r"action:\s*(\S+)", content)
    target_m = re.search(r"target:\s*(\S+)", content)
    reason_m = re.search(r"reason:\s*(.+)", content)
    # "request:critic" / "challenge:<问题>" 等带冒号写法只取动作词本身
    action = (action_m.group(1).split(":")[0].strip().lower() if action_m else "none")
    if action not in _VALID_ACTIONS:
        return "none", "none", ""
    target = (target_m.group(1).strip().lower() if target_m else "none")
    # 容错："action: request:critic" 把 target 写进 action 冒号后
    if target not in ROLE_NAMES and action_m:
        tail = action_m.group(1).split(":", 1)
        if len(tail) == 2 and tail[1].strip().lower() in ROLE_NAMES:
            target = tail[1].strip().lower()
    if target not in ROLE_NAMES:
        return "none", "none", ""
    reason = reason_m.group(1).strip() if reason_m else ""
    return action, target, reason


def parse_reasoning(text: str) -> tuple[str, str]:
    """proposer 输出 → (推理过程, 最终答案)。"""
    reasoning = re.search(r"推理过程：(.+?)(?=最终答案：|<|$)", text, re.S)
    answer = re.search(r"最终答案：(.+)", text)
    if not answer:
        nums = re.findall(r"-?\d+\.?\d*", text)
        ans_str = nums[-1] if nums else ""
    else:
        ans_str = answer.group(1).strip()
    return (reasoning.group(1).strip() if reasoning else text, ans_str)


def parse_score(text: str) -> float | None:
    """verifier 输出 → [0,1] 分数；解析失败返回 None。"""
    m = re.search(r"分数:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text)
    if not m:
        return None
    return max(0.0, min(1.0, float(m.group(1))))


def critic_found_errors(critic_output: str) -> bool:
    """鲁棒判断 critic 是否报了错（沿用 v1.x Fix 3 的解析逻辑）。"""
    if "无错误" in critic_output or "无错" in critic_output:
        return False
    err_match = re.search(r"错误分析[：:]\s*(.+?)(?=<|$)", critic_output, re.S)
    if err_match:
        err_text = err_match.group(1).strip()
        return bool(err_text) and "无错误" not in err_text and "无错" not in err_text
    # 无「错误分析」段（响应未按格式）——保守不判 flag
    return False
