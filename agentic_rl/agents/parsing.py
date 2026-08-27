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


# 无界文本上限（v3.1）。v3 实测：parse 0.95→0.80 后，兜底抽取的 answer 可能
# 是整段文本，reasoning 缺失时更是直接返回全部输出；两者都会进 responder
# prompt 与黑板文本，随轮数累积后撑破 max_model_len（step 151 实测 5036>4096）。
# 另：answer 是投票池的键，长文本 answer 彼此各不相同→各占一票稀释投票。
MAX_ANSWER_CHARS = 64      # 数学答案（含 LaTeX）远不到此长度；超出即视为解析失败
MAX_REASONING_CHARS = 1500  # 完整推理保留上限（够容纳正常多步解题）


def parse_reasoning(text: str) -> tuple[str, str]:
    """proposer 输出 → (推理过程, 最终答案)。两路返回值均有硬上限。

    v3.2（M1）：`最终答案：` 的正则加了 `(?=<|\\n|$)` 前瞻。M1 把
    `<interaction>` 块从开头移到了**答案之后**，若模型把块写在同一行
    （`最终答案：4<interaction>...`），原来的 `(.+)` 会把块吃进答案——长度常
    不足 64 字符，于是这个带标签的垃圾串会被当成合法答案进投票池，各自占一票。
    三个分支缺一不可：`\\n` 覆盖块换行写（最常见），`<` 覆盖同行写，`$` 覆盖
    答案就是输出末尾。注意光写 `$` 不够：未开 re.M 时 `$` 只匹配串尾，
    非贪婪的 `.+?` 又跨不过换行，会导致整个匹配失败。
    """
    reasoning = re.search(r"推理过程：(.+?)(?=最终答案：|<|$)", text, re.S)
    answer = re.search(r"最终答案：(.+?)(?=<|\n|$)", text)
    if not answer:
        nums = re.findall(r"-?\d+\.?\d*", text)
        ans_str = nums[-1] if nums else ""
    else:
        ans_str = answer.group(1).strip()
        # 「最终答案：」后又接着展开论述 → 不是答案，当解析失败处理，
        # 而非截断后当真（截断会造出一个“看上去像答案”的假票污染投票池）。
        if len(ans_str) > MAX_ANSWER_CHARS:
            nums = re.findall(r"-?\d+\.?\d*", ans_str)
            ans_str = nums[-1] if nums else ""
    reasoning_str = reasoning.group(1).strip() if reasoning else text
    return (reasoning_str[:MAX_REASONING_CHARS], ans_str)


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
