"""把 v2 SFT 的 `user` 重建成 RL 时真正会出现的形状（response 一字不动）。

## 要修的病

v2 那批 SFT 数据是当初让 API 整段编出来的，**连 `user` 一起编**。于是模型被教着
去读一句人话摘要，而 RL 上线后收到的是 `Blackboard.to_text()` 拼出来的结构化转
储。实测：v2 的 1765 个 turn 里只有 5 个（0.3%）含 `当前状态：`，而 v3 与 RL 侧
都是 100%。差别不是措辞，是**黑板那一整段在 v2 的输入里根本不存在**。

最要紧的是 controller：v3 一条 controller 数据都没有，884 条全在 v2。也就是说
SFT 结束时三个 worker 已经在带黑板的输入上练过了，**controller 一次都没见过黑
板**——它学的是"读一句现成总结→ continue/stop"，而 RL 给它结构化转储。这是
v3 那几条观测（controller 停不准、必须靠 stop_gate、`gate` 10→273、黑板信道看
着没用）的一个候选统一解释。

## 为什么重建是可行的（而不是又一次猜测）

v2 的 323 行**本身就是有序 episode**，且拓扑全部落在 RL 可达的范围内。实测
worker 子序列只有六种：`prop→crit→veri` 195、`prop→veri` 104、
`prop→crit→prop→veri` 16、`prop→veri→crit` 6、`prop→crit` 1、
`prop→veri→prop→veri` 1；**323/323 行的第一个 worker 都是 proposer**，与 RL 的
"proposer 固定起点"一致。四 worker 的那两种看着超了 `max_hops=2`，其实正是
`gate_unlock` 那条**不占 hop 预算**的独立 verifier 批次（`agentic_executor.py`
§3.5），所以依然可达。

因此重建就是：按 turn 顺序把更早 turn 的 response 喂进一块真 `Blackboard`，再用
`agentic_executor.py` 里那几行**同样的字面量**渲染 `user`。

## 一处必须说清的不忠实

v2 的拓扑是「controller → 一个 worker → controller → 下一个 worker → …」，
controller 出现在**每个** worker 之前；RL 的 controller 只在每轮开头跑一次。所以
884 条里有 561 条 controller turn 在 RL 的拓扑里不会出现在那个位置。

**仍然全部保留**，理由是：这里要教的是「黑板状态 → continue/stop」这个映射，而
这 561 条看到的状态（1 条 trace / trace+flaw / trace+score）**都是 RL 真会呈现给
controller 的状态**（第 2、3 轮开头就长这样）。所以每一对 (状态, 决策) 都是有效
监督，不因它在 v2 episode 里的位置而失效。丢掉它们等于把仅有的 884 条 controller
数据砍到 323 条，代价远大于收益。这条是**刻意的取舍，不是疏忽**。

## request_context 的判据

RL 只在 `not forced and target != "proposer"` 时追加请求上下文。v2 没存 forced，
但存了发起方的 `<interaction>` 块：**上一个 worker 声明的 target == 本 turn 的角
色**就是自发请求（追加上下文），否则等价于 ε / 闸门强制注入（不追加）。实测 558
个相邻 worker 对里 347 对匹配，其余多为「声明 none 却来了 verifier」——那在 RL
里正是强制注入的样子，所以不追加恰好是忠实的。

## 顺序要紧

必须跑在 `prepare_sft.py` 的 M1 重排**之后**：黑板喂的是
`parse_reasoning(response)` / `strip_interaction(response)`，而 `parse_reasoning`
的 `最终答案：` 前瞻是按块在尾部设计的。反了会把块吃进 answer。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.parsing import (  # noqa: E402
    ROLE_NAMES,
    critic_found_errors,
    parse_interaction,
    parse_reasoning,
    parse_score,
    strip_interaction,
)
from envs.blackboard import Blackboard, Message, MessageType  # noqa: E402
from llm.prompt_templates import PromptTemplates  # noqa: E402

# 与 configs/agentic/default.yaml 对齐。写死而不读 yaml 是有意的：SFT 的输入形状
# 必须与**将要跑的那次 RL** 一致，让它随手改 yaml 就静默漂移，正是这个文件在修的
# 病。改了配置就得回来改这里，并重新派生数据。
FLAW_IN_PRIMARY = False


class Unreplayable(Exception):
    """该 episode 无法忠实重建（缺发起方上下文等）。整行剔除，不猜。"""


# 黑板段的行首标记。RL 侧四个角色的 user 全都含它（`agentic_executor.py` 的四条
# 字面量各自都拼了 `当前状态：`），所以"缺它"就等价于"这条 prompt 不是 RL 形状"。
_BB_MARK = "当前状态："


def needs_replay(row: dict) -> bool:
    """这一行的 `user` 是否是旧形状（需要重建）。

    判据是**内容**而不是文件名或 turn 数，理由有两条：① 幂等——重放过的行必然
    每个 turn 都含黑板段，再跑一次自动跳过，派生链可以随便重跑；② 以后往
    `--inputs` 里追加新数据文件时不需要回来改开关，形状对的行自然不被碰。

    实测（v2 323 行 / v3 1149 行）：v2 有 318 行**没有任何** turn 含黑板段，另 5
    行混合（只有中间某个 controller 含）；v3 的 1149 行**每个** turn 都含。所以
    `any(缺)` 恰好选中全部 323 行 v2、零行 v3，不需要任何按文件的硬编码。
    用 `any` 而非 `all` 就是为了收进那 5 行混合的——它们同样是手写摘要，只是碰
    巧有一句里出现了这四个字。
    """
    return any(_BB_MARK not in t["user"] for t in row["turns"])


def _pending_after(role: str, resp: str, shown: str, flagged: bool):
    """本 turn 结束后交给下一跳的发起上下文，镜像 executor 的两个 append 点。

    critic 标错 → 机械触发 proposer 修正（写死 target=proposer，不看 critic 自己
    的块）；否则按它自己声明的块走，且要求 `t2 != 本角色`（RL 侧的守卫）。
    返回 None 表示 RL 在此处不会有 pending——下一个 worker 只能是强制注入。
    """
    a2, t2, r2 = parse_interaction(resp)
    if role == "critic" and flagged:
        return (role, shown, "request", "proposer", r2)
    if a2 != "none" and t2 != role:
        return (role, shown, a2, t2, r2)
    return None


def replay_row(row: dict) -> dict:
    """重建一行的全部 `user`。就地改 `row`，`response` / `role_name` 不动。"""
    q = row["question"]
    bb = Blackboard()
    pending = None          # 上一个 worker 留下的发起上下文（或 None = 强制注入）
    seen_worker = False     # 第一个 worker 一定是 proposer primary

    for t in row["turns"]:
        role = t["role_name"]
        resp = t["response"]
        last = bb.traces[-1] if bb.traces else ("", "")

        # ── 建 prompt（严格照 agentic_executor.py 的字面量） ──────────────
        if role == "controller":
            user = f"问题：{q}\n当前状态：{bb.to_text()}"

        elif role == "proposer" and not seen_worker:
            user = (f"问题：{q}\n当前状态："
                    f"{bb.to_text(include_flaws=FLAW_IN_PRIMARY)}")

        elif role == "proposer":
            # 修正轮。没有发起方就无法忠实重建——RL 里 proposer 的修正 prompt
            # 必须带「你之前的解法被 X 指出问题」，凭空编一个就又回到 v2 的老病。
            if pending is None:
                raise Unreplayable("proposer 修正轮缺发起方上下文")
            init_role, init_out, _a, _t, _r = pending
            # 与 executor 同一个判据：黑板的「发现问题」与 initiator_output 在
            # critic 硬触发路径上是逐字节相同的两份拷贝，去掉一份。
            dup = bool(bb.flaws) and bb.flaws[-1]["content"][:300] == init_out[:300]
            user = PromptTemplates.proposer_correction_user(
                q, ROLE_NAMES.get(init_role, init_role), init_out,
                bb.to_text(include_flaws=not dup))

        elif role == "critic":
            user = (f"待审查解法：{last[0]}\n答案：{last[1]}\n"
                    f"当前状态：{bb.to_text()}")

        elif role == "verifier":
            user = (f"待验证答案：{last[1]}\n推理：{last[0]}\n"
                    f"当前状态：{bb.to_text()}")

        else:
            raise ValueError(f"未知角色：{role}")

        # 请求上下文：只有"上一个 worker 确实点名了本角色"才算自发请求。
        # target == "proposer" 的分支不加（RL 侧同样排除，修正 prompt 自带上下文）。
        if role in ("critic", "verifier") and pending is not None:
            init_role, init_out, action, target, reason = pending
            if target == role:
                user += PromptTemplates.request_context(
                    ROLE_NAMES.get(init_role, init_role), action, reason, init_out)

        t["user"] = user

        # ── 再把本 turn 的产出喂进黑板（RL 也是先建 prompt 后喂） ──────────
        shown = strip_interaction(resp)
        if role == "controller":
            continue                      # controller 不写黑板

        seen_worker = True
        if role == "proposer":
            reasoning, answer = parse_reasoning(resp)
            bb.add_message(Message(0, MessageType.TRACE, (reasoning, answer)))
            action, target, reason = parse_interaction(resp)
            if action != "none" and target == "proposer":
                action, target, reason = "none", "none", ""   # 自指归一化
            if action != "none":
                bb.add_message(Message(
                    0, MessageType.INTERACTION,
                    {"from": "proposer", "action": action,
                     "target": target, "reason": reason}))
                pending = ("proposer", shown, action, target, reason)
            else:
                pending = None
            continue

        flagged = False
        if role == "critic":
            flagged = critic_found_errors(resp)
            if flagged:
                bb.add_message(Message(1, MessageType.FLAW, {"content": shown}))
        else:  # verifier
            score = parse_score(resp)
            bb.add_message(Message(
                2, MessageType.SCORE,
                (last[1], score if score is not None else 0.5)))

        nxt = _pending_after(role, resp, shown, flagged)
        if nxt is not None:
            _, _, a2, t2, r2 = nxt
            bb.add_message(Message(
                list(ROLE_NAMES).index(role), MessageType.INTERACTION,
                {"from": role, "action": a2, "target": t2, "reason": r2}))
        pending = nxt

    return row
