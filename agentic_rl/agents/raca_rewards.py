"""RACA v2 Phase 2：角色奖励 + 交互因果奖励（纯 Python，零 torch 依赖）。

对应 RACA_ALGORITHM.md v2 §4：
- 4.1 Proposer 主 turn：r_prop = p_t
- 4.2 交互发起奖励 r_int（固定成本 −c_int + 因果矩阵），r_turn = r_prop + λ·r_int
- 4.3 Critic 四格矩阵；末轮真阳性固定 +0.2
- 4.4 Verifier 校准奖励 1 − |v − p|
- 4.5 Controller 公式不变（底薪 + 效率提成 + 对称惩罚）
- 4.6 响应 turn 按角色语义独立计分

实施决策（规格未完全定死处，据 §4.3 与 §4.6 合并语义）：
- Critic 真阳性的效果量 q 分三级解析：
  1) 本轮内该 critic turn 之后发生了 proposer 修正 → q = 本轮末答案正确性
     （§4.6 "p_{t+1} 以响应后本轮修正结果计"）
  2) 无本轮修正但存在下一轮 → q = 下一轮 primary 正确性（flag 留在黑板，
     因果通路是影响下一轮）
  3) 末轮且无修正 → 固定 +0.2（§4.3 末轮修正，切断幸存者偏差回流）
- r_int 只挂在 primary proposer turn（发起决策是它做的）；响应方在链上
  再发起的下一跳属于链机制，不单独计交互奖励。
- forced 注入轮：发起方 r_int = 0（决策不是它做的）；响应方正常计分。
"""

from __future__ import annotations

from agents.grader import math_equal


def _r_int(u: bool, forced: bool, p_primary: bool, p_end: bool,
           c_int: float) -> float:
    """§4.2 交互发起奖励矩阵。"""
    if forced:
        return 0.0                        # 决策不是发起方做的，不奖不罚
    if u:
        if not p_primary and p_end:
            gain = 0.3                    # 有效求助：错 → 对
        elif not p_primary:
            gain = 0.0                    # 无效求助：错 → 仍错
        else:
            gain = -0.2                   # 画蛇添足：本来就对
        return -c_int + gain
    # 不发起
    return 0.1 if p_primary else 0.0     # 正确的自信 / 机会成本已含在 r_prop


def _r_critic(flagged: bool, p_reviewed: bool, q: float | None) -> float:
    """§4.3 四格矩阵。q 为效果量（None 表示末轮无修正 → 固定分）。"""
    if flagged and not p_reviewed:        # 真阳性
        if q is None:
            return 0.2                    # 末轮修正：固定正分
        return 0.3 * q + 0.1 * (1.0 - q)
    if flagged and p_reviewed:            # 假阳性
        return -0.2
    if not flagged and p_reviewed:        # 真阴性
        return 0.1
    return 0.0                            # 漏检


def compute_turn_data(
    round_records: list,
    correct_answer: str,
    is_correct: bool,
    max_rounds: int,
    cfg: dict,
    stop_ctrl_tid: int | None = None,
    stop_sigma: str = "verify",
) -> tuple[dict, list]:
    """RACA v2 逐 turn 奖励。

    round_record 结构（executor 落盘）：
      sigma, ctrl_tid, gate_blocked, primary_tid, primary_answer,
      corrected_answer(None 表示本轮无修正), u, forced, target,
      critic_turns:  [{tid, flagged, reviewed_answer, correction_followed}],
      verifier_turns:[{tid, score, reviewed_answer}],
      correction_turns:[{tid, answer}]

    Returns:
      turn_data:  {tid: {role, round, sigma, is_response, reward}}
      round_meta: [{u, forced, p_primary, p_end, target, gate_blocked}]（证据指标用）
    """
    alpha      = cfg.get("ctrl_alpha", 0.3)
    beta       = cfg.get("ctrl_beta", 0.2)
    gamma      = cfg.get("ctrl_gamma", 0.3)
    c_int      = cfg.get("c_int", 0.05)
    lambda_int = cfg.get("lambda_int", 1.0)

    def eq(ans: str | None) -> bool:
        return bool(ans) and math_equal(ans, correct_answer)

    turn_data: dict = {}
    round_meta: list = []

    # 预计算逐轮 primary / 轮末正确性
    p_prim_list = [eq(r["primary_answer"]) for r in round_records]
    p_end_list = [
        eq(r["corrected_answer"]) if r["corrected_answer"] is not None else p_prim_list[i]
        for i, r in enumerate(round_records)
    ]

    last_ctrl_tid = None

    for t, rnd in enumerate(round_records):
        sigma  = rnd["sigma"]
        p_prim = p_prim_list[t]
        p_end  = p_end_list[t]

        # ── controller（占位 0.0，episode 结果奖励在最后统一赋给一个 turn） ──
        turn_data[rnd["ctrl_tid"]] = {
            "role": "controller", "round": t, "sigma": sigma,
            "is_response": False, "reward": 0.0,
        }
        last_ctrl_tid = rnd["ctrl_tid"]

        # ── proposer 主 turn：r_prop + λ·r_int ────────────────────────────
        r_prop = 1.0 if p_prim else 0.0
        r_int = _r_int(rnd["u"], rnd["forced"], p_prim, p_end, c_int)
        turn_data[rnd["primary_tid"]] = {
            "role": "proposer", "round": t, "sigma": sigma,
            "is_response": False, "reward": r_prop + lambda_int * r_int,
        }

        # ── critic 响应 turn（四格矩阵，三级 q 解析） ─────────────────────
        for ct in rnd["critic_turns"]:
            p_rev = eq(ct["reviewed_answer"])
            if ct["flagged"] and not p_rev:
                if ct["correction_followed"]:
                    q = float(p_end)                      # 本轮内因果窗口
                elif t + 1 < len(round_records):
                    q = float(p_prim_list[t + 1])         # 跨轮因果窗口
                else:
                    q = None                              # 末轮无修正 → 固定分
            else:
                q = 0.0  # 非真阳性分支不使用 q
            turn_data[ct["tid"]] = {
                "role": "critic", "round": t, "sigma": sigma,
                "is_response": True,
                "reward": _r_critic(ct["flagged"], p_rev, q),
            }

        # ── verifier 响应 turn（校准奖励） ────────────────────────────────
        for vt in rnd["verifier_turns"]:
            if vt["score"] is not None:
                r = 1.0 - abs(vt["score"] - float(eq(vt["reviewed_answer"])))
            else:
                r = 0.0                                   # 输出不可解析
            turn_data[vt["tid"]] = {
                "role": "verifier", "round": t, "sigma": sigma,
                "is_response": True, "reward": r,
            }

        # ── proposer 修正响应 turn（新答案的 p，§4.6） ────────────────────
        for pt in rnd["correction_turns"]:
            turn_data[pt["tid"]] = {
                "role": "proposer", "round": t, "sigma": sigma,
                "is_response": True,
                "reward": 1.0 if eq(pt["answer"]) else 0.0,
            }

        round_meta.append({
            "u":            bool(rnd["u"]),
            "forced":       bool(rnd["forced"]),
            "p_primary":    p_prim,
            "p_end":        p_end,
            "target":       rnd.get("target"),
            "gate_blocked": bool(rnd.get("gate_blocked", False)),
        })

    # ── Controller episode 结果奖励（§4.5，公式与 v1 一致） ─────────────────
    t_stop = len(round_records)
    remaining = (max_rounds - t_stop) / max(max_rounds, 1)
    outcome_tid = stop_ctrl_tid if stop_ctrl_tid is not None else last_ctrl_tid

    if outcome_tid is not None:
        c = float(is_correct)
        ctrl_reward = (
            c
            + alpha * c * remaining
            - beta * (1.0 - c)
            - gamma * (1.0 - c) * remaining
        )
        entry = turn_data.get(outcome_tid)
        if entry is None:
            # stop turn 不在 round_records 中——为其建立条目（继承 v1.x Fix 1）
            turn_data[outcome_tid] = {
                "role": "controller", "round": t_stop, "sigma": stop_sigma,
                "is_response": False, "reward": ctrl_reward,
            }
        else:
            entry["reward"] = ctrl_reward

    return turn_data, round_meta
