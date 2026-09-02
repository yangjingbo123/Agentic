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
- forced 注入轮：发起方数值 r_int = 0，**但 r_int_w=None、完全不进入交互优势
  通道**（决策不是它做的）；r_prop 与响应方奖励照常。注意 0 与 None 不等价：
  组相对归一化中 0 仍参与排名，08-28 v32_miss 已实测它会高于 −int_miss、给
  模型实际输出的 action:none 正优势，使 int_rate 再次塌到 0。
"""

from __future__ import annotations

from agents.grader import math_equal


def _r_int(u: bool, forced: bool, p_primary: bool, p_end: bool,
           c_int: float, gain: float, overkill: float, miss: float = 0.0) -> float:
    """§4.2 交互发起奖励矩阵（v2.1 重设计，见 RACA_ALGORITHM.md §13）。

    v2.0 的矩阵有数学缺陷：「不发起+答对」给 +0.1 是一笔**无条件补贴**，
    与「发起+答对」的 −0.2 共同制造了 0.3p 的固定 gap，而有效求助最多只能
    赚 0.3(1−p)。p>0.5 时无论求助有效率 q 多高，发起的边缘期望恒为负
    —— int_rate→0 是数学必然。实测：step 20 交互率就从 0.58 塔到 0.02。

    v2.1 修正：
    - 删除不发起的无条件补贴（penalty-only 原则：不为“什么都不做”发奖）
    - 画蛇添足惩罚 −0.2 → −overkill(0.05)，交互成本 0.05 → c_int(0.02)
    使边缘期望在现实区间跨越零点：弱（p≈0.3）时求助划算、强（p≈0.85）时不划算，
    模型的自我置信度成为决定因素（即“选择性交互”）。

    ── v3.2 第十二轮：`miss`（该问而没问的机会成本） ────────────────────────
    **v2.1 只移动了零点，没有改变"工作点落在负期望区"这个性质——于是 v2.0 的
    病原样复发了。** 08-28 两跑（`v32_m1` 对照 / `v32_open`）实测：int_rate 从
    step1 的 0.75 在 40 步内塌到 **0.01**，随后 160 步不动；两跑一致，`sel` 始终
    在 +0.01~+0.05、从未转负。

    用实测数字复算（`c_int=0`、`gain=0.3`、`overkill=0.05`、实测 `eff`=q=0.12）：

        E[求助] = P(错)·q·gain − P(对)·overkill = 0.086·P(错) − 0.05

    零点在 **P(错)=0.581**，而模型的实际轮级答错率约 0.35~0.45 —— 零点落在工作
    区间**之外**，且与 v2.0 同在一侧。所以"塌"不是训练不稳，是策略把这道题算对了。

    08-28 `v32_miss` 又补了一层反证：加 miss 后 int_rate 先维持到 step15，但仍在
    step25 塌到 0。此前说“forced 轮 r_int=0，所以给发起决策零梯度”**不完整**：
    在下面这层奖励函数里 0 确实表示不加总奖励，但后面的 GRPO 会按 p_primary 分层
    z-score，**数值 0 仍参与排名**。在首答错层里它高于自发不求助的 −miss，给 forced
    轮模型实际写出的 `action:none` 正优势。真正的零梯度必须由调用方用
    `r_int_w=None` 让该 turn 完全退出 int anchor group，而不是在这里返回一个 0。

    **修法不是调 gain/overkill。** 那两个方向都只把边缘期望整体推正 → 无差别求助
    → v3 已实测（int_rate=0.78 而 sel≈0）。两端都踩过了，问题不在系数大小。

    真正缺的一项是**不作为的定价**：求助犯的错（画蛇添足）有价，不求助犯的错
    （该问而没问）免费。补上 `miss` 之后零点变成 0.05/(0.086+miss)：
    miss=0.05 → 0.368；**miss=0.10 → 0.269**（取后者，因为 P(错) 目前只有从 acc
    反推的区间 0.35~0.45，没有直接埋点；选不依赖精确值的那个。同一轮已补
    `p_primary_rate`，下一跑起 δ 可事后核）。条件方向上的 gap 也从 0.036 放大到
    0.136，而"对的时候问仍扣 overkill"保持不变——**双向压力都在，不会塌向另一端。**

    **原注释里那句反对意见保留在这里，因为它半对半错，值得记住**：
    > "错而不求助的机会成本已由 r_prop=0 体现，不需双重计罚。"
    对 `r_prop` 这一半是对的——答错确实已经被罚过一次。错的是"不需双重计罚"这个
    推论：`r_prop` 对「问不问」**没有区分度**（答错就是 0，问了也 0、不问也 0），
    所以它给发起决策贡献的梯度**恒为零**。两者定价的是不同信道上的不同动作：
    `r_prop` 定价"答案对不对"，`miss` 定价"错了要不要求助"。不是同一件事罚两遍。

    已知副作用（预期，不是 bug）：会出现**保险式求助**——自认为可能错时，即使求助
    救不回来（拿 0）也优于不求助（拿 −miss）。所以 int_rate 会偏高，其中混着一部分
    无效求助。这一点由 `eff` 如实暴露（eff 低而 int_rate 高 = 保险式求助多）；若偏
    得过头，下一步是把 `c_int` 从 0 抬起来给求助加固定门槛，**但那是另一轮的事**。
    """
    if forced:
        # 这里只保持 total reward 的历史口径为 0；**不能把这个 0 送进组相对优势**。
        # 调用方在 turn_data 里写 r_int_w=None，使该 turn 完全退出 int 通道。
        return 0.0
    if u:
        if not p_primary and p_end:
            g = gain                      # 有效求助：错 → 对
        elif not p_primary:
            g = 0.0                       # 无效求助：错 → 仍错（只付成本）
        else:
            g = -overkill                 # 画蛇添足：本来就对
        return -c_int + g
    # 不发起：答对 → 0（penalty-only，绝不为"什么都不做"发奖，v2.0 那笔无条件
    # 补贴已证伪）；答错 → −miss（该问而没问的机会成本）。miss=0 时退化为 v2.1
    # 的行为，所以这一项是可消融的。
    return 0.0 if p_primary else -miss


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
    c_int      = cfg.get("c_int", 0.02)
    lambda_int = cfg.get("lambda_int", 1.0)
    gain       = cfg.get("int_gain", 0.3)        # 有效求助收益
    overkill   = cfg.get("int_overkill", 0.05)   # 画蛇添足惩罚（正数，内部取负）
    # 该问而没问的机会成本（正数，内部取负）。**默认 0.0 而不是 0.10**：缺配置时
    # 退化成 v2.1 行为，不会静默改掉历史口径；要启用必须在 yaml 里显式写。
    miss       = cfg.get("int_miss", 0.0)

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
        r_int = _r_int(rnd["u"], rnd["forced"], p_prim, p_end,
                       c_int, gain, overkill, miss)
        turn_data[rnd["primary_tid"]] = {
            "role": "proposer", "round": t, "sigma": sigma,
            "is_response": False, "reward": r_prop + lambda_int * r_int,
            # v2.3 双通道（§18）：优势阶段 r_prop 与 λ·r_int 各自组内归一化后
            # 相加。r_prop 通道不分层（恢复解题信号：v2.1 的全量分层使
            # int_rate=0 后主 turn 组内零方差、解题能力停训）；r_int 通道
            # 仍按 layer_key=p_t 分层（v2.1 的隔离目的不变）。
            "r_prop": r_prop,
            # **forced 必须是 None（通道缺席），不能是数值 0。** 08-28 的 `v32_miss`
            # 已经给了反证：int_miss 把崩塌从 step10 延后到 step25，但 eval 从
            # step30 起仍是 int_rate=0。根因是 r_int 按 p_primary 分层 z-score：
            # 在「首答错」层里，自发不求助是 −miss，而 forced 轮虽然模型也输出
            # `action:none`，旧代码给 0。标准化后 0 > −miss，forced 的 `none` 反而
            # 获得**正优势**，在训练模型不交互；int_rate 越低，ε forced 占比越高，
            # 这个反向信号越强，最终再次塌到 0。
            #
            # `None` 的语义是：该 turn **不进入 int anchor group**（见 raca_adv.py），
            # 真正做到「决策不是 proposer 做的，所以没有交互梯度」。它仍保留：
            # ① r_prop 解题通道；② critic/verifier 响应奖励；③ q_forced 随机对照读数。
            # total reward 继续用上面的 r_int=0，所以历史日志的 reward 口径不变。
            "r_int_w": None if rnd["forced"] else lambda_int * r_int,
            "layer_key": int(p_prim),
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
            # 格式健康（可解析率监控）；缺失时保守计为已解析
            "primary_parsed": bool(rnd.get("primary_parsed", True)),
            # 把上面那个合取拆开的两个读数（#23）。**必须在这里显式带一手**：
            # `round_meta` 是重新拼的白名单 dict，不是 `round_records` 的透传，
            # 所以在 executor 里记了不等于 `metrics` 读得到——漏掉这两行的后果是
            # 两个指标照样打印、但永远是 0.00，比不加更坏（假的「没问题」）。
            # 默认取 False（= 没失败）而不是 `primary_parsed` 的 True：这两个是
            # **失败计数**，缺键时低报比虚报安全。
            "no_label":     bool(rnd.get("no_label", False)),
            "empty_answer": bool(rnd.get("empty_answer", False)),
            # 修正漏斗（v2.2）：flag → correction → flip。计入全部轮
            # （含 forced 注入），比 eff（仅自发求助）覆盖更广。
            "n_flagged":     sum(1 for ct in rnd["critic_turns"] if ct["flagged"]),
            "n_corrections": len(rnd["correction_turns"]),
            "flip":          (not p_prim) and p_end,
            # 第十轮补的反向计数。**没有它，"修正到底值不值得进投票池"这个问题结构
            # 上答不了。** v3 step150 的 `fnl=561/566/76` 只说"救回来 76 次"，而
            # `configs/agentic/default.yaml` 里 `correction_in_vote` 那条注释记的是
            # 净效应为负（Δ=−0.062/−0.018）——两者同时成立只能说明反向次数 **>76**，
            # 但具体多少现有落盘一个字节都查不到（`train.py` 只 dump `{"step": step}`）。
            # 于是打开开关之后 acc 变了 0.01，你分不清是"帮 76 毁 70"还是"帮 200 毁
            # 194"，而这两种情形下一步该做的事完全相反。
            #
            # 注意 `p_end` 的缺省语义（见 `p_end_list` 的构造）：没有修正 turn 时
            # `p_end = p_prim`，所以 `flip` 与 `unflip` 在无修正的轮上都为 False，
            # 两个计数天然只统计真发生过修正的轮，不需要额外守卫。
            # 又：`round_meta` 是**重新拼的白名单 dict、不是 `round_records` 的透传**，
            # 漏带这一手的后果是指标照样打印、但永远 0.00（#23 已经在这一跳栽过一次）。
            "unflip":        p_prim and (not p_end),
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
