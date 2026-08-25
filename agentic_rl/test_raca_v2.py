"""RACA v2 单元测试（纯 CPU，零 torch 依赖）。

运行：python test_raca_v2.py 或 pytest test_raca_v2.py
覆盖：机械 σ 推导 / v2 解析器 / 奖励矩阵（r_int、critic 四格、响应计分、
controller 结果）/ 两层优势（anchor 分组 + 组内去重）。
"""

from agents.parsing import parse_decision, parse_interaction
from agents.raca_rewards import compute_turn_data
from envs.blackboard import Blackboard, Message, MessageType
from training.raca_adv import compute_raca_advantages


# ── helpers ──────────────────────────────────────────────────────────────────

CFG = {"ctrl_alpha": 0.3, "ctrl_beta": 0.2, "ctrl_gamma": 0.3,
       "c_int": 0.02, "lambda_int": 1.0,
       "int_gain": 0.3, "int_overkill": 0.05}


def make_round(t, sigma="explore", primary="4", corrected=None, u=False,
               forced=False, target=None, gate_blocked=False,
               critic_turns=(), verifier_turns=(), correction_turns=()):
    """构造 executor 落盘格式的 round record。tid 规则：每轮基数 t*10。"""
    return {
        "sigma": sigma, "ctrl_tid": t * 10, "gate_blocked": gate_blocked,
        "primary_tid": t * 10 + 1, "primary_answer": primary,
        "corrected_answer": corrected, "u": u, "forced": forced,
        "target": target,
        "critic_turns": list(critic_turns),
        "verifier_turns": list(verifier_turns),
        "correction_turns": list(correction_turns),
    }


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# ── 机械 σ 推导 ──────────────────────────────────────────────────────────────

def test_derive_sigma():
    bb = Blackboard()
    assert bb.derive_sigma() == "explore"                    # 无 trace
    bb.add_message(Message(0, MessageType.TRACE, ("r", "4")))
    assert bb.derive_sigma() == "verify"                     # 有候选、无 flag
    bb.add_message(Message(1, MessageType.FLAW, {"content": "x"}))
    assert bb.derive_sigma() == "refine"                     # flag 未处理
    bb.add_message(Message(0, MessageType.TRACE, ("r2", "5")))
    assert bb.derive_sigma() == "verify"                     # flag 已被新 trace 处理
    bb.add_message(Message(2, MessageType.SCORE, ("5", 0.9)))
    assert bb.derive_sigma() == "verify"                     # score 不影响 σ


# ── v2 解析器 ────────────────────────────────────────────────────────────────

def test_parse_decision():
    assert parse_decision("<meta-plan>\ndecision: stop\nreason: x\n</meta-plan>") == "stop"
    assert parse_decision("decision: continue") == "continue"
    assert parse_decision("乱码 without decision") == "continue"   # 保守默认


def test_parse_interaction():
    ok = "<interaction>\naction: request\ntarget: critic\nreason: 需要审查\n</interaction>"
    assert parse_interaction(ok) == ("request", "critic", "需要审查")
    # challenge 合法
    ch = "<interaction>\naction: challenge\ntarget: verifier\nreason: 分数存疑\n</interaction>"
    assert parse_interaction(ch)[0] == "challenge"
    # v1 遗留 bug 回归：support 带冒号不得穿透
    sp = "<interaction>\naction: support:42\ntarget: critic\nreason: x\n</interaction>"
    assert parse_interaction(sp) == ("none", "none", "")
    assert parse_interaction("<interaction>\naction: none\ntarget: none\n</interaction>") == ("none", "none", "")
    # 容错：target 写进 action 冒号后
    fused = "<interaction>\naction: request:verifier\nreason: x\n</interaction>"
    assert parse_interaction(fused)[:2] == ("request", "verifier")
    # 非法 target
    bad = "<interaction>\naction: request\ntarget: controller\nreason: x\n</interaction>"
    assert parse_interaction(bad) == ("none", "none", "")
    assert parse_interaction("没有 interaction 块") == ("none", "none", "")


# ── 奖励矩阵 ─────────────────────────────────────────────────────────────────

def test_r_int_effective_help():
    """u=1、primary 错、修正对：r_prop=0，r_int=−0.02+0.3=0.28。"""
    rounds = [make_round(0, primary="5", corrected="4", u=True, target="critic",
                         critic_turns=[{"tid": 2, "flagged": True,
                                        "reviewed_answer": "5",
                                        "correction_followed": True}],
                         correction_turns=[{"tid": 3, "answer": "4"}])]
    td, meta = compute_turn_data(rounds, "4", True, 4, CFG)
    assert approx(td[1]["reward"], 0.28)                     # proposer 主 turn
    assert approx(td[2]["reward"], 0.3)                      # critic 真阳性、本轮修对 q=1
    assert approx(td[3]["reward"], 1.0)                      # 修正响应：新答案对
    assert td[2]["is_response"] and td[3]["is_response"]
    assert not td[1]["is_response"]
    assert td[1]["layer_key"] == 0                           # p_t=0 分层
    assert meta[0]["u"] and not meta[0]["p_primary"] and meta[0]["p_end"]


def test_r_int_useless_and_overkill():
    # 无效求助：错→仍错，r_int=−0.02+0 → reward=−0.02
    rounds = [make_round(0, primary="5", u=True, target="verifier",
                         verifier_turns=[{"tid": 2, "score": 0.2,
                                          "reviewed_answer": "5"}])]
    td, _ = compute_turn_data(rounds, "4", False, 4, CFG)
    assert approx(td[1]["reward"], -0.02)
    # 画蛇添足：对还求助，1.0 + (−0.02−0.05) = 0.93
    rounds = [make_round(0, primary="4", u=True, target="critic",
                         critic_turns=[{"tid": 2, "flagged": False,
                                        "reviewed_answer": "4",
                                        "correction_followed": False}])]
    td, _ = compute_turn_data(rounds, "4", True, 4, CFG)
    assert approx(td[1]["reward"], 0.93)
    assert approx(td[2]["reward"], 0.1)                      # critic 真阴性


def test_r_int_no_unconditional_subsidy():
    """v2.1 核心回归：不发起一律 0，不得给无条件补贴。

    v2.0 给“不发起+答对”+0.1，这笔补贴与画蛇添足惩罚共同使 p>0.5 时
    发起的边缘期望恒为负（即使 q=1.0）→ int_rate 必然塌到 0。
    """
    # 不发起 + 对：只有 r_prop=1.0，无额外奖励
    td, _ = compute_turn_data([make_round(0, primary="4")], "4", True, 4, CFG)
    assert approx(td[1]["reward"], 1.0)
    assert td[1]["layer_key"] == 1
    # 不发起 + 错：0（机会成本已由 r_prop=0 体现，不双重计罚）
    td, _ = compute_turn_data([make_round(0, primary="5")], "4", False, 4, CFG)
    assert approx(td[1]["reward"], 0.0)

    # 期望值校准：模型已知自身对错时的条件最优选择必须正确
    c, A, B = CFG["c_int"], CFG["int_gain"], CFG["int_overkill"]
    assert -c + A * 0.5 > 0.0, "p_t=0 且求助半数有效时，发起必须划算"
    assert -c - B < 0.0, "p_t=1 时发起必须不划算"
    # 边缘期望：p=0.6、q=0.5 时应跨越零点（而非恒负）
    p, q = 0.6, 0.5
    marginal = p * (-c - B) + (1 - p) * (-c + A * q)
    assert marginal > 0, f"p=0.6/q=0.5 边缘期望应为正，实测 {marginal:+.4f}"
    # 强模型（p=0.85）应不求助
    p = 0.85
    marginal = p * (-c - B) + (1 - p) * (-c + A * 1.0)
    assert marginal < 0, f"p=0.85 时即使 q=1 也不应求助，实测 {marginal:+.4f}"


def test_r_int_no_initiation_and_forced():
    # forced 注入：发起方不计 r_int → reward = r_prop
    rounds = [make_round(0, primary="5", forced=True,
                         critic_turns=[{"tid": 2, "flagged": True,
                                        "reviewed_answer": "5",
                                        "correction_followed": False}])]
    td, meta = compute_turn_data(rounds, "4", False, 4, CFG)
    assert approx(td[1]["reward"], 0.0)
    assert meta[0]["forced"] and not meta[0]["u"]
    # forced 且答对：仍只有 r_prop，不因被强制调用而被罚
    rounds = [make_round(0, primary="4", forced=True,
                         verifier_turns=[{"tid": 2, "score": 0.9,
                                          "reviewed_answer": "4"}])]
    td, _ = compute_turn_data(rounds, "4", True, 4, CFG)
    assert approx(td[1]["reward"], 1.0)


def test_critic_matrix():
    ct = lambda flagged, followed: [{"tid": 2, "flagged": flagged,
                                     "reviewed_answer": "5",
                                     "correction_followed": followed}]
    # 假阳性（审的是对的答案却挑错）
    rounds = [make_round(0, primary="4",
                         critic_turns=[{"tid": 2, "flagged": True,
                                        "reviewed_answer": "4",
                                        "correction_followed": False}])]
    td, _ = compute_turn_data(rounds, "4", True, 4, CFG)
    assert approx(td[2]["reward"], -0.2)
    # 漏检
    rounds = [make_round(0, primary="5",
                         critic_turns=[{"tid": 2, "flagged": False,
                                        "reviewed_answer": "5",
                                        "correction_followed": False}])]
    td, _ = compute_turn_data(rounds, "4", False, 4, CFG)
    assert approx(td[2]["reward"], 0.0)
    # 真阳性 + 无本轮修正 + 有下一轮：q = 下一轮 primary（对）→ 0.3
    rounds = [make_round(0, primary="5", forced=True, critic_turns=ct(True, False)),
              make_round(1, primary="4")]
    td, _ = compute_turn_data(rounds, "4", True, 4, CFG)
    assert approx(td[2]["reward"], 0.3)
    # 真阳性 + 无修正 + 下一轮仍错：0.3*0+0.1*1 = 0.1
    rounds = [make_round(0, primary="5", forced=True, critic_turns=ct(True, False)),
              make_round(1, primary="5")]
    td, _ = compute_turn_data(rounds, "4", False, 4, CFG)
    assert approx(td[2]["reward"], 0.1)
    # 末轮真阳性 + 无修正：固定 +0.2（v2 §4.3）
    rounds = [make_round(0, primary="5", forced=True, critic_turns=ct(True, False))]
    td, _ = compute_turn_data(rounds, "4", False, 4, CFG)
    assert approx(td[2]["reward"], 0.2)


def test_verifier_calibration():
    vt = lambda score, ans: [{"tid": 2, "score": score, "reviewed_answer": ans}]
    # 审对的答案给 0.8 分：1−|0.8−1|=0.8
    td, _ = compute_turn_data([make_round(0, primary="4", verifier_turns=vt(0.8, "4"))],
                              "4", True, 4, CFG)
    assert approx(td[2]["reward"], 0.8)
    # 审错的答案给 0.9 分：1−|0.9−0|=0.1（过度自信被罚）
    td, _ = compute_turn_data([make_round(0, primary="5", verifier_turns=vt(0.9, "5"))],
                              "4", False, 4, CFG)
    assert approx(td[2]["reward"], 0.1)
    # 不可解析输出
    td, _ = compute_turn_data([make_round(0, primary="4", verifier_turns=vt(None, "4"))],
                              "4", True, 4, CFG)
    assert approx(td[2]["reward"], 0.0)


def test_controller_outcome():
    # stop 于第 2 轮末（t_stop=2, max=4, rem=0.5），答对：1 + 0.3*0.5 = 1.15
    rounds = [make_round(0, primary="4"), make_round(1, primary="4")]
    td, _ = compute_turn_data(rounds, "4", True, 4, CFG,
                              stop_ctrl_tid=99, stop_sigma="verify")
    assert approx(td[99]["reward"], 1.15)
    assert td[99]["role"] == "controller" and td[99]["sigma"] == "verify"
    assert approx(td[0]["reward"], 0.0) and approx(td[10]["reward"], 0.0)  # 占位
    # 立即 stop（t_stop=0, rem=1）答错：−0.2 − 0.3 = −0.5；空轮次也有梯度（Fix 1）
    td, meta = compute_turn_data([], "4", False, 4, CFG,
                                 stop_ctrl_tid=0, stop_sigma="explore")
    assert approx(td[0]["reward"], -0.5) and meta == []
    # 耗尽轮次（rem=0）答错：结果奖励落在最后一个 controller turn，= −0.2
    rounds = [make_round(t, primary="5") for t in range(4)]
    td, _ = compute_turn_data(rounds, "4", False, 4, CFG, stop_ctrl_tid=None)
    assert approx(td[30]["reward"], -0.2)


# ── 两层优势 ─────────────────────────────────────────────────────────────────

def mk_turn(role, rnd, sigma, is_resp, reward):
    return {"role": role, "round": rnd, "sigma": sigma,
            "is_response": is_resp, "reward": reward}


def test_layer1_controller():
    eps = [{0: mk_turn("controller", 0, "explore", False, 1.3)},
           {0: mk_turn("controller", 0, "explore", False, -0.2)}]
    adv = compute_raca_advantages(eps)
    assert approx(adv[0][0], 1.0) and approx(adv[1][0], -1.0)
    # 零方差组：controller 层整体丢弃
    eps = [{0: mk_turn("controller", 0, "explore", False, 1.0)},
           {0: mk_turn("controller", 0, "explore", False, 1.0)}]
    adv = compute_raca_advantages(eps)
    assert adv[0] == {} and adv[1] == {}


def test_layer2_anchor_separation():
    # 同 role 不同 σ / 不同 is_response 不互相比较
    eps = [
        {1: mk_turn("proposer", 0, "explore", False, 1.0),
         2: mk_turn("proposer", 1, "refine", False, 0.0)},
        {1: mk_turn("proposer", 0, "explore", False, 0.0),
         2: mk_turn("proposer", 1, "refine", False, 1.0)},
    ]
    adv = compute_raca_advantages(eps)
    # explore 组: [1,0] → ±1; refine 组: [0,1] → ∓1
    assert approx(adv[0][1], 1.0) and approx(adv[1][1], -1.0)
    assert approx(adv[0][2], -1.0) and approx(adv[1][2], 1.0)
    # is_response 隔离：主 turn 与响应 turn 各自不足 2 样本 → 无优势
    eps = [
        {1: mk_turn("proposer", 0, "explore", False, 1.0)},
        {1: mk_turn("proposer", 0, "explore", True, 0.0)},
    ]
    adv = compute_raca_advantages(eps)
    assert 1 not in adv[0] and 1 not in adv[1]


def test_layer2_dedup_broadcast():
    """v2 §5.2：同 episode 同轮同角色同 reward 的多 turn 去重后入组，优势广播。"""
    eps = [
        # ep0：同轮两个 critic 响应 turn，reward 相同（去重后只算 1 个代表）
        {5: mk_turn("critic", 0, "verify", True, 1.0),
         6: mk_turn("critic", 0, "verify", True, 1.0)},
        {5: mk_turn("critic", 0, "verify", True, 0.0)},
    ]
    adv = compute_raca_advantages(eps)
    # 代表样本 [1.0, 0.0] → μ=0.5, σ=0.5 → ±1.0（若不去重 μ=2/3，数值会不同）
    assert approx(adv[0][5], 1.0) and approx(adv[0][6], 1.0)   # 广播回两个 turn
    assert approx(adv[1][5], -1.0)


def test_layer2_p_t_stratification():
    """v2.1：proposer 主 turn 按 p_t 分层，隔离 r_prop 对交互信号的淹没。

    不分层时：奖励 [1.0(对/不求助), 0.93(对/求助), 0.28(错/求助修对), 0(错/不求助)]
    归一化后优势主要由“对/错”驱动，交互决策的 0.07 差异被 1.0 的差异淹没。
    分层后：p_t=1 组内比 [1.0, 0.93] → 不求助胜；p_t=0 组内比 [0.28, 0]
    → 有效求助胜。两个结论都是“交互决策对不对”，而非“答对了吗”。
    """
    def prop(reward, p_t):
        v = mk_turn("proposer", 0, "explore", False, reward)
        v["layer_key"] = p_t
        return v

    eps = [
        {1: prop(1.00, 1)},   # 对 + 不求助（正确行为）
        {1: prop(0.93, 1)},   # 对 + 求助（画蛇添足）
        {1: prop(0.28, 0)},   # 错 + 求助且修对（正确行为）
        {1: prop(0.00, 0)},   # 错 + 不求助（错过机会）
    ]
    adv = compute_raca_advantages(eps)
    # p_t=1 组：不求助得正优势、求助得负优势
    assert adv[0][1] > 0 > adv[1][1]
    # p_t=0 组：有效求助得正优势、不求助得负优势
    assert adv[2][1] > 0 > adv[3][1]
    # 关键：两组优势量级相当（都是±1）——交互信号未被 r_prop 淹没
    assert approx(abs(adv[0][1]), abs(adv[2][1]))

    # 对照：若不分层（layer_key 全 None），同样四个样本的优势会被“对/错”主导
    eps_flat = [{1: mk_turn("proposer", 0, "explore", False, r)}
                for r in (1.00, 0.93, 0.28, 0.00)]
    adv_flat = compute_raca_advantages(eps_flat)
    # 不分层时，“对+求助”(0.93) 仍获正优势——画蛇添足没被惩罚
    assert adv_flat[1][1] > 0, "不分层时画蛇添足会被错误地奖励（这正是 v2.0 的问题）"


def test_dual_channel_advantage():
    """v2.3（§18）：主 turn 优势 = z(r_prop，不分层) + z(λ·r_int，p_t 分层)。

    四样本：对/不求助、对/求助(画蛇添足)、错/求助修对、错/不求助。
    prop 通道：{1,1,0,0} → 对的 +1、错的 −1；
    int 通道（p1 层 {0,−0.07} / p0 层 {0.28,0}）：决策对的 +1、错的 −1。
    叠加后：对+不求助 = +2；对+求助 = 0；错+修对 = 0；错+不求助 = −2。
    解题信号（v2.1 分层误删）与交互隔离（v2.1 的目的）同时成立。"""
    def prop(r_prop, r_int_w, p_t):
        v = mk_turn("proposer", 0, "explore", False, r_prop + r_int_w)
        v["r_prop"], v["r_int_w"], v["layer_key"] = r_prop, r_int_w, p_t
        return v

    eps = [
        {1: prop(1.0, 0.00, 1)},    # 对 + 不求助
        {1: prop(1.0, -0.07, 1)},   # 对 + 求助（画蛇添足）
        {1: prop(0.0, 0.28, 0)},    # 错 + 求助且修对
        {1: prop(0.0, 0.00, 0)},    # 错 + 不求助
    ]
    adv = compute_raca_advantages(eps)
    assert approx(adv[0][1], 2.0) and approx(adv[1][1], 0.0)
    assert approx(adv[2][1], 0.0) and approx(adv[3][1], -2.0)


def test_dual_channel_no_absorbing_state():
    """v2.3：int_rate=0（r_int 全 0）时 int 通道失活，但 r_prop 通道仍供梯度。

    v2.1/v2.2 的全量分层在此场景下组内零方差 → 主 turn 整组被丢 →
    解题能力停训（v2.2 首跑实测 acc 横盘、len 组成坍缩）。"""
    def prop(r_prop, p_t):
        v = mk_turn("proposer", 0, "explore", False, r_prop)
        v["r_prop"], v["r_int_w"], v["layer_key"] = r_prop, 0.0, p_t
        return v

    eps = [{1: prop(1.0, 1)}, {1: prop(1.0, 1)},
           {1: prop(0.0, 0)}, {1: prop(0.0, 0)}]
    adv = compute_raca_advantages(eps)
    # 主 turn 仍有优势（来自 prop 通道）：对的 +1、错的 −1
    assert approx(adv[0][1], 1.0) and approx(adv[1][1], 1.0)
    assert approx(adv[2][1], -1.0) and approx(adv[3][1], -1.0)


def test_weighted_vote():
    """v2.3（§18 通道①）：verifier 分数加权投票 vs 朴素多数投票。"""
    from envs.blackboard import Blackboard, Message, MessageType
    bb = Blackboard()
    for ans in ("5", "5", "4"):
        bb.add_message(Message(0, MessageType.TRACE, ("r", ans)))
    bb.add_message(Message(2, MessageType.SCORE, ("4", 0.9)))
    bb.add_message(Message(2, MessageType.SCORE, ("5", 0.2)))

    ex_u = _mk_executor(None)                        # 默认 uniform
    ex_w = _mk_executor(None, vote_mode="weighted")
    assert ex_u._majority_vote(bb) == "5"            # 2 票 > 1 票
    assert ex_w._majority_vote(bb) == "4"            # 1×0.9 > 2×0.2
    # 无分数时 weighted 回退朴素投票
    bb2 = Blackboard()
    for ans in ("5", "5", "4"):
        bb2.add_message(Message(0, MessageType.TRACE, ("r", ans)))
    assert ex_w._majority_vote(bb2) == "5"


def test_correction_vote_exclusion():
    """v3（§19）：修正票退出投票池（修正正确率两次测量 ≤ 裸重采样）。"""
    from envs.blackboard import Blackboard, Message, MessageType
    bb = Blackboard()
    # traces：primary "5"、修正 "4"、修正 "4"（修正刷票场景）
    for ans in ("5", "4", "4"):
        bb.add_message(Message(0, MessageType.TRACE, ("r", ans)))
    ex = _mk_executor(None)
    assert ex._majority_vote(bb) == "4"                       # 不排除：修正胜
    assert ex._majority_vote(bb, ["4", "4"]) == "5"           # 排除：primary 胜
    # 全排空回退全量计票（不致空答案）
    bb3 = Blackboard()
    bb3.add_message(Message(0, MessageType.TRACE, ("r", "7")))
    assert ex._majority_vote(bb3, ["7"]) == "7"


def test_flaw_excluded_from_primary_prompt():
    """v3（§19）：include_flaws=False 时 FLAW 不出现在黑板文本，σ 推导不变。"""
    from envs.blackboard import Blackboard, Message, MessageType
    bb = Blackboard()
    bb.add_message(Message(0, MessageType.TRACE, ("r", "5")))
    bb.add_message(Message(1, MessageType.FLAW, {"content": "第二步符号错"}))
    assert "发现问题" in bb.to_text()
    assert "发现问题" not in bb.to_text(include_flaws=False)
    assert bb.derive_sigma() == "refine"   # σ 机制不受 include_flaws 影响


def test_end_to_end_rewards_to_advantages():
    """完整链路：两个 rollout 的 round_records → 奖励 → 优势。"""
    # rollout A：求助并修对；rollout B：不求助、答错
    ra = [make_round(0, primary="5", corrected="4", u=True, target="critic",
                     critic_turns=[{"tid": 2, "flagged": True,
                                    "reviewed_answer": "5",
                                    "correction_followed": True}],
                     correction_turns=[{"tid": 3, "answer": "4"}])]
    rb = [make_round(0, primary="5")]
    td_a, _ = compute_turn_data(ra, "4", True, 4, CFG, stop_ctrl_tid=None)
    td_b, _ = compute_turn_data(rb, "4", False, 4, CFG, stop_ctrl_tid=None)
    adv = compute_raca_advantages([td_a, td_b])
    # controller：A 对（rem=0.75… 无 stop，rem=(4−1)/4）vs B 错 → A 正 B 负
    assert adv[0][0] > 0 > adv[1][0]
    # proposer 主 turn：A 0.25 vs B 0.0 → A 正 B 负
    assert adv[0][1] > 0 > adv[1][1]


# ── 端到端集成：Fake 引擎跑通 executor 主流程 ──────────────────────────

class FakeTokenizer:
    def apply_chat_template(self, messages, **kw):
        return messages[0]["content"] + "\n---\n" + messages[1]["content"]

    # v3.1 prompt 长度保险需要 encode/decode。按字符切分充当 token（比真
    # tokenizer 更保守：字符数 ≥ token 数），足以验证截断逻辑。
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids):
        return "".join(ids)


class FakeEngine:
    """按角色出队的脚本引擎；记录调用以便断言。"""
    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def generate_batch(self, requests):
        out = []
        for r in requests:
            self.calls.append(r["role"])
            text = self.script[r["role"]].pop(0)
            out.append((text, [-0.5, -0.5, -0.5], [11, 12, 13]))
        return out


def _mk_executor(engine, **cfg_over):
    from agents.agentic_executor import AgenticExecutor
    cfg = {"max_rounds": 4, "max_hops": 2, "stop_gate": True, **CFG, **cfg_over}
    return AgenticExecutor(None, FakeTokenizer(), cfg, vllm_engine=engine)


INTER = "<interaction>\naction: {a}\ntarget: {t}\nreason: r\n</interaction>\n"


def test_integration_full_episode():
    """交互链 + stop 闸门 + 修正进多数投票的全流程。"""
    script = {
        "controller": [
            "<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
            "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>",   # 无分数→闸门拦截
            "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>",   # 有分数→终止
        ],
        "proposer": [
            INTER.format(a="request", t="critic") + "推理过程：x\n最终答案：5",   # 错
            INTER.format(a="none", t="none") + "推理过程：fix\n最终答案：4",     # 修正
            INTER.format(a="request", t="verifier") + "推理过程：ok\n最终答案：4",
        ],
        "critic": [
            INTER.format(a="request", t="proposer") + "错误分析：第二步计算错了",
        ],
        "verifier": [
            INTER.format(a="none", t="none") + "分数: 0.9\n验证说明：ok",
        ],
    }
    eng = FakeEngine(script)
    ex = _mk_executor(eng)
    res = ex.run_episodes_batch(["1+3=?"], ["4"])[0]

    # 终止行为：第2轮 stop 被闸门拦截（无分数），第3轮 stop 生效
    assert res["stopped"] is True
    assert len(res["raca_round_meta"]) == 2
    assert res["raca_round_meta"][1]["gate_blocked"] is True
    # 多数投票：traces = [5, 4, 4] → 4 → 答对
    assert res["final_answer"] == "4" and res["is_correct"] is True
    # 轮0：自发求助 + 修对
    m0 = res["raca_round_meta"][0]
    assert m0["u"] and not m0["p_primary"] and m0["p_end"]
    td = res["raca_turn_data"]
    rewards = {(v["role"], v["round"], v["is_response"]): v["reward"]
               for v in td.values()}
    assert approx(rewards[("proposer", 0, False)], 0.28)   # 0 + (−0.02+0.3)
    assert approx(rewards[("critic", 0, True)], 0.3)       # 真阳性且本轮修对
    assert approx(rewards[("proposer", 0, True)], 1.0)     # 修正响应
    assert approx(rewards[("verifier", 1, True)], 0.9)     # 审的是"4"（对）：1−|0.9−1|
    assert approx(rewards[("proposer", 1, False)], 0.93)   # 画蛇添足：1+(−0.02−0.05)
    # stop turn：t_stop=2, rem=0.5, 答对 → 1.15
    stop_r = [v["reward"] for v in td.values()
              if v["role"] == "controller" and v["round"] == 2]
    assert len(stop_r) == 1 and approx(stop_r[0], 1.15)
    # 脚本全部消耗（调用次数精确匹配预期流程）
    assert all(len(v) == 0 for v in eng.script.values()), eng.script
    # 记账对齐：log_probs 与 turn_ids 等长，每 turn 3 token
    assert len(res["log_probs"]) == len(res["turn_ids"]) == 3 * len(res["messages"])


def test_integration_forced_injection_and_ablation():
    # eps_force=1.0：proposer 不求助也会被强制注入审查
    # eps_verifier_share=0 固定注 critic，使本例确定性（verifier 注入由
    # test_integration_gate_blocked_injects_verifier 与 下方 share=1 用例覆盖）
    script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"] * 4,
        # v2.2 机械触发：critic 标错（即使不输出 <interaction>）→ 自动生成修正跳，
        # 故 proposer 脚本为「主 turn、修正」交替 ×4
        "proposer":   [INTER.format(a="none", t="none") + "推理过程：x\n最终答案：5",
                       INTER.format(a="none", t="none") + "推理过程：fix\n最终答案：5"] * 4,
        "critic":     [INTER.format(a="none", t="none") + "错误分析：有错"] * 4,
    }
    eng = FakeEngine(script)
    ex = _mk_executor(eng, eps_verifier_share=0.0)
    res = ex.run_episodes_batch(["q"], ["4"], eps_force=1.0)[0]
    assert all(m["forced"] and not m["u"] for m in res["raca_round_meta"])
    assert "critic" in eng.calls and "verifier" not in eng.calls
    # 强制轮：发起方 r_int=0 → proposer 主 turn 奖励 = 0.0
    prop_main = [v for v in res["raca_turn_data"].values()
                 if v["role"] == "proposer" and not v["is_response"]]
    assert all(approx(v["reward"], 0.0) for v in prop_main)
    # v2.2 机械触发：critic 无 <interaction> 也产生修正跳（漏斗 corr 不再恒 0）
    corr_turns = [v for v in res["raca_turn_data"].values()
                  if v["role"] == "proposer" and v["is_response"]]
    assert len(corr_turns) == 4
    m0 = res["raca_round_meta"][0]
    assert m0["n_flagged"] == 1 and m0["n_corrections"] == 1
    assert m0["flip"] is False   # 修了但仍错 → 未翻转

    # eps_verifier_share=1.0：强制注入改选 verifier（保障 verifier 有训练数据）
    script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"] * 4,
        "proposer":   [INTER.format(a="none", t="none") + "推理过程：x\n最终答案：5"] * 4,
        "verifier":   [INTER.format(a="none", t="none") + "分数: 0.3\n验证说明：可疑"] * 4,
    }
    eng = FakeEngine(script)
    ex = _mk_executor(eng, eps_verifier_share=1.0)
    res = ex.run_episodes_batch(["q"], ["4"], eps_force=1.0)[0]
    assert "verifier" in eng.calls and "critic" not in eng.calls
    # verifier 响应 turn 获得校准奖励（审的是错答案 → 1−|0.3−0|=0.7）
    vt = [v for v in res["raca_turn_data"].values() if v["role"] == "verifier"]
    assert vt and all(approx(v["reward"], 0.7) for v in vt)

    # 消融A：max_hops=0 彻底禁用交互（求助被忽略，u=False，无响应 turn）
    script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"] * 4,
        "proposer":   [INTER.format(a="request", t="critic") + "推理过程：x\n最终答案：4"] * 4,
    }
    eng = FakeEngine(script)
    ex = _mk_executor(eng, max_hops=0, stop_gate=False)
    res = ex.run_episodes_batch(["q"], ["4"], eps_force=1.0)[0]
    assert all(not m["u"] and not m["forced"] for m in res["raca_round_meta"])
    assert "critic" not in eng.calls and "verifier" not in eng.calls


def test_integration_gate_blocked_injects_verifier():
    """v2.1：controller 想停但黑板无分数时，强制注入 verifier 解锁终止路径。

    v2.0 的 ε 注入只注 critic，verifier 无任何兜底通道：一旦 int_rate→0，
    黑板永无 verifier 分数 → stop_gate 拦下所有 stop → episode 全部跑满
    max_rounds（实测：eval 侧 stop_rate 恒 0、avg_turns 恒 8.0）。
    """
    script = {
        "controller": [
            "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>",   # 无分数→被拦
            "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>",   # 此时已有分数→生效
        ],
        "proposer": [
            INTER.format(a="none", t="none") + "推理过程：x\n最终答案：4",
        ],
        "verifier": [
            INTER.format(a="none", t="none") + "分数: 0.9\n验证说明：ok",
        ],
    }
    eng = FakeEngine(script)
    # eps_force=0：证明 verifier 注入是由 gate_blocked 触发，而非 ε 概率
    ex = _mk_executor(eng)
    res = ex.run_episodes_batch(["q"], ["4"], eps_force=0.0)[0]

    m0 = res["raca_round_meta"][0]
    assert m0["gate_blocked"] is True          # 首轮 stop 被闸门拦下
    assert m0["forced"] and not m0["u"]        # 强制注入、非自发
    assert "verifier" in eng.calls             # 注入的是 verifier 而非 critic
    assert "critic" not in eng.calls
    # 分数入板后，第二轮 stop 生效 → episode 不再跑满 max_rounds=4
    assert res["stopped"] is True
    assert len(res["raca_round_meta"]) == 1
    # 被强制注入的轮次，发起方不计 r_int（奖励 = r_prop = 1.0）
    prop_main = [v for v in res["raca_turn_data"].values()
                 if v["role"] == "proposer" and not v["is_response"]]
    assert len(prop_main) == 1 and approx(prop_main[0]["reward"], 1.0)


def test_gate_unlock_when_proposer_self_initiates_to_critic():
    """v3.1：自发起交互且选 critic 时，gate 仍能解锁（v3 崩溃场景）。

    v2.1 的解锁通道写在 `elif`（proposer 未自发起）分支里，当时
    int_rate→0 所以总能触发。v3 零成本冷启动把 int_rate 救到 0.78，
    78% 的轮次改走 `if u:` 分支 → 解锁被遮蔽；其中 75% 又选 critic
    （tgtC=0.75）→ 黑板永无分数 → gate 拦停 10→193 → 跑满 max_rounds
    → prompt 膨胀至 step 151 崩。即“前一个修复的成功造成后一个失效”。
    """
    def _script():
        return {
            "controller": [
                "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>",  # 无分数→拦
                "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>",  # 有分数→生效
                "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>",
                "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>",
            ],
            # 自发起交互且 target=critic——这使旧解锁分支永不执行
            "proposer": [INTER.format(a="request", t="critic")
                         + "推理过程：x\n最终答案：4"] * 4,
            # 无错误：不触发修正跳，保证本轮交互链不会附带产出分数
            "critic": ["错误分析：无错误\n修正建议：无"] * 4,
            "verifier": ["分数: 0.9\n验证说明：ok"] * 4,
        }

    # ── 修复后：gate_unlock=True ─────────────────────────────────
    eng = FakeEngine(_script())
    ex = _mk_executor(eng, gate_unlock=True)
    res = ex.run_episodes_batch(["q"], ["4"], eps_force=0.0)[0]

    m0 = res["raca_round_meta"][0]
    assert m0["gate_blocked"] is True        # stop 被拦
    assert m0["u"] is True                   # 且是自发起（非强制注入）
    assert m0["forced"] is False
    assert m0["target"] == "critic"           # proposer 选的是 critic
    assert "verifier" in eng.calls            # ← 关键：解锁批次仍补上了 verifier
    assert res["stopped"] is True             # 第二轮能停
    assert len(res["raca_round_meta"]) == 1   # 没跑满 max_rounds=4
    assert ex.n_gate_unlocked == 1
    # 解锁 turn 领 verifier 自己的校准奖励，不污染 proposer 的 r_int 归因
    vt = [v for v in res["raca_turn_data"].values() if v["role"] == "verifier"]
    assert len(vt) == 1 and approx(vt[0]["reward"], 1.0 - abs(0.9 - 1.0))

    # ── 消融：gate_unlock=False 应复现死锁（证明是该开关在起作用） ──────
    eng2 = FakeEngine(_script())
    ex2 = _mk_executor(eng2, gate_unlock=False)
    res2 = ex2.run_episodes_batch(["q"], ["4"], eps_force=0.0)[0]
    assert "verifier" not in eng2.calls          # 旧分支被遮蔽，永无分数
    assert res2["stopped"] is False              # stop 永远被拦
    assert len(res2["raca_round_meta"]) == 4     # 跑满 max_rounds——即 v3 故障


def test_unbounded_text_is_capped():
    """v3.1：三个无界文本点均有硬上限（step 151 崩溃的根因）。

    实测：decoder prompt 5036 > max_model_len 4096。膨胀链：
    parse 0.95→0.80 → 兜底抽取出长文本 answer / reasoning 缺失时返回
    整个输出 → 两者进 responder prompt 与黑板文本 → 随轮数累积→撑破上限。
    另：answer 是投票池的键，长文本 answer 彼此各不相同→各占一票稀释投票。
    """
    from agents.parsing import parse_reasoning, MAX_ANSWER_CHARS, MAX_REASONING_CHARS

    # ①「最终答案：」后接大段论述 → 当解析失败，回退抽数字，而非截断后当真
    _, ans = parse_reasoning("推理过程：x\n最终答案：" + "啦" * 500 + "42")
    assert len(ans) <= MAX_ANSWER_CHARS and ans == "42"
    # 正常答案（含 LaTeX）不受影响
    _, ok = parse_reasoning("推理过程：x\n最终答案：\\frac{\\pi}{2}")
    assert ok == "\\frac{\\pi}{2}"

    # ②reasoning 缺失时返回整个输出，必须限长
    reasoning, _ = parse_reasoning("没有任何格式的长输出" + "哦" * 5000)
    assert len(reasoning) <= MAX_REASONING_CHARS

    # ③黑板文本：答案数量与单个长度均不随轮数无限增长
    from envs.blackboard import Blackboard, Message, MessageType
    bb = Blackboard()
    for k in range(30):
        bb.add_message(Message(0, MessageType.TRACE, ("r", f"ans{k}" + "垃" * 200)))
    bb.add_message(Message(0, MessageType.TRACE, ("r", "")))   # 解析失败的空串
    text = bb.to_text()
    assert "" not in bb._answers_for_display()      # 空串不进展示
    assert len(bb._answers_for_display()) <= bb._MAX_SHOWN_ANSWERS
    assert len(text) < 1000                        # 31 个×200 字本来会过万
    # 但去重语义不变（投票与奖励仍看到全部答案）
    assert len(bb.get_distinct_answers()) == 31


def test_distinct_answers_order_is_deterministic():
    """v3.1：保序去重。list(set) 的顺序依赖字符串 hash（Python 默认
    随机化），会让 prompt 内容与 max(..., key=) 的平分 tie-break 跳进程
    不可复现——同一 checkpoint 重跑得不到同一结果。"""
    from envs.blackboard import Blackboard, Message, MessageType
    bb = Blackboard()
    for a in ["7", "3", "11", "3", "5"]:
        bb.add_message(Message(0, MessageType.TRACE, ("r", a)))
    assert bb.get_distinct_answers() == ["7", "3", "11", "5"]   # 严格插入序


def test_parse_rate_metric():
    """primary_parsed 逐层透传：无「最终答案：」字段时计为未解析。"""
    # 未按格式输出（靠抽末尾数字兜底）
    script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"] * 4,
        "proposer":   [INTER.format(a="none", t="none") + "算下来就是 4"] * 4,
    }
    ex = _mk_executor(FakeEngine(script), stop_gate=False)
    res = ex.run_episodes_batch(["q"], ["4"])[0]
    assert all(m["primary_parsed"] is False for m in res["raca_round_meta"])
    # 按格式输出
    script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"] * 4,
        "proposer":   [INTER.format(a="none", t="none") + "推理过程：x\n最终答案：4"] * 4,
    }
    ex = _mk_executor(FakeEngine(script), stop_gate=False)
    res = ex.run_episodes_batch(["q"], ["4"])[0]
    assert all(m["primary_parsed"] is True for m in res["raca_round_meta"])


def test_ref_adapter_naming_avoids_peft_substring_trap():
    """ref adapter 名必须与角色名互不为子串（v2.1 首启即崩的根因）。

    PEFT 的 get_peft_model_state_dict 用**子串匹配**筛选 adapter 权重：
        {k: v for k, v in sd.items() if ("lora_" in k and adapter_name in k) or ...}
    旧命名 `ref_controller` 含 `controller` 子串，导出 controller 时 ref 权重
    被误纳入；而后续 key 清理用 `k.replace(f".{adapter_name}", "")`，
    `.ref_controller.` 中 `controller` 前面是 `_` 而非 `.`，替换不生效，
    于是 `lora_A.ref_controller.weight` 原样写入 safetensors，vLLM 报
    `ValueError: ... is unsupported LoRA weight`。
    """
    from llm.adapter_names import ADAPTER_NAMES, REF_ADAPTER, REF_PREFIX

    refs = list(REF_ADAPTER.values())
    assert len(refs) == len(ADAPTER_NAMES) == 4
    assert len(set(refs)) == 4, f"ref 名重复: {refs}"

    # 核心：任一角色名不得是任一 ref 名的子串（反之亦然）
    for role in ADAPTER_NAMES:
        for ref in refs:
            assert role not in ref, f"ref {ref!r} 含角色名 {role!r} → PEFT 导出会混入"
            assert ref not in role, f"角色名 {role!r} 含 ref 名 {ref!r}"
    # 旧命名作为反例：必须被上述规则判为非法
    for role in ADAPTER_NAMES:
        assert role in f"ref_{role}", "反例自洽失败"

    # 模拟 PEFT 的筛选 + key 清理，验证新命名下导出干净
    def peft_export(sd_keys, adapter_name):
        picked = [k for k in sd_keys if "lora_" in k and adapter_name in k]
        return [k.replace(f".{adapter_name}", "") for k in picked]

    for role in ADAPTER_NAMES:
        ref = REF_ADAPTER[role]
        sd = [f"base.layers.0.q_proj.lora_A.{r}.weight" for r in ADAPTER_NAMES] + \
             [f"base.layers.0.q_proj.lora_A.{r}.weight" for r in refs]
        out = peft_export(sd, role)
        assert out == ["base.layers.0.q_proj.lora_A.weight"], \
            f"导出 {role} 时 key 不干净: {out}"
        # 旧命名下的对照：会泄出带 ref 后缀的 key（vLLM 崩溃的直接原因）
        sd_old = [f"base.layers.0.q_proj.lora_A.{r}.weight" for r in ADAPTER_NAMES] + \
                 [f"base.layers.0.q_proj.lora_A.ref_{r}.weight" for r in ADAPTER_NAMES]
        out_old = peft_export(sd_old, role)
        assert any("ref_" in k for k in out_old), "旧命名应复现泄露问题"

    # lora_parameters / save_pretrained 的过滤条件与命名一致
    for ref in refs:
        assert ref.startswith(REF_PREFIX)
        assert f".{REF_PREFIX}" in f"x.lora_A.{ref}.weight"
    for role in ADAPTER_NAMES:
        assert f".{REF_PREFIX}" not in f"x.lora_A.{role}.weight"


def test_signal_quality_metrics():
    """信号质量统计：全对组/全错组占比 + 组内 std（零 torch 依赖）。"""
    from training.metrics import rollout_metrics as T_metrics

    def ep(correct, stopped=True, meta=None):
        return {"is_correct": correct, "stopped": stopped,
                "raca_round_meta": meta if meta is not None else []}

    # 4 组：全对 / 全错 / 混合 / 混合
    batch = [
        [ep(True), ep(True)],
        [ep(False), ep(False)],
        [ep(True), ep(False)],
        [ep(True), ep(False)],
    ]
    m = T_metrics(batch)
    assert approx(m["all_pass_frac"], 0.25)
    assert approx(m["all_fail_frac"], 0.25)
    # std: [0, 0, 0.5, 0.5] → 均值 0.25
    assert approx(m["group_reward_std"], 0.25)
    # stop 校准：全部 stopped=True → stop_acc = 4/8，无 exhaust_acc
    assert approx(m["stop_acc"], 0.5) and "exhaust_acc" not in m
    assert approx(m["stop_rate"], 1.0)

    # 信号枯竭极端：全部组全对 → all_pass=1, std=0（无可用梯度）
    m = T_metrics([[ep(True), ep(True)], [ep(True), ep(True)]])
    assert approx(m["all_pass_frac"], 1.0) and approx(m["group_reward_std"], 0.0)

    # 交互指标 + 可解析率（从 round_meta 聚合）
    meta_ok = [{"u": True, "forced": False, "p_primary": False, "p_end": True,
                "target": "critic", "gate_blocked": False, "primary_parsed": True}]
    meta_bad = [{"u": False, "forced": False, "p_primary": True, "p_end": True,
                 "target": None, "gate_blocked": True, "primary_parsed": False}]
    m = T_metrics([[ep(True, meta=meta_ok), ep(False, meta=meta_bad)]])
    assert approx(m["int_rate"], 0.5)
    assert approx(m["int_effectiveness"], 1.0)   # 唯一求助样本修对了
    assert approx(m["parse_rate"], 0.5)
    assert m["gate_blocked"] == 1
    assert approx(m["int_selectivity"], -1.0)    # u 与 p_primary 完全负相关

    # 空输入不崩
    assert T_metrics([]) == {}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"All {len(fns)} v2 tests passed.")
