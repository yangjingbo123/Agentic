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


def test_parsers_survive_m1_trailing_interaction_block():
    """M1：`<interaction>` 移到实质输出之后，三个角色的解析器都不能被带偏。

    这不是可选的兼容性测试——M1 是 Eq (12) 良定义性的前提（决策必须在已知
    p_primary 时做出），而它一旦落地，块就位于 `最终答案：` 之后。原来的
    `最终答案：(.+)` 没有 `<` 前瞻，同行书写会把整个块吃进答案；块长 68 字符
    但答案段可能不足 64，于是这串带标签的垃圾会当成合法答案进投票池占一票。
    """
    from agents.parsing import (
        critic_found_errors, parse_reasoning, parse_score, MAX_ANSWER_CHARS)

    blk = "<interaction>\naction: request\ntarget: critic\nreason: 不确定\n</interaction>"

    # proposer：块换行写（模板规定的形态）
    rs, ans = parse_reasoning(f"推理过程：先算 6*7=42\n最终答案：42\n{blk}")
    assert ans == "42"
    assert "<interaction>" not in rs and rs == "先算 6*7=42"
    assert parse_interaction(f"推理过程：x\n最终答案：42\n{blk}")[:2] == ("request", "critic")

    # proposer：块紧贴同一行——正则修的就是这一路
    _, ans_inline = parse_reasoning(f"推理过程：x\n最终答案：42{blk}")
    assert ans_inline == "42", f"块被吃进答案：{ans_inline!r}"

    # 答案本身就是输出末尾（无块、无换行）仍要能取到
    assert parse_reasoning("推理过程：x\n最终答案：42")[1] == "42"
    # LaTeX 答案不含数字时也不能退化成抽末尾数字
    assert parse_reasoning(f"推理过程：x\n最终答案：\\frac{{\\pi}}{{2}}\n{blk}")[1] \
        == "\\frac{\\pi}{2}"
    # 「最终答案：」后接长篇论述仍按解析失败处理（v3.1 行为不受 M1 影响）
    long_ans = "啦" * (MAX_ANSWER_CHARS + 10) + "42"
    assert parse_reasoning(f"推理过程：x\n最终答案：{long_ans}\n{blk}")[1] == "42"

    # critic：错误分析在前，块在后；分析文本不得混入块
    ct = f"错误分析：第二步把 6*7 算成了 41\n{blk}"
    assert critic_found_errors(ct) is True
    assert parse_interaction(ct)[:2] == ("request", "critic")
    assert critic_found_errors(f"错误分析：无错误\n{blk}") is False

    # verifier：分数在前，块在后
    vt = f"分数: 0.35\n验证说明：第二步可疑\n{blk}"
    assert parse_score(vt) == 0.35
    assert parse_interaction(vt)[:2] == ("request", "critic")


def test_reasoning_lookahead_is_block_tag_not_bare_lt():
    """推理的前瞻必须是块的开标签 `<interaction`，不能是裸 `<`。

    裸 `<` 配 `re.S` + 非贪婪 `.+?` 会让推理停在**第一个 `<`**——而数学里 `<`
    是不等号。实测 v2+v3 的 599 个 proposer turn 中 32 个（5.3%）中招，中位只
    保留 24% 的推理，最坏 316 字剩 10 字（就是下面这条的形状：分段函数的
    `若 x < -2`）。这段残文正是 `traces[-1][0]`，会原样进 critic 的
    `待审查解法：` 与 verifier 的 `推理：`，也就是让 critic 去审查一个被截在
    第一个不等号处的片段还要回答"有没有计算错误"。与 flaw 窗口 80 同类
    （静默信道截断），自 v2 就在，与 M1 无关。
    """
    from agents.parsing import parse_reasoning

    blk = "<interaction>\naction: none\ntarget: none\nreason: 有把握\n</interaction>"
    resp = (
        "推理过程：\n"
        "首先解第一段：若 x < -2，令 2x+7 = -5，得 x = -6，满足定义域。\n"
        "然后解第二段：若 x ≥ -2，令 -x^2 - x + 1 = -5，得 x = -3 或 x = 2，\n"
        "其中 x = -3 不满足 x ≥ -2 舍去。\n"
        "所以两解为 -6 与 2，其和为 -4。\n"
        "最终答案：-4\n" + blk
    )
    rsn, ans = parse_reasoning(resp)
    assert ans == "-4"
    # 不等号后面的内容一个字都不能丢：整条推理必须完整保留
    assert rsn.endswith("其和为 -4。"), f"推理被不等号截断：{rsn!r}"
    assert "x ≥ -2" in rsn and "因式" not in rsn  # 内容完整，且没越界吃别的
    # 块仍然被排除在推理之外（前瞻仍在起作用，不是简单删掉了分支）
    assert "<interaction>" not in rsn and "action:" not in rsn

    # 答案标签之后出现不等号时也不能被误截
    assert parse_reasoning(f"推理过程：x\n最终答案：x < 3\n{blk}")[1] == "x < 3"


def test_reasoning_fallback_never_leaks_interaction_block():
    """缺「推理过程：」标签时的兜底路径也不能把块带出来。

    前瞻只在**标签存在**时挡住块；标签缺失时旧代码直接 `return text`，块就跟着
    整段输出被当成推理返回。而这个串正是 `traces[-1][0]`——会原样进 critic 的
    `待审查解法：` 与 verifier 的 `推理：`，白占 68 字符窗口，还教 critic 去审查
    一段带着协议标签的文本。实测 SFT 数据 599 个 proposer turn 中 17 个（2.8%）
    如此；运行时只会更多，因为 v3 的 `parse_rate` 一度掉到 0.80，掉的正是标签。

    这条不是审出来的，是 `prepare_sft.py` 里「重放后 user 剥块必须是空操作」那句
    断言炸出来的——所以那种"本该是空操作"的断言值得多写。
    """
    from agents.parsing import parse_reasoning

    blk = ("<interaction>\naction: request\ntarget: critic\n"
           "reason: 请复核代数\n</interaction>")
    # 真实数据里的形态：整条英文，没有中文标签
    eng = f"Reasoning:\nWe have |a+b|^2 = 4, so the answer is 4.\n{blk}"
    rsn, ans = parse_reasoning(eng)
    assert "<interaction>" not in rsn and "action:" not in rsn, \
        f"块从兜底路径漏进推理：{rsn!r}"
    assert rsn.strip().endswith("the answer is 4."), rsn
    assert "interaction" not in ans

    # 残块（漏了闭标签）：`strip_interaction` 的正则匹配不上，所以还要按开标签
    # 截一刀，否则这一路照样漏
    broken = "Reasoning:\nthe answer is 4.\n<interaction>\naction: request"
    rsn2, _ = parse_reasoning(broken)
    assert "<interaction" not in rsn2 and "action:" not in rsn2, \
        f"残块从兜底路径漏进推理：{rsn2!r}"


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


def test_vote_pool_instrumentation():
    """票池埋点：_vote_pool 与 _majority_vote 共用同一真相源，dist/deg/margin 可信。

    deg（dist≤1）是拿来判「加大投票池是不是杠杆」的量，错了会直接误导方向。
    """
    from collections import Counter
    from envs.blackboard import Blackboard, Message, MessageType
    ex = _mk_executor(None)

    def stats(bb, exclude=None):
        pool = ex._vote_pool(bb, exclude) if bb.traces else Counter()
        n = sum(pool.values())
        top2 = pool.most_common(2)
        marg = ((top2[0][1] - (top2[1][1] if len(top2) > 1 else 0)) / n) if n else 0.0
        return n, len(pool), marg

    # 退化池：4 票全相同 → dist=1、margin=1.0（投票没做任何事）
    bb_deg = Blackboard()
    for _ in range(4):
        bb_deg.add_message(Message(0, MessageType.TRACE, ("r", "5")))
    assert stats(bb_deg) == (4, 1, 1.0)

    # 健全池：3:1 → dist=2、margin=0.5
    bb_ok = Blackboard()
    for ans in ("5", "5", "5", "4"):
        bb_ok.add_message(Message(0, MessageType.TRACE, ("r", ans)))
    assert stats(bb_ok) == (4, 2, 0.5)

    # 平票 2:2 → margin=0（聚合没有分辨力，不能和退化池混淆）
    bb_tie = Blackboard()
    for ans in ("5", "5", "4", "4"):
        bb_tie.add_message(Message(0, MessageType.TRACE, ("r", ans)))
    n, dist, marg = stats(bb_tie)
    assert (n, dist) == (4, 2) and marg == 0.0

    # exclude 后的池才是实际计票池；且与 _majority_vote 的结果一致
    assert stats(bb_ok, ["5"]) == (3, 2, 1.0 / 3)
    assert ex._majority_vote(bb_ok, ["5"]) == "5"      # 2 票 > 1 票
    assert ex._majority_vote(bb_ok, ["5", "5", "5"]) == "4"

    # 全排空回退全量计票：埋点不能报 0 票（否则 deg 会被污染）
    bb_one = Blackboard()
    bb_one.add_message(Message(0, MessageType.TRACE, ("r", "7")))
    assert stats(bb_one, ["7"]) == (1, 1, 1.0)

    # 空黑板：不得抛异常，投票数为 0
    assert stats(Blackboard()) == (0, 0, 0.0)


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
        self.prompts = []          # (role, prompt) —— 供「下游到底看到了什么」的断言

    def generate_batch(self, requests):
        out = []
        for r in requests:
            self.calls.append(r["role"])
            self.prompts.append((r["role"], r["prompt"]))
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


def test_self_target_counts_as_no_interaction():
    """proposer 自指 target 必须记为「没发起」，而不是发起了一次不执行的交互。

    hop 循环里 `target == initiator` 会被 continue 掉——交互从未执行。但旧代码
    仍按 u=True 记账：INTERACTION 落黑板、int_rate 计入一次、r_int 按「发起了
    求助」定价。被污染的恰好是 eff / sel / 修正漏斗，也就是用来判断「是否学会
    何时求助」的那三组数字。所以 u 的谓词必须与 hop 的执行条件共用同一个判据。

    注：`parse_interaction` 已把非法 target 整体退回 ("none","none","")，所以
    自指是这条丢弃路径**唯一**可达的情形——本测试即覆盖全部可达面。
    """
    def _script(t):
        return {
            "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"],
            "proposer": [INTER.format(a="request", t=t) + "推理过程：x\n最终答案：5"],
            "critic":   ["错误分析：无错误\n修正建议：无"],
            "verifier": ["分数: 0.9\n验证说明：ok"],
        }

    def _run(t):
        eng = FakeEngine(_script(t))
        # max_rounds=1 + continue：单轮自然耗尽，不触发闸门（gate_blocked 会
        # 强制注入 verifier，那会盖掉本测试要看的 forced=False）
        ex = _mk_executor(eng, max_rounds=1, gate_unlock=False)
        res = ex.run_episodes_batch(["q"], ["4"], eps_force=0.0)[0]
        return eng, ex, res

    eng, ex, res = _run("proposer")
    m0 = res["raca_round_meta"][0]
    assert m0["u"] is False                  # ← 核心：不是「发起了但被丢弃」
    assert m0["forced"] is False             # 也不该被 ε 注入顶替（eps_force=0）
    assert m0["target"] is None              # 不落 target，q_spont/q_forced 不被稀释
    assert ex.n_self_target == 1             # 但要留下埋点，好看清模型多久犯一次
    assert eng.calls == ["controller", "proposer"]   # 交互链一跳都没跑

    # 对照组：显式 action=none。语义上「自指」≡「没发起」，两者的逐 turn 奖励
    # 必须逐位相同——这比断言某个 r_int 常量更强，因为它锁死了整条归因链。
    eng_n, ex_n, res_n = _run("none")
    assert ex_n.n_self_target == 0
    assert res_n["raca_round_meta"][0]["u"] is False
    rw   = sorted((v["role"], v["reward"]) for v in res["raca_turn_data"].values())
    rw_n = sorted((v["role"], v["reward"]) for v in res_n["raca_turn_data"].values())
    assert len(rw) == len(rw_n)
    for (r1, v1), (r2, v2) in zip(rw, rw_n):
        assert r1 == r2 and approx(v1, v2)

    # 阳性对照：合法 target 仍照常发起（证明上面的归一化没有一刀切关掉交互）
    eng_c, ex_c, res_c = _run("critic")
    assert res_c["raca_round_meta"][0]["u"] is True
    assert res_c["raca_round_meta"][0]["target"] == "critic"
    assert ex_c.n_self_target == 0
    assert "critic" in eng_c.calls


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

    def ep(correct, stopped=True, meta=None, **pool):
        return {"is_correct": correct, "stopped": stopped,
                "raca_round_meta": meta if meta is not None else [], **pool}

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

    # ── 票池埋点（训练侧，v3.2 新增） ────────────────────────────────
    # deg=1.0 意味着池子塌成单一答案：此时投票只是重复确认第一个答案，
    # 加大 k 不会有收益。marg 是「按分歧触发交互」方案的触发信号，必须
    # 先确认它在训练分布下不是常数 1.0，否则那条路一开始就没得选。
    pool_batch = [[
        ep(True,  n_votes=4, n_distinct=1, vote_margin=1.0),   # 全票一致 → 退化
        ep(False, n_votes=4, n_distinct=3, vote_margin=0.25),  # 有分歧
    ]]
    m = T_metrics(pool_batch)
    assert approx(m["pool_votes"], 4.0)
    assert approx(m["pool_distinct"], 2.0)        # (1+3)/2
    assert approx(m["pool_degenerate"], 0.5)      # n_distinct<=1 的占比
    assert approx(m["vote_margin"], 0.625)        # (1.0+0.25)/2

    # 缺字段时按 0 计（老 checkpoint / 评测脚本产的 episode 不能让聚合崩）
    m = T_metrics([[ep(True)]])
    assert approx(m["pool_votes"], 0.0) and approx(m["pool_degenerate"], 1.0)

    # 空输入不崩
    assert T_metrics([]) == {}


# ── v3.2：信道修复（剥块 + 放宽 flaw 窗口）────────────────────────────────

def test_strip_interaction_only_affects_display_copy():
    """块对**接收方**零信息量，必须在展示前剥掉；但训练目标一个字符都不能动。

    为什么这条测试是必需的：`strip_interaction` 用错地方就是一个静默的灾难。
    `record()` 存下的 `response` 与 `token_ids` / `log_probs` 逐位对齐，若把剥过
    块的文本存进去，per-token ratio 会整体错位——训练照跑、loss 照降，只是梯度
    对着错误的 token。所以这里同时钉住两侧：展示副本必须没有块，落库的
    `response` 必须仍带块。
    """
    from agents.parsing import strip_interaction

    blk = "<interaction>\naction: request\ntarget: critic\nreason: 不确定\n</interaction>"
    # 尾部（M1 之后的常态）、头部（M1 之前的历史数据）都要剥掉
    assert strip_interaction(f"错误分析：第二步错了\n{blk}") == "错误分析：第二步错了"
    assert strip_interaction(f"{blk}\n错误分析：第二步错了") == "错误分析：第二步错了"
    # 没有块时原样返回（只 strip 首尾空白）
    assert strip_interaction("错误分析：第二步错了") == "错误分析：第二步错了"

    # 端到端：critic 的意见进了 proposer 的修正 prompt，但块没跟着进去
    script = {
        "controller": [
            "<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
            "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>",
        ],
        "proposer": [
            "推理过程：x\n最终答案：5\n" + INTER.format(a="request", t="critic"),
            "推理过程：fix\n最终答案：4\n" + INTER.format(a="none", t="none"),
        ],
        "critic": ["错误分析：第二步把 6*7 算成了 41\n"
                   + INTER.format(a="request", t="proposer")],
        "verifier": ["分数: 0.9\n验证说明：ok\n" + INTER.format(a="none", t="none")],
    }
    eng = FakeEngine(script)
    # stop_gate=False：本测试关心的是「块有没有泄漏到下游 prompt」，不是终止逻辑。
    # 开着闸门时第 2 轮的 stop 会因无分数被拦下、多跑一轮，与本测试无关。
    res = _mk_executor(eng, stop_gate=False).run_episodes_batch(["6*7=?"], ["42"])[0]

    # 只看 user 部分。system 里本来就有 `<interaction>` 的格式说明（那是在教模型
    # 怎么写块），拿整个 prompt 去搜必然命中，断言会恒真——假绿。
    # FakeTokenizer 用 "\n---\n" 连接 system 与 user，据此切开。
    corr = [p for role, p in eng.prompts if role == "proposer"][1].split("\n---\n", 1)[1]
    assert "第二步把 6*7 算成了 41" in corr, "critic 的意见没送到修正 prompt"
    assert "<interaction>" not in corr, f"块泄漏进下游 prompt：{corr[-200:]!r}"

    # 训练目标一侧：critic 的 response 必须仍带块（否则 token 对齐被破坏）
    critic_msgs = [m for m in res["messages"] if m.get("role_name") == "critic"]
    assert critic_msgs, "没抓到 critic 的落库记录"
    assert "<interaction>" in critic_msgs[0]["response"], \
        "落库的 response 被剥了块——token_ids/log_probs 会错位"


def test_flaw_window_fits_a_realistic_critic_message():
    """flaw 窗口 80 → 300：v3「critic 说了等于没说」的直接成因。

    实测 critic 真报错时中位 267 字、p75 341 字。80 的窗口只有 4% 能完整送达，
    而块又固定占 68 字——真正传下去的实质内容约 11 字。这条测试用一段中位长度的
    批评钉住修复：剥块之后，300 的窗口要能把结论部分带过去。
    """
    from envs.blackboard import Blackboard, Message, MessageType

    bb = Blackboard()
    bb.add_message(Message(0, MessageType.TRACE, ("r", "5")))
    # 一段 ~250 字的批评，结论在末尾（真实 critic 就是这么写的：先复述再定位）
    concl = "，所以第二步的 41 应为 42"
    flaw = ("第一步把条件抄错了：题目给的是 6×7，" + "复述与推导过程" * 31 + concl)
    assert 250 <= len(flaw) <= 340, f"样例长度 {len(flaw)} 不在实测中位区间"
    bb.add_message(Message(1, MessageType.FLAW, {"content": flaw}))

    # 核心断言：结论句要真的出现在下游看到的文本里
    txt = bb.to_text()
    assert "发现问题" in txt
    assert concl in txt, "critic 的结论仍被窗口截掉了——信道没修好"
    # 反向对照：旧的 80 窗口连复述都没说完，结论一定丢
    assert concl not in flaw[:80]


def test_parser_label_tolerance():
    """半角冒号与英文别名：解析失败在这三处都不是「安全默认」而是静默污染。

    - proposer：抽不到答案会回落「文本里最后一个数字」，`最终答案: 24` 里的
      `45,045` 会变成 `045` 进投票池，各自占一票稀释多数投票；
    - verifier：回落 0.5 先验不是中性的，它系统性压制未验证的正确答案。
    critic 刻意**不给**英文别名：`无错误`/`错误分析` 是一对互补判据，单边加别名
    会让每条英文 critic 都被判成 flag，假 flag 直接污染 r^critic 与漏斗指标。
    """
    from agents.parsing import (critic_found_errors, has_answer_label,
                                parse_reasoning, parse_score)

    # 半角冒号（SFT 数据里真实出现过）
    assert parse_reasoning("推理过程：x\n最终答案: 24")[1] == "24"
    assert has_answer_label("最终答案: 24")
    assert parse_score("分数: 0.35") == 0.35
    # 全角冒号
    assert parse_score("分数：0.35") == 0.35
    # 英文别名
    assert parse_reasoning("推理过程：x\nFinal Answer: 24")[1] == "24"
    assert parse_score("Score: 1.0") == 1.0
    # 越界仍要夹回 [0,1]
    assert parse_score("Score: 1.7") == 1.0 and parse_score("分数: -2") == 0.0
    # 真的没有分数时仍返回 None（不能被别名放宽成「抓任意数字」）
    assert parse_score("验证说明：第二步可疑，涉及 42 与 6") is None
    # critic 不认英文——这是刻意的，不是遗漏
    assert critic_found_errors("Error analysis: step two is wrong") is False


def test_correction_prompt_does_not_duplicate_critic_text():
    """critic 硬触发修正时，黑板的「发现问题」与 initiator_output 是同一段文本。

    窗口从 80 放宽到 300 之后这份重复才显形：同一段批评在一个 prompt 里出现两次，
    白占 300 字预算（`max_prompt_tokens` ≈ 3008，且黑板文本嵌进每个角色的 prompt）。
    去重按**内容比对**而非「initiator 是不是 critic」——critic 未标错却主动请求
    修正时 `flaws[-1]` 是更早的另一条，那份信息是真的、不能扔。
    """
    script = {
        "controller": [
            "<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
            "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>",
        ],
        "proposer": [
            "推理过程：x\n最终答案：5\n" + INTER.format(a="request", t="critic"),
            "推理过程：fix\n最终答案：4\n" + INTER.format(a="none", t="none"),
        ],
        "critic": ["错误分析：第二步把 6*7 算成了 41\n"
                   + INTER.format(a="request", t="proposer")],
        "verifier": ["分数: 0.9\n验证说明：ok\n" + INTER.format(a="none", t="none")],
    }
    eng = FakeEngine(script)
    _mk_executor(eng, stop_gate=False).run_episodes_batch(["6*7=?"], ["42"])[0]

    corr = [p for role, p in eng.prompts if role == "proposer"][1]
    assert corr.count("第二步把 6*7 算成了 41") == 1, \
        f"critic 的意见在修正 prompt 里出现了 {corr.count('第二步把 6*7 算成了 41')} 次"


def test_replay_rebuilds_v2_prompts_into_rl_shape():
    """v2 的手写 `user` → RL 形状（`response` 一字不动）。

    v2 那批 SFT 数据连 `user` 一起是编出来的：模型被教着读一句人话摘要，而 RL
    给的是 `Blackboard.to_text()` 的结构化转储。实测 v2 的 1765 个 turn 里只有 5
    个含 `当前状态：`，其中 884 条 controller turn **全部**在 v2——也就是
    controller 一次都没在带黑板的 prompt 上被 SFT 过。
    """
    from data.prepare_sft import reorder_response
    from data.replay_v2_prompts import needs_replay, replay_row

    blk_req = ("<interaction>\naction: request\ntarget: critic\n"
               "reason: 请审查\n</interaction>")
    blk_none = ("<interaction>\naction: none\ntarget: none\n"
                "reason: 有把握\n</interaction>")
    row = {
        "question": "1+1=?", "answer": "2",
        "turns": [
            {"role_name": "controller", "system": "", "user": "现在开始解题。",
             "response": "<meta-plan>\ndecision: continue\nreason: 还没有解法\n"
                         "</meta-plan>"},
            {"role_name": "proposer", "system": "", "user": "请解这道题。",
             "response": f"{blk_req}\n推理过程：算错了，1+1=3\n最终答案：3"},
            {"role_name": "critic", "system": "",
             "user": "Proposer 给出答案 3，请审查。",
             "response": f"{blk_none}\n错误分析：1+1 应为 2，不是 3"},
            {"role_name": "proposer", "system": "",
             "user": "Critic 指出错误，请修正。",
             "response": f"{blk_none}\n推理过程：改回 1+1=2\n最终答案：2"},
        ],
    }
    assert needs_replay(row) is True
    orig_resp = [t["response"] for t in row["turns"]]

    for t in row["turns"]:                      # 阶段一：M1 重排（重放的前置）
        t["response"] = reorder_response(t["response"])[0]
    reordered = [t["response"] for t in row["turns"]]
    replay_row(row)

    # response 不能被重放动过（它是训练目标，与 token_ids 逐位对齐）
    assert [t["response"] for t in row["turns"]] == reordered
    # 块确实只是搬了位置，实质内容一字不改
    for old, new in zip(orig_resp, reordered):
        assert sorted(old) == sorted(new)

    users = [t["user"] for t in row["turns"]]
    assert all("当前状态：" in u for u in users)
    assert all("<interaction>" not in u for u in users)

    # controller 第一轮：黑板空
    assert users[0] == "问题：1+1=?\n当前状态：尚无信息"
    # proposer primary：同样是空黑板（flaw_in_primary=False 时无 flaw 段）
    assert users[1] == "问题：1+1=?\n当前状态：尚无信息"

    # critic：字面量是从 agentic_executor.py 的 responder_prompt 抄下来的。
    # 这里写死是有意的——重放器单方面改了字面量就会在这里炸。反方向（executor
    # 改了而重放没跟）挡不住，因为 executor 的 prompt 是内联 f-string，无法调用。
    assert users[2] == (
        "待审查解法：算错了，1+1=3\n答案：3\n"
        "当前状态：已有1个解法，答案：['3']\n"
        "最近交互：proposer→critic（request）\n"
        "协作者（Proposer）发起了交互：request，理由：请审查\n"
        "对方内容：推理过程：算错了，1+1=3\n最终答案：3\n"
        "请按你的标准输出格式回应。")

    # proposer 修正轮：点名 Critic，且黑板的「发现问题」被去重（同一段话已经
    # 在「你之前的解法被 Critic 指出问题：」里逐字引过了，再来一份纯占窗口）
    assert "你之前的解法被 Critic 指出问题：错误分析：1+1 应为 2" in users[3]
    assert users[3].count("1+1 应为 2") == 1, f"flaw 被重复塞了两份：{users[3]!r}"

    # 重放完就是 RL 形状了：再判一次不该还要重放（幂等）
    assert needs_replay(row) is False


def test_replay_skips_rows_already_in_rl_shape():
    """v3 那 1149 行本来就是按模板生成的，不能被碰。"""
    from data.replay_v2_prompts import needs_replay

    row = {"question": "2+2=?", "answer": "4", "turns": [
        {"role_name": "proposer", "system": "", "user": "问题：2+2=?\n当前状态：尚无信息",
         "response": "推理过程：2+2=4\n最终答案：4"}]}
    assert needs_replay(row) is False


def test_replay_refuses_to_invent_correction_context():
    """修正轮没有发起方 → 抛 Unreplayable 整行剔除，绝不凭空编一句。

    编出来就又回到 v2 的病：SFT 教模型在一句自造的上下文下作答。
    """
    from data.replay_v2_prompts import Unreplayable, replay_row

    row = {"question": "1+1=?", "answer": "2", "turns": [
        # proposer 声明 none → 后面那个 proposer 修正轮在 RL 里不可能发生
        {"role_name": "proposer", "system": "", "user": "手写摘要",
         "response": "推理过程：1+1=3\n最终答案：3\n<interaction>\naction: none\n"
                     "target: none\nreason: 有把握\n</interaction>"},
        {"role_name": "proposer", "system": "", "user": "手写摘要",
         "response": "推理过程：改成 2\n最终答案：2"},
    ]}
    try:
        replay_row(row)
    except Unreplayable as e:
        assert "缺发起方上下文" in str(e)
    else:
        raise AssertionError("修正轮缺发起方时必须抛 Unreplayable")


def test_prepare_sft_replays_before_filtering_turns():
    """派生链的阶段顺序：重放必须在剔除**之前**。

    这是最容易写错、且错了不会报错的一处。若先剔 turn 再重放，被剔掉那个 turn
    的产出就不会进黑板，后续 turn 看到的状态与 RL 真实呈现的不一致——等于用一份
    自造的状态去教模型。

    造法：中间放一条**英文打分**的 verifier turn。`keep_turn` 会剔掉它（判据是
    严格中文 `分数：`，因为 SFT 的职责就是教中文格式），但运行时 `parse_score`
    认得 `Score:`（宽容用于兜底），所以 RL 里这一分**确实上了黑板**。于是下一条
    controller 的 prompt 里必须有「最高置信答案」那一行——那行只可能来自这条被
    剔掉的 turn。顺序写反就没有这行。
    """
    import json
    import os
    import tempfile

    from data.prepare_sft import convert, verify

    row = {"question": "1+1=?", "answer": "2", "turns": [
        {"role_name": "proposer", "system": "", "user": "手写摘要",
         "response": "推理过程：1+1=2\n最终答案：2\n<interaction>\naction: request\n"
                     "target: verifier\nreason: 请验证\n</interaction>"},
        # 英文打分：SFT 判据剔除，运行时解析器认得 → 分数仍进黑板
        {"role_name": "verifier", "system": "", "user": "手写摘要",
         "response": "Score: 0.9\nVerification: looks right"},
        {"role_name": "controller", "system": "", "user": "手写摘要",
         "response": "<meta-plan>\ndecision: stop\nreason: 已有验证分数\n</meta-plan>"},
    ]}
    d = tempfile.mkdtemp()
    p_in = os.path.join(d, "in.jsonl")
    p_out = os.path.join(d, "out.jsonl")
    with open(p_in, "w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    st = convert(p_in, p_out)
    assert st["rows_replayed"] == 1 and st["dropped"] == 1, st
    verify(p_in, p_out)                       # 独立重推，逐字节对齐

    out = [json.loads(x) for x in open(p_out, encoding="utf-8") if x.strip()]
    assert len(out) == 1
    kept = out[0]["turns"]
    assert [t["role_name"] for t in kept] == ["proposer", "controller"]
    ctrl_user = kept[1]["user"]
    assert "最高置信答案：2（分数0.90）" in ctrl_user, (
        f"被剔掉的 verifier 那一分没进黑板——重放跑在剔除之后了？\n{ctrl_user!r}")
    # 顺带确认 verifier 的自发请求也留在了黑板上
    assert "最近交互：proposer→verifier（request）" in ctrl_user


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"All {len(fns)} v2 tests passed.")
