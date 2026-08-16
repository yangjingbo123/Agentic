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
       "c_int": 0.05, "lambda_int": 1.0}


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
    """u=1、primary 错、修正对：r_prop=0，r_int=−0.05+0.3=0.25。"""
    rounds = [make_round(0, primary="5", corrected="4", u=True, target="critic",
                         critic_turns=[{"tid": 2, "flagged": True,
                                        "reviewed_answer": "5",
                                        "correction_followed": True}],
                         correction_turns=[{"tid": 3, "answer": "4"}])]
    td, meta = compute_turn_data(rounds, "4", True, 4, CFG)
    assert approx(td[1]["reward"], 0.25)                     # proposer 主 turn
    assert approx(td[2]["reward"], 0.3)                      # critic 真阳性、本轮修对 q=1
    assert approx(td[3]["reward"], 1.0)                      # 修正响应：新答案对
    assert td[2]["is_response"] and td[3]["is_response"]
    assert not td[1]["is_response"]
    assert meta[0]["u"] and not meta[0]["p_primary"] and meta[0]["p_end"]


def test_r_int_useless_and_overkill():
    # 无效求助：错→仍错，r_int=−0.05+0 → reward=−0.05
    rounds = [make_round(0, primary="5", u=True, target="verifier",
                         verifier_turns=[{"tid": 2, "score": 0.2,
                                          "reviewed_answer": "5"}])]
    td, _ = compute_turn_data(rounds, "4", False, 4, CFG)
    assert approx(td[1]["reward"], -0.05)
    # 画蛇添足：对还求助，1.0 + (−0.05−0.2) = 0.75
    rounds = [make_round(0, primary="4", u=True, target="critic",
                         critic_turns=[{"tid": 2, "flagged": False,
                                        "reviewed_answer": "4",
                                        "correction_followed": False}])]
    td, _ = compute_turn_data(rounds, "4", True, 4, CFG)
    assert approx(td[1]["reward"], 0.75)
    assert approx(td[2]["reward"], 0.1)                      # critic 真阴性


def test_r_int_no_initiation_and_forced():
    # 不发起 + 对：1.0 + 0.1（正确的自信）
    td, _ = compute_turn_data([make_round(0, primary="4")], "4", True, 4, CFG)
    assert approx(td[1]["reward"], 1.1)
    # 不发起 + 错：0（机会成本不双计）
    td, _ = compute_turn_data([make_round(0, primary="5")], "4", False, 4, CFG)
    assert approx(td[1]["reward"], 0.0)
    # forced 注入：发起方不计 r_int → reward = r_prop
    rounds = [make_round(0, primary="5", forced=True,
                         critic_turns=[{"tid": 2, "flagged": True,
                                        "reviewed_answer": "5",
                                        "correction_followed": False}])]
    td, meta = compute_turn_data(rounds, "4", False, 4, CFG)
    assert approx(td[1]["reward"], 0.0)
    assert meta[0]["forced"] and not meta[0]["u"]


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
    assert approx(rewards[("proposer", 0, False)], 0.25)   # 0 + (−0.05+0.3)
    assert approx(rewards[("critic", 0, True)], 0.3)       # 真阳性且本轮修对
    assert approx(rewards[("proposer", 0, True)], 1.0)     # 修正响应
    assert approx(rewards[("verifier", 1, True)], 0.9)     # 审的是"4"（对）：1−|0.9−1|
    assert approx(rewards[("proposer", 1, False)], 0.75)   # 画蛇添足：1+(−0.05−0.2)
    # stop turn：t_stop=2, rem=0.5, 答对 → 1.15
    stop_r = [v["reward"] for v in td.values()
              if v["role"] == "controller" and v["round"] == 2]
    assert len(stop_r) == 1 and approx(stop_r[0], 1.15)
    # 脚本全部消耗（调用次数精确匹配预期流程）
    assert all(len(v) == 0 for v in eng.script.values()), eng.script
    # 记账对齐：log_probs 与 turn_ids 等长，每 turn 3 token
    assert len(res["log_probs"]) == len(res["turn_ids"]) == 3 * len(res["messages"])


def test_integration_forced_injection_and_ablation():
    # eps_force=1.0：proposer 不求助也会被强制注入 critic 审查
    script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"] * 4,
        "proposer":   [INTER.format(a="none", t="none") + "推理过程：x\n最终答案：5"] * 4,
        "critic":     [INTER.format(a="none", t="none") + "错误分析：有错"] * 4,
    }
    ex = _mk_executor(FakeEngine(script))
    res = ex.run_episodes_batch(["q"], ["4"], eps_force=1.0)[0]
    assert all(m["forced"] and not m["u"] for m in res["raca_round_meta"])
    # 强制轮：发起方 r_int=0 → proposer 主 turn 奖励 = 0.0
    prop_main = [v for v in res["raca_turn_data"].values()
                 if v["role"] == "proposer" and not v["is_response"]]
    assert all(approx(v["reward"], 0.0) for v in prop_main)

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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"All {len(fns)} v2 tests passed.")
