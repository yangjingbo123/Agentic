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


def test_vllm_worker_stops_on_each_roles_terminal_structure():
    """M1 后 worker 的 interaction 块位于末尾，闭标签必须成为生成 stop。"""
    from llm.vllm_worker import _stop_for_role

    assert _stop_for_role("controller") == ["</meta-plan>"]
    for role in ("proposer", "critic", "verifier"):
        assert _stop_for_role(role) == ["</interaction>"], role


def test_trailing_interaction_span_only_accepts_a_complete_final_block():
    """credit split 只认**末尾完整块**，正文示例/残块/块后正文都不能切。

    这条比 `parse_interaction` 故意更严格：解析器可以宽容地从全文找动作，但 token
    credit 一旦切错边界会把两门优势施加到错误 token，所以宁可返回 None。
    """
    from agents.parsing import trailing_interaction_span

    block = ("<interaction>\naction: request\ntarget: critic\n"
             "reason: 请审查\n</interaction>")
    text = "推理过程：x\n最终答案：4\n" + block + "  \n"
    span = trailing_interaction_span(text)
    assert span is not None
    start, end = span
    assert text[start:].lstrip().startswith("<interaction>")
    assert end == len(text)
    assert "最终答案：4" in text[:start]

    # 正文里引用一个块，但最后仍有真正的决策块：只能取最后一个。
    earlier = ("<interaction>\naction: challenge\ntarget: verifier\n"
               "reason: 仅是正文示例\n</interaction>")
    quoted = "示例：" + earlier + "\n最终答案：4\n" + block
    start2, _ = trailing_interaction_span(quoted)
    assert quoted[start2:].lstrip() == block
    assert quoted[:start2].count("<interaction>") == 1
    assert parse_interaction(quoted) == ("request", "critic", "请审查"), \
        "执行的动作与获得 interaction credit 的末尾块不是同一个"

    malformed_prefix = ("正文里误写 <interaction> 但没闭合\n最终答案：4\n" + block)
    start3, _ = trailing_interaction_span(malformed_prefix)
    assert malformed_prefix[start3:].lstrip() == block, \
        "前一个未闭合标签吞到了最终闭标签，credit span 选错块"
    assert parse_interaction(malformed_prefix) == ("request", "critic", "请审查")

    extra_close = "最终答案：4\n" + block + "\n</interaction>"
    assert trailing_interaction_span(extra_close) is None, \
        "重复闭标签使 credit span 延伸到 parser 实际执行块之外"

    assert trailing_interaction_span("正文 " + block + " 后面继续解释") is None
    assert trailing_interaction_span("正文 <interaction>\naction:none") is None
    assert trailing_interaction_span("只有答案：4") is None


def test_token_credit_components_route_channels_without_overlap():
    """solution / interaction 优势只能落到各自 token，且旧 scalar 保持整段兼容。"""
    from training.raca_adv import token_credit_components

    spec = {"solution": 1.5, "interaction": -0.25}
    # 原始位置 6 假设是被可见文本省略的 EOS：span=[4,6) 不应把它收进任一 PG 通道。
    parts = token_credit_components(spec, interaction_span=(4, 6),
                                    original_positions=[0, 1, 3, 4, 5, 6])
    assert parts == [
        ("solution", 1.5, [0, 1, 2]),
        ("interaction", -0.25, [3, 4]),
    ]
    covered = [idx for _, _, ids in parts for idx in ids]
    assert len(set(covered)) == len(covered), "两个通道 token 不能重叠"
    assert 5 not in covered, "文本外 EOS/special token 不应收到任一 PG credit"

    # 新显式双 span 可以在中间留下 gap；位置 3 是跨界 token，只接受 KL。
    gapped = token_credit_components(
        spec, {"solution": (0, 3), "interaction": (4, 6)}, range(7))
    assert gapped == [
        ("solution", 1.5, [0, 1, 2]),
        ("interaction", -0.25, [4, 5]),
    ]
    assert 3 not in [i for _, _, ids in gapped for i in ids]

    # interaction 缺失时仍保留 solution，不再因格式问题丢掉整条解题信号。
    solution_only = token_credit_components(
        spec, {"solution": (0, 5)}, range(7))
    assert solution_only == [("solution", 1.5, [0, 1, 2, 3, 4])]

    # forced：只有 solution，interaction block 与尾部 EOS 都没有 PG credit（仍有 KL）。
    forced = token_credit_components({"solution": -1.0}, (4, 6), range(7))
    assert forced == [("solution", -1.0, [0, 1, 2, 3])]

    # 旧格式无精确/越界边界时仍整 turn fail-closed；新显式 spans 才能安全降级。
    assert token_credit_components(spec, None, range(7)) == []
    assert token_credit_components(spec, (4, 99), range(7)) == [], \
        "span.end 超出原始 response_ids 仍被接受"
    # 旧 episode / 其它角色 scalar：整段广播，保持向后兼容。
    assert token_credit_components(0.7, None, [0, 2, 4]) == [
        ("default", 0.7, [0, 1, 2])]


def test_trainer_routes_channel_advantages_to_token_spans():
    """源码级不变量：torch 训练路径必须按 channel token span 算 PG。

    本地测试环境没有 torch/bitsandbytes，无法执行 `_compute_loss`；但如果这里只测
    `token_credit_components`，训练器仍可能把返回值忽略、继续用旧的 whole-turn
    scalar。故用 AST/源码守住接线，并通过变异验证确认它承重。
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent /
           "training/grpo_trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert any(getattr(n.func, "id", "") == "token_credit_components" for n in calls), \
        "训练器没有消费结构化 channel advantage"
    assert any(getattr(n.func, "attr", "") == "index_select" for n in calls), \
        "训练器没有按 token 索引切分 solution / interaction span"
    assert "pg_parts.append" in src and "torch.stack(pg_parts).sum()" in src, \
        "两个通道没有先各自取 token mean 再相加"
    assert "-advantage * ratio_tok" not in src, \
        "旧的 whole-turn scalar 广播仍在，会让 solution / interaction 信号再次混合"
    assert 'len(ids) != lp_count.get(tid, 0)' in src, \
        "预计数没检查 response/logprob 长度，实际跳过的 turn 仍会稀释 normalization"
    assert "any(tok < 0 or tok >= vocab_size for tok in ids)" in src, \
        "非法 token 仍可能在预计数中算有效、训练时再被跳过"
    assert src.count('msg.get("logprob_aligned", True)') >= 2, \
        "预计数与实际 loss 没有同时拒绝被 0.0 补齐的假 logprob"


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
    """v2.1 核心回归：不发起**且答对**一律 0，不得给无条件补贴。

    v2.0 给“不发起+答对”+0.1，这笔补贴与画蛇添足惩罚共同使 p>0.5 时
    发起的边缘期望恒为负（即使 q=1.0）→ int_rate 必然塌到 0。

    **这条测试有一处历史局限，值得写下来当教训**：下面那段"边缘期望校准"用的是
    **假设的** q=0.5，而 08-28 两跑实测 q（=`eff`）只有 **0.12**。于是这几条断言
    一直全绿，而真实的边缘期望是负的、int_rate 在 40 步内塌到 0.01。
    **它验证的是设计意图，不是运行现实** —— 把假设值写进断言，等于只证明了
    "如果世界如我所愿，则设计成立"。这和本仓库反复栽的"两把尺子"是同一类错误，
    只不过尺子这次量的是一个还没发生的世界。
    实测 q 那一档的判据现在由 `test_int_reward_zero_crossing_is_inside_operating_range`
    承担，那条按 `configs/agentic/default.yaml` 的**活配置** + **实测 q** 算零点。
    """
    # 不发起 + 对：只有 r_prop=1.0，无额外奖励
    td, _ = compute_turn_data([make_round(0, primary="4")], "4", True, 4, CFG)
    assert approx(td[1]["reward"], 1.0)
    assert td[1]["layer_key"] == 1
    # 不发起 + 错：0 —— **仅在 int_miss=0（消融）时成立**。CFG 刻意不带 int_miss，
    # 所以这里走的是 v2.1 的退化路径。原注释写的"机会成本已由 r_prop=0 体现，不
    # 双重计罚"**已被第十二轮推翻**：r_prop 对「问不问」没有区分度（答错就是 0，
    # 问了也 0、不问也 0），给发起决策的梯度恒为 0。启用后的行为由
    # `test_r_int_miss_prices_inaction` 守。
    assert CFG.get("int_miss", 0.0) == 0.0, "本行断言的前提是 miss 被消融"
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


def test_r_int_miss_prices_inaction():
    """`int_miss`：「不发起 + 答错」从 0 改为 −miss，给不作为定价。

    为什么需要它（08-28 两跑实测）：int_rate 从 step1 的 0.75 在 40 步内塌到 0.01，
    两跑一致、`sel` 从未转负。根因是矩阵里**不作为零成本**——求助犯的错（画蛇添足）
    扣 0.05，不求助犯的错（该问而没问）扣 0，于是从发起决策的视角看"错而不问"与
    "对而不问"完全等价，没有任何压力去问。

    四格必须逐个钉住，因为只改对角线的任何一格都会破坏双向压力：
      不发起+对 → 0        （penalty-only，绝不为不作为发奖；v2.0 那笔补贴已证伪）
      不发起+错 → −miss    （本轮新增）
      求助+对   → −overkill（保持！否则塌向"全都问"，v3 已实测 sel≈0）
      forced    → 0        （决策不是发起方做的，miss 不该影响它）
    """
    M = 0.10
    cfg_on = {**CFG, "int_miss": M}

    # 不发起 + 错 → r_prop(0) + λ·(−miss)
    td, _ = compute_turn_data([make_round(0, primary="5")], "4", False, 4, cfg_on)
    assert approx(td[1]["reward"], -M), \
        f"不发起+答错应罚 −{M}，实测 {td[1]['reward']:+.4f}"
    # 不发起 + 对 → 仍是 1.0，**miss 不许溢到这一格**（那就变成罚"答对"了）
    td, _ = compute_turn_data([make_round(0, primary="4")], "4", True, 4, cfg_on)
    assert approx(td[1]["reward"], 1.0), "miss 溢到了「不发起+答对」格"
    # forced 轮不受影响：决策不是发起方做的
    td, _ = compute_turn_data(
        [make_round(0, primary="5", forced=True)], "4", False, 4, cfg_on)
    assert approx(td[1]["reward"], 0.0), "forced 轮不该被 miss 罚"
    # 画蛇添足惩罚必须还在（否则双向压力塌成单向）
    td, _ = compute_turn_data(
        [make_round(0, primary="4", u=True, target="critic")], "4", True, 4, cfg_on)
    assert approx(td[1]["reward"], 1.0 - CFG["c_int"] - CFG["int_overkill"]), \
        "求助+答对 的画蛇添足惩罚被 miss 改动波及了"
    # miss=0 必须与启用前**逐位等价**（这一项要可消融）
    for prim, ok in (("5", False), ("4", True)):
        a, _ = compute_turn_data([make_round(0, primary=prim)], "4", ok, 4, CFG)
        b, _ = compute_turn_data([make_round(0, primary=prim)], "4", ok, 4,
                                 {**CFG, "int_miss": 0.0})
        assert approx(a[1]["reward"], b[1]["reward"]), "miss=0 不等价于消融"


def test_int_reward_zero_crossing_is_inside_operating_range():
    """边缘期望的零点必须落在**模型真实的工作区间**内，否则 int_rate 必然塌。

    这条测试存在的理由，是上面 `test_r_int_no_unconditional_subsidy` 那个教训：
    它用**假设的** q=0.5 校准边缘期望，于是一直全绿，而实测 q=0.12 时边缘期望是
    负的、int_rate 在 40 步内塌到 0.01。**所以这条改用实测值，并读活配置。**

    边缘期望（c=c_int, A=int_gain, B=int_overkill, M=int_miss, x=P(错)）：
        E[求助]   = x·q·A + (1−x)·(−B) − c
        E[不求助] = x·(−M)
        差 = x·(q·A + B + M) − B − c
    零点：x* = (B + c) / (q·A + B + M)
    x > x* 时求助划算。**要求 x* 明显低于工作区间下沿**，否则策略会先学会闭嘴。

    实测输入（08-28 两跑）：q = eff = 0.12；轮级 P(错) ≈ 0.35~0.45（由 acc 反推，
    区间偏宽正是因为 `p_primary_rate` 此前没埋点——同一轮已补上，下一跑可核准）。
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent
    y = (root / "configs/agentic/default.yaml").read_text(encoding="utf-8")

    def cfgnum(key):
        m = re.search(rf'^{key}:\s*([\d.]+)', y, re.M)
        assert m, f"default.yaml 里找不到 {key}"
        return float(m.group(1))

    c, A, B, M = (cfgnum("c_int"), cfgnum("int_gain"),
                  cfgnum("int_overkill"), cfgnum("int_miss"))
    Q_MEASURED = 0.12          # 08-28 两跑的 eff（对照 0.118 / 打开 0.123）
    X_LOW = 0.35               # 工作区间下沿（轮级答错率的保守估计）

    def crossing(miss):
        return (B + c) / (Q_MEASURED * A + B + miss)

    x_star = crossing(M)
    assert x_star < X_LOW, (
        f"边缘期望零点 x*={x_star:.3f} 没有落在工作区间（下沿 {X_LOW}）之内 —— "
        f"策略会先学会不求助。当前 c={c} A={A} B={B} M={M}，q(实测)={Q_MEASURED}。"
        f"要么抬 int_miss，要么先用 p_primary_rate 核准工作区间。")

    # 反向：把 miss 消融回 0，零点必须落在工作区间**之外** ——
    # 这一条把 08-28 实测的崩塌钉成回归。它不是在测代码，是在钉住"为什么要有 miss"。
    x0 = crossing(0.0)
    assert x0 > 0.5, (
        f"miss=0 时零点应在 0.5 以上（08-28 实测 0.581，对应 int_rate 40 步塌到 "
        f"0.01），实测 {x0:.3f} —— 若这条失败说明 gain/overkill 被人动过，"
        f"那 miss 的取值理由需要重算")
    # 且启用 miss 必须把零点显著往下推，而不是聊胜于无
    assert x0 - x_star > 0.2, \
        f"miss 只把零点从 {x0:.3f} 推到 {x_star:.3f}，幅度不足以脱离工作区间"

    # 条件方向不许被 miss 破坏：自认为错时求助更优、自认为对时不求助更优
    assert (Q_MEASURED * A) > (-M), "自认为答错时，求助必须优于不求助"
    assert (-c - B) < 0.0, "自认为答对时，求助必须不划算（双向压力的另一半）"


def test_p_primary_rate_is_round_level_not_episode_accuracy():
    """`p_primary_rate` 必须报出来，且必须是**轮级首答**正确率，不是 episode 的 acc。

    它是 `int_miss` 零点位置的唯一直接输入（零点 = (B+c)/(q·A+B+M) 与工作区间
    P(错)=1−p_primary_rate 比较）。此前这个数**一直被算出来却从来没报过** ——
    `ps` 就是它，只用来算 `sel` 的相关系数 —— 所以 08-28 定 miss 时只能从 acc
    反推出 0.35~0.45 这么宽的区间。

    这条测试专门防一种"化简"：有人看到 `accuracy` 也在，就把这一项接到那上面。
    两者差十几个点（acc 是**投票之后**的 episode 结果，这个是逐轮 proposer 首答），
    接错了零点就算错，而错法是静默的——数照样打印。所以构造一个两者必然不等的场景。
    """
    from training.metrics import rollout_metrics

    # 两轮：一轮首答对、一轮首答错 → 轮级 p_primary_rate 必须是 0.5
    rounds = [make_round(0, primary="4"), make_round(1, primary="5")]
    _, meta = compute_turn_data(rounds, "4", True, 4, CFG)
    # episode 层面刻意标成 is_correct=True（投票救回来了）→ 与 0.5 必然不等
    st = rollout_metrics([[{"is_correct": True, "raca_round_meta": meta,
                            "raca_turn_data": {}, "stopped": True}]])
    assert "p_primary_rate" in st, \
        "p_primary_rate 没被报出来 —— int_miss 的零点位置就没有直接输入了"
    assert approx(st["p_primary_rate"], 0.5), \
        f"应是轮级均值 0.5，实测 {st['p_primary_rate']}"
    assert st["p_primary_rate"] != st.get("accuracy", 1.0), \
        "p_primary_rate 被接到了 episode 级 accuracy 上（两者相差十几个点）"


def test_r_int_no_initiation_and_forced():
    """forced 保留 r_prop，但必须用 `r_int_w=None` 完全退出交互优势通道。

    旧测试只断 total reward 等于 r_prop，以为 `r_int=0` 就是“不奖不罚”。
    `v32_miss` 的 50 步反证了它：r_int 按 p_primary 分层 z-score，数值 0 仍参与
    排名；在首答错层里 0 高于 −miss，给模型实际输出的 `action:none` 正优势。
    所以总 reward 断言还要保留（历史口径不变），但真正承重的是 `r_int_w is None`。
    """
    # forced + 答错：总 reward 仍只有 r_prop=0，但 int 通道必须缺席
    rounds = [make_round(0, primary="5", forced=True,
                         critic_turns=[{"tid": 2, "flagged": True,
                                        "reviewed_answer": "5",
                                        "correction_followed": False}])]
    td, meta = compute_turn_data(rounds, "4", False, 4, CFG)
    assert approx(td[1]["reward"], 0.0)
    assert td[1]["r_int_w"] is None, \
        "forced 写成数值 0 仍会进入 z-score；必须是 None（通道缺席）"
    assert 2 in td, "forced critic 响应 turn 被整条跳过了"
    assert approx(td[2]["reward"], 0.2), \
        "forced 只应屏蔽 proposer 的 int 通道，critic 响应奖励必须保留"
    assert meta[0]["forced"] and not meta[0]["u"]
    # forced + 答对：仍只有 r_prop=1；同样不进 int 通道
    rounds = [make_round(0, primary="4", forced=True,
                         verifier_turns=[{"tid": 2, "score": 0.9,
                                          "reviewed_answer": "4"}])]
    td, _ = compute_turn_data(rounds, "4", True, 4, CFG)
    assert approx(td[1]["reward"], 1.0)
    assert td[1]["r_prop"] == 1.0 and td[1]["r_int_w"] is None
    assert 2 in td, "forced verifier 响应 turn 被整条跳过了"
    assert approx(td[2]["reward"], 0.9), \
        "forced verifier 的校准奖励必须保留"

    # q_forced 随机对照必须保留：forced 只退出优势，不退出 round_meta / metrics。
    # 构造一条 forced 后错→对，q_forced 应为 1.0。
    fixed = [make_round(
        0, primary="5", corrected="4", forced=True,
        correction_turns=[{"tid": 3, "answer": "4"}],
    )]
    td_fixed, meta_fixed = compute_turn_data(fixed, "4", True, 4, CFG)
    from training.metrics import rollout_metrics
    st = rollout_metrics([[{"is_correct": True, "raca_round_meta": meta_fixed,
                            "raca_turn_data": td_fixed, "stopped": True}]])
    assert approx(st["q_forced"], 1.0), \
        "forced 退出 int 优势后 q_forced 随机对照被误删了"

    # 用**真实 compute_turn_data 产物**走进 advantage，而不是只测手搓字典：
    # 一条 forced 错、一条 forced 对，二者 int 通道都缺席，但 r_prop 仍给 ±1。
    wrong = [make_round(0, primary="5", forced=True)]
    td_wrong, _ = compute_turn_data(wrong, "4", False, 4, CFG)
    adv = compute_raca_advantages([td_wrong, td])
    assert 1 in adv[0] and 1 in adv[1], \
        "真实 forced turn 连 r_prop 优势也丢了（只该退出 int 通道）"
    assert set(adv[0][1]) == set(adv[1][1]) == {"solution"}
    assert approx(adv[0][1]["solution"], -1.0)
    assert approx(adv[1][1]["solution"], 1.0)

    # 同一个 p_t=0 anchor 里混入 spontaneous none / spontaneous request / forced。
    # forced 若退化成 raw r_int=0，就会与 request 一起高于 none=-0.1，并给模型
    # 实际输出的 action:none 正优势；正确实现中它必须完全没有 interaction key。
    cfg_miss = {**CFG, "c_int": 0.0, "int_miss": 0.10}
    spontaneous_none = [make_round(0, primary="5")]
    spontaneous_ask = [make_round(0, primary="5", u=True, target="critic")]
    forced_none = [make_round(0, primary="5", forced=True, target="critic")]
    td_none, _ = compute_turn_data(spontaneous_none, "4", False, 4, cfg_miss)
    td_ask, _ = compute_turn_data(spontaneous_ask, "4", False, 4, cfg_miss)
    td_forced, _ = compute_turn_data(forced_none, "4", False, 4, cfg_miss)
    mixed = compute_raca_advantages([td_none, td_ask, td_forced])
    assert mixed[0][1]["interaction"] < 0 < mixed[1][1]["interaction"]
    assert 1 not in mixed[2] or "interaction" not in mixed[2][1], \
        "forced 数值 0 泄漏回 int anchor group，正在奖励 action:none"


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


def test_lambda_int_weights_the_standardized_interaction_advantage():
    """lambda_int 必须乘在 z-score **之后**，否则任何正值都被尺度不变性消掉。

    旧实现把 `r_int_w=lambda*r_int` 送入标准化，因此
    `z(0.25*r)==z(2*r)`，配置里的 0.25/1/2 实际完全相同。这里构造两个 raw
    interaction reward（−0.1 / +0.3），solution reward 相同而零方差，断言最终
    interaction advantage 随 lambda 线性缩放，而不是永远 ±1。
    """
    def run(weight):
        def turn(r_int):
            v = mk_turn("proposer", 0, "verify", False, r_int)
            v.update({"r_prop": 0.0, "r_int": r_int,
                      "r_int_w": weight * r_int, "lambda_int": weight,
                      "layer_key": 0, "token_credit": True})
            return {1: v}
        return compute_raca_advantages([turn(-0.1), turn(0.3)])

    zero, low, high = run(0.0), run(0.25), run(2.0)
    assert zero == [{}, {}], "lambda_int=0 必须精确关闭 interaction 通道"
    for adv in (low, high):
        assert set(adv[0][1]) == set(adv[1][1]) == {"interaction"}, adv
    assert approx(low[0][1]["interaction"], -0.25)
    assert approx(low[1][1]["interaction"], +0.25)
    assert approx(high[0][1]["interaction"], -2.0)
    assert approx(high[1][1]["interaction"], +2.0)
    assert approx(abs(high[0][1]["interaction"] /
                      low[0][1]["interaction"]), 8.0), \
        "lambda 在标准化前被消掉了：2.0/0.25 应让优势强度相差 8 倍"


def test_structured_proposer_advantages_do_not_cancel_before_token_routing():
    """solution 与 interaction 两门成绩必须保持分离，不能先相加成一个 turn 标量。

    构造「答案好但交互坏」与「答案坏但交互好」：两者若先相加都会变成 0，
    正确答案得不到强化、错误求助也不受罚。结构化返回应分别保留 ±1，供训练侧
    路由到不同 token span。
    """
    def turn(r_prop, r_int, p_t):
        v = mk_turn("proposer", 0, "verify", False, r_prop + r_int)
        v.update({"r_prop": r_prop, "r_int": r_int, "r_int_w": r_int,
                  "lambda_int": 1.0, "layer_key": p_t, "token_credit": True})
        return {1: v}

    # solution 通道要有对/错差异；每个 p_t 层内 interaction 也各有好/坏比较。
    eps = [
        turn(1.0, 0.0, 1),      # 对 + 不问（两门都好）
        turn(1.0, -0.05, 1),    # 对 + 多问（solution好 / interaction坏）
        turn(0.0, 0.30, 0),     # 错 + 求助修对（solution坏 / interaction好）
        turn(0.0, -0.10, 0),    # 错 + 不问（两门都坏）
    ]
    adv = compute_raca_advantages(eps)
    assert adv[1][1]["solution"] > 0 > adv[1][1]["interaction"]
    assert adv[2][1]["solution"] < 0 < adv[2][1]["interaction"]
    assert set(adv[1][1]) == {"solution", "interaction"}
    assert set(adv[2][1]) == {"solution", "interaction"}


def test_forced_turn_exits_int_advantage_but_keeps_prop_advantage():
    """forced proposer 不进入 int anchor group，但仍然正常进入 r_prop 通道。

    这是第十四轮的核心测试。只断 `r_int_w is None` 还不够：优势聚合若忘了识别
    None，仍可能把它塞进组里（报错或被误当 0）；反过来若粗暴跳过整个 turn，
    r_prop 也丢了，解题能力会停训。两种坏法要分别造场景抓住。

    场景 A（全是首答错，r_prop 全 0）：
      自发不求助 −0.10、主动求助但未修对 0、forced None。
    正确结果：int 通道只比较前两者（±1），forced 完全没有优势条目。旧代码把
    forced 写成 0 时，它会与主动求助并列高于 −0.10，从而给 action:none 正优势。

    场景 B（只有 forced，一对一错）：int 通道两条都缺席，但 r_prop={0,1} 仍应
    产生 ±1，证明我们跳过的是**通道**而不是整个 proposer turn。
    """
    def prop(r_prop, r_int_w, p_t):
        v = mk_turn("proposer", 0, "explore", False,
                    r_prop + (r_int_w if r_int_w is not None else 0.0))
        v["r_prop"], v["r_int_w"], v["layer_key"] = r_prop, r_int_w, p_t
        return v

    # A：forced 必须完全退出 int 通道
    eps = [
        {1: prop(0.0, -0.10, 0)},   # 自发 none + 错 → miss
        {1: prop(0.0, 0.00, 0)},    # 自发求助但未修对
        {1: prop(0.0, None, 0)},    # forced：模型也写 none，但决策不是它做的
    ]
    adv = compute_raca_advantages(eps)
    assert approx(adv[0][1], -1.0) and approx(adv[1][1], 1.0)
    assert 1 not in adv[2], (
        "forced 获得了 int 优势 —— 数值 0 仍在组内排名；必须用 None 完全退出通道")

    # B：forced 仍保留 r_prop 通道（不能粗暴跳过整个 turn）
    forced_only = [
        {1: prop(0.0, None, 0)},
        {1: prop(1.0, None, 1)},
    ]
    adv = compute_raca_advantages(forced_only)
    assert all(1 in ep for ep in adv), \
        "forced 整个 proposer turn 被跳过了；只应退出 int 通道，r_prop 必须保留"
    assert approx(adv[0][1], -1.0) and approx(adv[1][1], 1.0), \
        "forced 连 r_prop 也被跳过了 —— 只该退出 int 通道，解题能力必须继续训练"


def test_dual_channel_keeps_prop_training_when_int_channel_flat():
    """v2.3：int_rate=0（r_int 全 0）时 int 通道失活，但 r_prop 通道仍供梯度。

    v2.1/v2.2 的全量分层在此场景下组内零方差 → 主 turn 整组被丢 →
    解题能力停训（v2.2 首跑实测 acc 横盘、len 组成坍缩）。

    旧函数名叫 `test_dual_channel_no_absorbing_state`，**说过头了**：它只证明
    r_prop 仍有梯度、解题能力不停训，并不证明交互动作能从 int_rate=0 恢复。
    08-28 两跑已实测交互率到 0 后 160 步不恢复，交互决策仍是吸收态。
    """
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
    # proposer 主 turn：两者首答都错，solution 通道零方差被丢；interaction 通道
    # 比较「求助并修对」与「不求助仍错」，返回结构化 ±1。
    assert set(adv[0][1]) == set(adv[1][1]) == {"interaction"}
    assert adv[0][1]["interaction"] > 0 > adv[1][1]["interaction"]


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
        self.temps = []            # 每次请求的 temperature（eval_mode 应恒 0.0）

    def generate_batch(self, requests):
        out = []
        for r in requests:
            self.calls.append(r["role"])
            self.prompts.append((r["role"], r["prompt"]))
            self.temps.append(r.get("temperature"))
            text = self.script[r["role"]].pop(0)
            out.append((text, [-0.5, -0.5, -0.5], [11, 12, 13]))
        return out


def _mk_executor(engine, eval_mode=False, tokenizer=None, **cfg_over):
    from agents.agentic_executor import AgenticExecutor
    cfg = {"max_rounds": 4, "max_hops": 2, "stop_gate": True, **CFG, **cfg_over}
    # eval_mode / tokenizer 是构造参数而非 cfg 项，不能混进 cfg_over
    return AgenticExecutor(None, tokenizer or FakeTokenizer(), cfg,
                           vllm_engine=engine, eval_mode=eval_mode)


INTER = "<interaction>\naction: {a}\ntarget: {t}\nreason: r\n</interaction>\n"


def test_executor_records_exact_interaction_token_boundary():
    """primary proposer 的 interaction_start 必须直接索引原始 response_ids。

    用字符充当 token，使预期边界可以逐位验证；这条同时守住三件事：response 文本
    与 IDs 不被重写、只有 primary proposer 带 credit spans、interaction 缺失时
    保留 solution 并留下计数。真实 tokenizer 还会走原始 IDs 累计 decode 回退。
    """
    class CharEngine(FakeEngine):
        def generate_batch(self, requests):
            out = []
            for r in requests:
                self.calls.append(r["role"])
                self.prompts.append((r["role"], r["prompt"]))
                self.temps.append(r.get("temperature"))
                text = self.script[r["role"]].pop(0)
                out.append((text, [-0.5] * len(text), list(text)))
            return out

    block = INTER.format(a="none", t="none").strip()
    response = "推理过程：1+1=2\n最终答案：2\n" + block + "\n"
    script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
                       "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>"],
        "proposer": [response], "critic": [], "verifier": [],
    }
    eng = CharEngine(script)
    ex = _mk_executor(eng, stop_gate=False, max_rounds=2)
    result = ex.run_episodes_batch(["1+1=?"], ["2"])[0]
    primary = [m for m in result["messages"] if m["role_name"] == "proposer"][0]
    span = primary["interaction_span"]
    assert isinstance(span, tuple) and 0 < span[0] < span[1]
    assert primary["response_ids"] == list(response), "原始 vLLM IDs 被重写了"
    assert "".join(primary["response_ids"][:span[0]]).rstrip().endswith("最终答案：2")
    assert "".join(primary["response_ids"][span[0]:span[1]]).lstrip().startswith(
        "<interaction>")
    assert span[1] == len(response), "可见文本的末端 token 边界不对"
    assert ex.n_credit_split_failed == 0

    # vLLM 可能在 IDs 末尾附带被文本 decode 隐去的 EOS。它可以参与 KL，但不属于
    # solution 或 interaction 内容，故 interaction_span.end 必须停在可见文本末端。
    class SpecialTokenizer(FakeTokenizer):
        all_special_ids = [999]

        def __call__(self, text, add_special_tokens=False,
                     return_offsets_mapping=False):
            assert return_offsets_mapping
            return {"input_ids": list(text),
                    "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    class SpecialEngine(CharEngine):
        def generate_batch(self, requests):
            out = []
            for r in requests:
                self.calls.append(r["role"])
                self.prompts.append((r["role"], r["prompt"]))
                self.temps.append(r.get("temperature"))
                text = self.script[r["role"]].pop(0)
                ids = list(text) + [999]
                out.append((text, [-0.5] * len(ids), ids))
            return out

    eos_script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
                       "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>"],
        "proposer": [response], "critic": [], "verifier": [],
    }
    eos_ex = _mk_executor(SpecialEngine(eos_script), stop_gate=False, max_rounds=2,
                          tokenizer=SpecialTokenizer())
    eos_res = eos_ex.run_episodes_batch(["1+1=?"], ["2"])[0]
    eos_msg = [m for m in eos_res["messages"] if m["role_name"] == "proposer"][0]
    assert eos_msg["interaction_span"][1] == len(response)
    assert len(eos_msg["response_ids"]) == len(response) + 1

    # 无完整末尾块：interaction 缺席并计数，但 solution 仍继续训练。
    bad_script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
                       "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>"],
        "proposer": ["推理过程：1+1=2\n最终答案：2"],
        "critic": [], "verifier": [],
    }
    bad_ex = _mk_executor(CharEngine(bad_script), stop_gate=False, max_rounds=2)
    bad = bad_ex.run_episodes_batch(["1+1=?"], ["2"])[0]
    bad_primary = [m for m in bad["messages"] if m["role_name"] == "proposer"][0]
    assert bad_primary["interaction_span"] is None
    assert bad_primary["interaction_span_error"] == "no_close_tag"
    assert bad_primary["credit_spans"] == {
        "solution": (0, len(bad_primary["response_ids"]))}, \
        "缺 interaction 标签时 solution credit 没有保留"
    assert bad_ex.n_credit_split_failed == 1
    assert bad_ex.n_credit_split_failures == {"no_close_tag": 1}
    from training.raca_adv import token_credit_components
    parts = token_credit_components(
        {"solution": 1.0, "interaction": -1.0},
        bad_primary["credit_spans"], range(len(bad_primary["response_ids"])))
    assert [name for name, _, _ in parts] == ["solution"]

    # logprob 少于 token_ids 时，扁平数组仍补位以免后续 turn 错位，但 message 必须
    # 标成无效，训练器会整 turn 跳过；不能把补出的 0.0 当成真实 old logprob。
    class MismatchEngine(CharEngine):
        def generate_batch(self, requests):
            out = []
            for r in requests:
                self.calls.append(r["role"])
                self.prompts.append((r["role"], r["prompt"]))
                self.temps.append(r.get("temperature"))
                text = self.script[r["role"]].pop(0)
                ids = list(text)
                lps = ([-0.5] if r["role"] == "proposer" else [-0.5] * len(ids))
                out.append((text, lps, ids))
            return out

    mm_script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
                       "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>"],
        "proposer": [response], "critic": [], "verifier": [],
    }
    mm_ex = _mk_executor(MismatchEngine(mm_script), stop_gate=False, max_rounds=2)
    mm = mm_ex.run_episodes_batch(["1+1=?"], ["2"])[0]
    mm_primary = [m for m in mm["messages"] if m["role_name"] == "proposer"][0]
    assert mm_primary["logprob_aligned"] is False
    assert mm_ex.n_logprob_mismatch == 1
    assert len(mm["log_probs"]) == len(mm["turn_ids"]), \
        "即使 turn 无效，扁平占位也必须保持后续 turn 的全局索引对齐"


def test_cross_boundary_token_is_excluded_instead_of_dropping_turn():
    """边界穿过一个 BPE token 时，双通道保留、只排除该 token。"""
    block = INTER.format(a="none", t="none").strip()
    response = "推理过程：1+1=2\n最终答案：2\n" + block
    from agents.parsing import trailing_interaction_span
    char_start, _ = trailing_interaction_span(response)
    response_ids = [101, 102, 103]

    class CrossingTokenizer(FakeTokenizer):
        all_special_ids = []

        def __call__(self, text, add_special_tokens=False,
                     return_offsets_mapping=False):
            assert text == response and return_offsets_mapping
            return {
                "input_ids": response_ids,
                "offset_mapping": [
                    (0, char_start - 1),
                    (char_start - 1, char_start + 1),
                    (char_start + 1, len(text)),
                ],
            }

    class CrossingEngine(FakeEngine):
        def generate_batch(self, requests):
            out = []
            for req in requests:
                self.calls.append(req["role"])
                self.prompts.append((req["role"], req["prompt"]))
                self.temps.append(req.get("temperature"))
                text = self.script[req["role"]].pop(0)
                ids = response_ids if req["role"] == "proposer" else list(text)
                out.append((text, [-0.5] * len(ids), ids))
            return out

    script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"],
        "proposer": [response], "critic": [], "verifier": [],
    }
    ex = _mk_executor(
        CrossingEngine(script), tokenizer=CrossingTokenizer(),
        stop_gate=False, max_rounds=1, max_hops=0)
    ep = ex.run_episodes_batch(["1+1=?"], ["2"])[0]
    msg = next(m for m in ep["messages"] if m["role_name"] == "proposer")
    assert msg["credit_spans"] == {
        "solution": (0, 1), "interaction": (2, 3)}
    assert msg["credit_boundary_gap_tokens"] == 1
    assert ex.n_credit_boundary_tokens == 1
    assert ex.n_credit_split_failed == 0


def test_original_token_decode_recovers_noncanonical_segmentation():
    """重新 encode 的 IDs 不同时，原始采样 IDs 的累计 decode 仍能定位边界。"""
    block = INTER.format(a="none", t="none").strip()
    response = "推理过程：x\n最终答案：2\n" + block
    from agents.parsing import trailing_interaction_span
    char_start, _ = trailing_interaction_span(response)
    original_ids = [201, 202, 203]

    class NonCanonicalTokenizer(FakeTokenizer):
        all_special_ids = []

        def __call__(self, text, add_special_tokens=False,
                     return_offsets_mapping=False):
            assert text == response and return_offsets_mapping
            return {
                "input_ids": [1, 2, 3],
                "offset_mapping": [(0, 1), (1, 2), (2, len(text))],
            }

        def encode(self, text, add_special_tokens=False):
            if text == response or text == response[char_start:]:
                return [1, 2, 3, 4]  # 故意无法匹配原始 IDs 或其后缀
            return list(text)

        def decode(self, ids, **kwargs):
            table = {
                (): "",
                (201,): response[:char_start],
                (201, 202): response[:char_start] + "<interaction>",
                (201, 202, 203): response,
            }
            key = tuple(ids)
            if key in table:
                return table[key]
            return "".join(str(x) for x in ids)

    class NonCanonicalEngine(FakeEngine):
        def generate_batch(self, requests):
            out = []
            for req in requests:
                self.calls.append(req["role"])
                self.prompts.append((req["role"], req["prompt"]))
                self.temps.append(req.get("temperature"))
                text = self.script[req["role"]].pop(0)
                ids = original_ids if req["role"] == "proposer" else list(text)
                out.append((text, [-0.5] * len(ids), ids))
            return out

    script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"],
        "proposer": [response], "critic": [], "verifier": [],
    }
    ex = _mk_executor(
        NonCanonicalEngine(script), tokenizer=NonCanonicalTokenizer(),
        stop_gate=False, max_rounds=1, max_hops=0)
    ep = ex.run_episodes_batch(["1+1=?"], ["2"])[0]
    msg = next(m for m in ep["messages"] if m["role_name"] == "proposer")
    assert msg["credit_spans"] == {
        "solution": (0, 1), "interaction": (1, 3)}
    assert ex.n_credit_decode_fallback == 1
    assert ex.n_credit_split_failed == 0


def test_rollout_preserves_the_exact_prompt_used_for_sampling():
    """prompt 被截断时，训练必须复用 rollout 真正发送给 vLLM 的字符串。"""
    from pathlib import Path

    class CharEngine(FakeEngine):
        def generate_batch(self, requests):
            out = []
            for req in requests:
                self.calls.append(req["role"])
                self.prompts.append((req["role"], req["prompt"]))
                self.temps.append(req.get("temperature"))
                text = self.script[req["role"]].pop(0)
                out.append((text, [-0.5] * len(text), list(text)))
            return out

    response = ("推理过程：1+1=2\n最终答案：2\n"
                + INTER.format(a="none", t="none").strip())
    script = {
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"],
        "proposer": [response], "critic": [], "verifier": [],
    }
    tokenizer = FakeTokenizer()
    engine = CharEngine(script)
    executor = _mk_executor(
        engine, tokenizer=tokenizer, max_rounds=1, max_hops=0,
        stop_gate=False, max_prompt_tokens=500, token_credit=False)
    episode = executor.run_episodes_batch(["Q" * 1000], ["2"])[0]

    assert executor.n_prompt_clipped > 0, "测试没有真正触发 prompt 截断"
    assert len(episode["messages"]) == len(engine.prompts)
    for message, (role, sampled_prompt) in zip(episode["messages"], engine.prompts):
        assert message["role_name"] == role
        assert message["prompt_text"] == sampled_prompt, \
            "message 没保存 rollout 实际使用的 prompt"
        rebuilt_from_original = tokenizer.apply_chat_template(
            [{"role": "system", "content": message["system"]},
             {"role": "user", "content": message["user"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert rebuilt_from_original != sampled_prompt, \
            "测试没有覆盖原始 user 与截断 prompt 不一致的场景"

    trainer_src = (Path(__file__).resolve().parent /
                   "training/grpo_trainer.py").read_text(encoding="utf-8")
    assert 'prompt_text = msg.get("prompt_text")' in trainer_src, \
        "训练器没有优先复用 rollout 保存的精确 prompt"


def test_token_credit_can_be_disabled_for_ablation():
    """配置开关关闭时恢复旧的整 turn 标量优势，便于严格消融。"""
    cfg = {**CFG, "token_credit": False}
    td_ok, _ = compute_turn_data([make_round(0, primary="4")], "4", True, 4, cfg)
    td_bad, _ = compute_turn_data([make_round(0, primary="5")], "4", False, 4, cfg)
    assert td_ok[1]["token_credit"] is False
    assert td_bad[1]["token_credit"] is False
    adv = compute_raca_advantages([td_ok, td_bad])
    assert isinstance(adv[0][1], float) and isinstance(adv[1][1], float), \
        "关闭 token_credit 后没有回退到旧的整 turn 标量优势"


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


def test_eval_mode_kills_epsilon_injection_but_not_the_gate():
    """eval 只准量策略自己的行为：ε 注入必须关，闸门注入必须留。

    两件事写在一个函数里，因为它们的正确答案**相反**，分开放容易只记住一半：
    ε 注入是给冷启动喂数据的**训练手段**，留在 eval 里等于拿注入出来的交互
    去报「学会了求助」；闸门注入是**运行时机制**（controller 想停但黑板没
    verifier 分数），线上一样触发，eval 关掉它就测不到真实终止行为。

    第一段是差分对照：同一份脚本、同一个 `eps_force=1.0`，只切 `eval_mode`。
    不要只写 eval 一侧——`if self.eval_mode: eps_force = 0.0` 被删掉时，
    单看 eval 侧的「没发生交互」照样成立（脚本本来也可能不触发），断言不吃劲。
    """
    def one(eval_mode):
        script = {
            "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>"] * 4,
            "proposer":   [INTER.format(a="none", t="none") + "推理过程：x\n最终答案：4"] * 4,
            "verifier":   [INTER.format(a="none", t="none") + "分数: 0.9\n验证说明：ok"] * 4,
        }
        eng = FakeEngine(script)
        # share=1.0 固定注 verifier，使"注没注"这件事确定性可断言
        ex = _mk_executor(eng, eval_mode=eval_mode, eps_verifier_share=1.0)
        return eng, ex.run_episodes_batch(["q"], ["4"], eps_force=1.0)[0]

    eng_tr, res_tr = one(False)
    assert "verifier" in eng_tr.calls, \
        "训练侧 eps_force=1.0 都没注入——对照组本身失效，eval 侧的断言就没有意义"
    assert all(m["forced"] and not m["u"] for m in res_tr["raca_round_meta"])
    assert all(t == 1.0 for t in eng_tr.temps), f"训练侧应采样：{eng_tr.temps}"

    eng_ev, res_ev = one(True)
    assert "verifier" not in eng_ev.calls, f"eval 仍在做 ε 注入：{eng_ev.calls}"
    assert all(not m["forced"] and not m["u"] for m in res_ev["raca_round_meta"])
    # greedy 是同一个开关的另一半，一起钉住：温度漂了，eval 就不是可复现的尺子
    assert all(t == 0.0 for t in eng_ev.temps), f"eval 应贪心解码：{eng_ev.temps}"

    # ── 第二段：闸门注入在 eval 里必须照常触发 ────────────────────────────
    script = {
        "controller": ["<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>"] * 2,
        "proposer":   [INTER.format(a="none", t="none") + "推理过程：x\n最终答案：4"],
        "verifier":   [INTER.format(a="none", t="none") + "分数: 0.9\n验证说明：ok"],
    }
    eng = FakeEngine(script)
    # 不传 eps_force：证明这一注入与 ε 无关，纯由 gate_blocked 触发
    res = _mk_executor(eng, eval_mode=True).run_episodes_batch(["q"], ["4"])[0]
    m0 = res["raca_round_meta"][0]
    assert m0["gate_blocked"] and m0["forced"] and not m0["u"]
    assert "verifier" in eng.calls, "eval 把闸门注入也关了 → stop 永远解不了锁"
    assert res["stopped"] is True and len(res["raca_round_meta"]) == 1

    # 后果（这是 train.py 的 eval 行必须多印一列的理由）：本轮 forced=True 而
    # u=False，于是量 u 的 eval_int_rate 报 0.00，可 verifier 确实被调了一次。
    # 只印 int_rate 的话，「策略没求助」与「策略没求助、机制替它求助了」在日志
    # 上是同一个 0.00——而这两种情况对"要不要继续加 ε"的答案正好相反。
    rounds = res["raca_round_meta"]
    assert sum(m["u"] for m in rounds) == 0
    assert sum(m["forced"] for m in rounds) == 1


def test_offline_eval_uses_the_runtime_parsers():
    """evaluate.py 的三把尺子必须与运行时同源，否则离线评测量的是另一个系统。

    旧版自己手写 `最终答案：(.+)` / `分数:\\s*([0-9.]+)` / 字面量
    `"action: none" not in resp`。实测 2864 个 SFT turn：分数 0/748、交互
    0/1982 一致，但 **答案 3/580 漂移**（模型写半角 `最终答案: 24`）。

    这里钉的是漂移的**后果**而不是正则长相——一个缺口污染两个指标：
      ① 抓不到答案 → `last_proposer_ans` 空串 → proposer_accuracy 少算；
      ② 空串使 `math_equal("", gold)` 恒假 → 同一 episode 里 critic 的挑错被
         记成真阳性 → critic_precision 虚高（本例从 0.00 虚高到 1.00）。
    所以两个都断；只断第一个的话，第二个漂回去时测试仍然绿。
    """
    import evaluate as offline
    from agents.parsing import critic_found_errors

    ep = {
        "is_correct": True,
        "turn_ids": [0, 1, 2, 3],
        "messages": [
            # 半角冒号 + 尾部块：旧尺子在这一条上抓空
            {"role_name": "proposer",
             "response": "推理过程：算了一遍\n最终答案: 24\n"
                         + INTER.format(a="request", t="verifier")},
            # proposer 其实答对了，所以这次挑错是**误报**
            {"role_name": "critic", "response": "错误分析：第二步系数不对"},
            {"role_name": "verifier", "response": "分数: 0.9\n验证说明：ok"},
            # 全角 `action：` 运行时解析不出块 → 不算交互；旧字面量判据会把它
            # 算成一次交互（`"action: none"` 搜不到 + 有 `<interaction>`）
            {"role_name": "verifier",
             "response": "验证说明：没给分\n<interaction>\naction：none\n</interaction>"},
        ],
    }

    class FakeEx:
        def run_episode(self, q, a):
            return ep
        _critic_found_errors = staticmethod(critic_found_errors)

    res = offline.evaluate(FakeEx(), [{"question": "q", "answer": "24"}])
    assert res["proposer_accuracy"] == 1.0, "半角冒号的答案没被认出来"
    assert res["critic_flag_rate"] == 1.0                      # 确实挑了一次错
    assert res["critic_precision"] == 0.0, \
        "唯一一次挑错是误报，precision 必须是 0——非 0 说明答案没解析出来"
    # verifier：一条有分、一条没分。没分的不能按 0.5 兜底进分母
    # （0.5 正落在 `score > 0.5` 的边界上，会把"没打分"系统性记成"打了低分"）
    assert res["verifier_unparsed"] == 1
    assert res["verifier_consistency"] == 1.0                   # 分母只有那条 0.9
    # 交互计数走 parse_interaction：唯一的自发求助是 proposer 那次
    assert res["interaction_rate"] == 1.0


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

    # ②reasoning 缺失时返回整个输出，必须限长——且截断必须**可见**。
    # 标记刻意不占 limit 预算（`text[:limit] + 标记`），所以上界是
    # limit + len(标记)，而不是 limit。这条断言的形状本身就是那个设计决定：
    # 「保证送达 limit 个字符正文」不因为加标记而缩水。
    from agents.parsing import CLIP_MARK
    reasoning, _ = parse_reasoning("没有任何格式的长输出" + "哦" * 5000)
    assert len(reasoning) <= MAX_REASONING_CHARS + len(CLIP_MARK)
    assert reasoning.endswith(CLIP_MARK), "跨角色截断必须留下痕迹，否则接收方分不清『说完了』和『被砍了』"
    assert len(reasoning) - len(CLIP_MARK) == MAX_REASONING_CHARS
    # 没超限就不该有标记（否则接收方会以为有话被砍、把完整信息当残文对待）
    short, _ = parse_reasoning("推理过程：短推理\n最终答案：4")
    assert CLIP_MARK not in short and short == "短推理"

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
    """flaw 窗口 80 → 300 → **600**：v3「critic 说了等于没说」的直接成因。

    实测 critic 真报错时中位 267 字、p75 341 字、p90 494 字。80 的窗口只有 4% 能
    完整送达，而块又固定占 68 字——真正传下去的实质内容约 11 字。300 是 63.3%，
    第十轮抬到 600 是 96.3%。

    这条测试**刻意仍只用中位长度的样例**，因为它钉的是语义（"结论句要真的到下游"）
    而不是某个具体窗口值；换成 p90 长度的样例就变成在钉 600 这个数字本身，那样每
    次调窗口都要改测试，而调窗口恰恰是允许的。窗口值本身由
    `test_cross_role_truncation_is_visible` 那条按常量走的断言守。
    """
    from agents.parsing import MAX_CHANNEL_CHARS
    from envs.blackboard import Blackboard, Message, MessageType

    bb = Blackboard()
    bb.add_message(Message(0, MessageType.TRACE, ("r", "5")))
    # 一段 ~250 字的批评，结论在末尾（真实 critic 就是这么写的：先复述再定位）
    concl = "，所以第二步的 41 应为 42"
    flaw = ("第一步把条件抄错了：题目给的是 6×7，" + "复述与推导过程" * 31 + concl)
    assert 250 <= len(flaw) <= 340, f"样例长度 {len(flaw)} 不在实测中位区间"
    assert len(flaw) <= MAX_CHANNEL_CHARS, \
        f"样例({len(flaw)})超过了当前窗口({MAX_CHANNEL_CHARS})，本条测试的前提没了"
    bb.add_message(Message(1, MessageType.FLAW, {"content": flaw}))

    # 核心断言：结论句要真的出现在下游看到的文本里
    txt = bb.to_text()
    assert "发现问题" in txt
    assert concl in txt, "critic 的结论仍被窗口截掉了——信道没修好"
    # 反向对照：旧的 80 窗口连复述都没说完，结论一定丢
    assert concl not in flaw[:80]


def test_no_bare_lt_lookahead_in_parsers():
    """源码级不变量：`parsing.py` 的任何前瞻都不许用**裸 `<`**。

    这一类已经犯到第五例（flaw 窗口 80、块未剥、`parse_reasoning` 的推理前瞻、
    无标签兜底泄漏、`critic_found_errors` 的错误分析前瞻）。前四例都是被动撞见的，
    说明靠"下次记得"不行——`<` 在数学文本里是**不等号**，写下 `(?=<...)` 的那一刻
    看起来完全无害，要到某个 turn 的推理里出现 `若 x < 3` 才会显形，而且不报错，
    只是悄悄少送 77% 的正文。所以这条不变量必须由一个**会失败的测试**来守。

    判据是"前瞻里的 `<` 后面必须紧跟 `interaction`"。允许 `_INTER_OPEN` 这个
    常量本身出现（它就是那个开标签），也允许 f-string 里插值它。
    """
    import ast
    import pathlib
    import re as _re

    src = pathlib.Path(__file__).with_name("agents").joinpath("parsing.py").read_text()
    tree = ast.parse(src)

    # 只扫**真正的字符串字面量**，不扫原文。第一版按原文 grep，立刻被自己的
    # 文档绊倒：`parse_reasoning` 的 docstring 里逐字引用了旧的坏正则
    # （`(?=最终答案[：:]|<|$)`）作为病灶记录。那句引用必须留着——它是这个
    # 缺陷唯一的现场证据，删了以后没人知道当年错在哪。所以尺子得换：走 AST
    # 天然排除注释，再把「独立成句的字符串」（即 docstring）剔掉，剩下的就是
    # 会被 `re` 真正执行的那些串。
    docstrings = {
        id(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
    }
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in docstrings]
    assert literals, "一个字面量都没扫到——AST 走空了，断言会恒真（假绿）"

    # 前瞻/前视里的字面 `<`：`(?=` / `(?!` 之后到闭括号之前那一段。
    # f-string 会被拆成若干常量片段（`(?=` 与插值的 `_INTER_OPEN` 分开），
    # 因此插值写法不会误报；而**写死**的裸 `<` 一定落在同一个片段里，会被抓到。
    bad = []
    for lit in literals:
        for m in _re.finditer(r"\(\?[=!][^)]*", lit):
            frag = m.group(0)
            for lt in _re.finditer(r"<", frag):
                if not frag[lt.start():].startswith("<interaction"):
                    bad.append(frag)
    assert not bad, ("前瞻里有裸 `<`，数学不等号会把正文截掉——"
                     f"请改用 _INTER_OPEN：{bad}")

    # 正向：块开标签确实还在用（防止上面那条被"删掉所有前瞻"这种方式假绿）
    assert "_INTER_OPEN" in src and 'r"<interaction"' in src
    # 并且那句病灶记录还在（这条不变量的存在理由本身也得钉住）
    assert "(?=最终答案[：:]|<|$)" in src, "病灶记录被删了——后人会重新踩一次"


def test_critic_flag_is_insensitive_to_angle_brackets():
    """critic 的 flag 判定不因输出里出现 `<` / `>` 而改变。

    **这条测试不守 `_INTER_OPEN` 那个改动**——说明白很重要，否则它看起来像是。
    变异测试证过：把 `critic_found_errors` 的前瞻回滚成裸 `<`，本测试照样全绿
    （长度 ≤4 穷举 11110 串 + 真实 2228 条 critic turn，两版判定 0 处不同）。
    原因是这个函数只看 `err_text` **空不空**，而 `.+?` 至少吃一个字符、前导空白
    又被 `\\s*` 吃掉了，所以恒非空；「无错误」也早在第一个分支按整段拦掉了。
    守那个改动的是 `test_no_bare_lt_lookahead_in_parsers`（源码级不变量）。

    那这条测试能抓什么、不能抓什么，三次变异实测如下（写全是因为一条抓不到
    任何单点变异的测试很容易被误当成守着什么）：

    - 单变异「前瞻回滚成裸 `<`」：**不响**。理由同上，判定只看空不空。
      这条由 `test_no_bare_lt_lookahead_in_parsers` 守。
    - 单变异「`bool(err_text)` 改成 `len(err_text) > 5`」：**也不响**。因为修好
      的前瞻不再在 `<` 处停，插入尖括号根本不会让 `err_text` 变短。
    - **两处同时变异：响**（`错误分析：第<二步 ...` 判定翻转）。

    所以它的价值就是这一条**合取**，不多也不少：将来若有人给判定加上任何"看
    `err_text` 内容或长度"的逻辑，这条测试就成了那条逻辑与前瞻之间的耦合警报——
    单看任一处都没问题、合起来才出事的那类缺陷，正是这个仓库反复栽的地方。
    """
    from agents.parsing import critic_found_errors

    # 插入位置限定在**正文**里。插进 `错误分析` 标签中间（`错<误分析：`）或
    # 「无错误」中间（`无<错误`）会真的改变这个串的含义——标签没了就不该解析出
    # 报错段，关键词被劈开就不再是"没错误"。那种情形下判定翻转是**正确行为**，
    # 不是缺陷。这条测试要钉的是"尖括号出现在 critic 说的话里"这个真实场景。
    label = "错误分析："
    cases = [
        # (正文, 期望判定)：正文里怎么插尖括号都不许改判定
        ("第二步 2x=-12 应为 x=-6，符号错了", True),
        ("无错误，各步系数与代入均核对过", False),
    ]
    for body, want in cases:
        base = label + body
        assert critic_found_errors(base) is want, f"基线就不对：{base!r}"
        # 「无错误」这个关键词本身也不能被劈开，故 False 那条从它之后开始插
        safe_from = len(label) + (len("无错误") if not want else 0)
        for pos in range(safe_from, len(base) + 1):
            for ch in ("<", ">", "<第二步>"):
                mutated = base[:pos] + ch + base[pos:]
                assert critic_found_errors(mutated) is want, \
                    f"在正文第 {pos} 位插入 {ch!r} 后判定翻转：{mutated!r}"

    # 不等号的真实用法（整句），以及块在末尾时仍要正常截住
    assert critic_found_errors("错误分析：第三步要求 x < 3，但解里取了 x=5") is True
    assert critic_found_errors(
        "错误分析：第二步算错了\n" + INTER.format(a="request", t="proposer")) is True
    assert critic_found_errors(
        "错误分析：无错误\n" + INTER.format(a="none", t="none")) is False


def test_cross_role_truncation_is_visible():
    """所有跨角色截断都必须留下 `CLIP_MARK`。

    第四轮复盘的结论：**坏的不是「截断」，是「截断不可见」。** 截断本身必要
    （prompt 预算有限），致命的是接收方拿到残文后分不清「对方说完了」和「对方
    还有话被砍了」，于是照常给一个自信的判断——critic 对着断在 `C\\sqrt{` 处的
    LaTeX 回答"有没有计算错误"，proposer 对着半句批评去"修正"。标记改变的是
    接收方的**可判定性**：看到它就该说"信息不全"，而不是"这步是错的"。

    覆盖三条信道（三处共用 `MAX_CHANNEL_CHARS`）：黑板 `发现问题`、
    `request_context` 的 `对方内容`、`proposer_correction_user` 的引述。
    `parse_reasoning` 的推理上限在 `test_unbounded_text_is_capped` 里钉。
    """
    from agents.parsing import CLIP_MARK, MAX_CHANNEL_CHARS, clip_text
    from envs.blackboard import Blackboard, Message, MessageType
    from llm.prompt_templates import PromptTemplates

    long = "复述与推导" * 200          # 1000 字，远超窗口
    short = "第二步把 6*7 算成了 41"    # 远低于窗口
    assert len(long) > MAX_CHANNEL_CHARS > len(short)

    # ① 单元语义：超限带标记且正文不缩水，未超限一字不改
    assert clip_text(long, MAX_CHANNEL_CHARS) == long[:MAX_CHANNEL_CHARS] + CLIP_MARK
    assert clip_text(short, MAX_CHANNEL_CHARS) == short

    # ② 黑板 flaw 窗口
    bb = Blackboard()
    bb.add_message(Message(0, MessageType.TRACE, ("r", "5")))
    bb.add_message(Message(1, MessageType.FLAW, {"content": long}))
    assert CLIP_MARK in bb.to_text(), "flaw 被截了却没留痕迹"

    bb2 = Blackboard()
    bb2.add_message(Message(0, MessageType.TRACE, ("r", "5")))
    bb2.add_message(Message(1, MessageType.FLAW, {"content": short}))
    assert CLIP_MARK not in bb2.to_text(), \
        "没超限却带了标记——接收方会把完整信息当残文对待"

    # ③ 两处模板引述
    ctx = PromptTemplates.request_context("Critic", "request", "请审查", long)
    assert CLIP_MARK in ctx
    corr = PromptTemplates.proposer_correction_user("6*7=?", "Critic", long, "尚无信息")
    assert CLIP_MARK in corr
    # 未超限时两处都不留标记
    assert CLIP_MARK not in PromptTemplates.request_context(
        "Critic", "request", "请审查", short)
    assert CLIP_MARK not in PromptTemplates.proposer_correction_user(
        "6*7=?", "Critic", short, "尚无信息")

    # ④ 标记本身要**能被人看懂**，且不能撞上块标签或黑板行首标记
    #   （它会出现在 prompt 里被模型读到，也会被 grep 用来量截断率）
    assert "截断" in CLIP_MARK
    assert "<" not in CLIP_MARK and "当前状态" not in CLIP_MARK


def test_answer_cap_admits_the_latex_answers_that_actually_occur():
    """答案上限要容得下**数据集里真实存在**的长答案（v3.2 第六轮：64 → 192）。

    64 是 v3.1 拍的，实测偏紧到会毁掉正确答案。下面这个矩阵是命中的那一次
    （v2 SFT 580 个 proposer turn 里唯一一次），原文照抄、68 字、完全正确，
    旧代码把它兜底抽成 `'4'`：一个对的答案被记成错，还带着 `'4'` 这张假票
    进投票池。gold 侧同样矛盾——`math_train_rl` 5265 题有 14 题 gold 超 64
    （最长 159），`math_test` 3669 题有 6 题（最长 81）。

    钉三件事：真实长答案原样返回、上限之上仍然退化（放宽不等于取消）、以及
    **不要顺手把黑板展示上限也当成同一个数**（两者职责不同，见
    `Blackboard._MAX_ANSWER_CHARS` 的注释）。
    """
    from agents.parsing import parse_reasoning, MAX_ANSWER_CHARS
    from envs.blackboard import Blackboard

    matrix = r"\begin{pmatrix} 2 & 0 & 7 \\ 3 & 5 & -1 \\ -8 & -2 & 4 \end{pmatrix}"
    # 不写死 68：只要它落在「旧上限砍掉、新上限放过」这个窗口里就够了，
    # 这正是本轮改动的语义。
    assert 64 < len(matrix) <= MAX_ANSWER_CHARS, len(matrix)
    _, ans = parse_reasoning(f"推理过程：逐项相乘\n最终答案：{matrix}")
    assert ans == matrix, f"合法矩阵答案又被兜底抽成数字了：{ans!r}"

    # 已观测最长 gold（math_train_rl，159 字）也必须过——上限是按它定的
    longest_gold = (r"\begin{pmatrix} \frac{4}{9} & -\frac{4}{9} & -\frac{2}{9} "
                    r"\\ -\frac{4}{9} & \frac{4}{9} & \frac{2}{9} "
                    r"\\ -\frac{2}{9} & \frac{2}{9} & \frac{1}{9} \end{pmatrix}")
    assert len(longest_gold) <= MAX_ANSWER_CHARS, (
        f"最长 gold {len(longest_gold)} 字都过不了上限 {MAX_ANSWER_CHARS}，"
        "这条上限还在和数据集矛盾")
    _, ans2 = parse_reasoning(f"推理过程：解方程\n最终答案：{longest_gold}")
    assert ans2 == longest_gold

    # 上限之上仍退化（`test_unbounded_text_is_capped` 用 500 字，这里贴着边界）
    _, ans3 = parse_reasoning(
        "推理过程：x\n最终答案：" + "啦" * MAX_ANSWER_CHARS + "42")
    assert ans3 == "42"

    # 两个上限**不应**相等。相等时黑板那两处截断永远切不到东西（第六轮之前就是
    # 这样：两个数都是 64，而解析后答案最长 55），于是「展示层会不会静默截断」
    # 这个问题被永久隐藏。放宽解析上限正是让它显形的那一步。
    assert MAX_ANSWER_CHARS > Blackboard._MAX_ANSWER_CHARS, \
        "两个上限被拉平了——展示层的截断会重新变成不可观测的"


def test_blackboard_answer_display_truncation_is_visible():
    """黑板**答案**信道的截断也必须留标记——第四轮漏掉的两处。

    第四轮把 flaw / request_context / correction 三条信道统一到 `clip_text`，
    并在记忆文档里写下"clip_text 是跨角色截断的唯一出口"。那句话当时是错的：
    `to_text` 里还有两处 `a[:self._MAX_ANSWER_CHARS]` 的裸切片（答案列表、
    最高置信答案）。它们没被发现是因为**当时切不到东西**——解析上限与展示上限
    都是 64，parse 出来的答案不可能超过 64。第六轮把解析上限放到 192，这两处
    才第一次真正开始截断。
    """
    from agents.parsing import CLIP_MARK
    from envs.blackboard import Blackboard, Message, MessageType

    long_ans = "x" * (Blackboard._MAX_ANSWER_CHARS + 30)
    short_ans = "42"

    # ① 答案列表那一行
    bb = Blackboard()
    bb.add_message(Message(0, MessageType.TRACE, ("r", long_ans)))
    assert CLIP_MARK in bb.to_text(), "答案列表被截了却没留痕迹"

    # ② 「最高置信答案」那一行（需要有分数才会出现）
    bb.add_message(Message(1, MessageType.SCORE, (long_ans, 0.9)))
    txt = bb.to_text()
    assert "最高置信答案" in txt
    assert txt.count(CLIP_MARK) == 2, f"两行各应有一个标记：{txt!r}"

    # ③ 未超限时一字不改（否则接收方会把完整答案当残文对待）
    bb2 = Blackboard()
    bb2.add_message(Message(0, MessageType.TRACE, ("r", short_ans)))
    bb2.add_message(Message(1, MessageType.SCORE, (short_ans, 0.9)))
    t2 = bb2.to_text()
    assert CLIP_MARK not in t2 and "最高置信答案：42（分数0.90）" in t2, t2

    # ④ 截断只发生在**展示层**：投票与判分读的是 traces，必须是原文。
    #    这条是整个改动的安全边界——展示带标记不能污染答案本身。
    assert bb.get_distinct_answers() == [long_ans]
    assert all(a == long_ans for _, a in bb.traces)


def test_no_bare_channel_slicing_in_blackboard():
    """源码级不变量：`blackboard.py` 不许再用 `_MAX_*CHARS` 做上界裸切片。

    第四轮的教训是"截断不可见"这一类靠人扫会漏——事实就是漏了两处（本轮补的
    答案信道）。而它漏得特别隐蔽：那两处当时**切不到东西**，所以任何行为测试
    都抓不到它们，只有源码级不变量能。这与
    `test_no_bare_lt_lookahead_in_parsers` 是同一种守法，理由也同一条：写下
    `a[:64]` 的那一刻看起来完全无害，要到上游某个常量被调大才会显形。

    判据：AST 里任何 `X[:Y]` 形式的切片，其上界不得是名字里带 `_CHARS` 的
    属性或变量——那种上界一律该走 `clip_text`。
    """
    import ast
    import pathlib

    path = pathlib.Path(__file__).with_name("envs").joinpath("blackboard.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def is_chars_bound(node):
        if isinstance(node, ast.Attribute):
            return "_CHARS" in node.attr
        if isinstance(node, ast.Name):
            return "_CHARS" in node.id
        return False

    bad = [ast.unparse(n) for n in ast.walk(tree)
           if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice)
           and n.slice.upper is not None and is_chars_bound(n.slice.upper)]
    assert not bad, ("信道文本被裸切片截断，接收方看不到痕迹——请改用 clip_text："
                     f"{bad}")

    # 正向：确认这个文件真的在用 clip_text，且真的有 `_CHARS` 常量可供误用。
    # 否则上面那条断言可能只是因为"文件里什么都没有"而恒真（假绿）。
    src = path.read_text(encoding="utf-8")
    assert "clip_text(" in src and "_MAX_ANSWER_CHARS" in src



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

    窗口从 80 放宽之后这份重复才显形：同一段批评在一个 prompt 里出现两次，白占
    `MAX_CHANNEL_CHARS` 字预算（`max_prompt_tokens` ≈ 3008，且黑板文本嵌进每个
    角色的 prompt）。去重按**内容比对**而非「initiator 是不是 critic」——critic
    未标错却主动请求修正时 `flaws[-1]` 是更早的另一条，那份信息是真的、不能扔。

    **这条只守 proposer 修正那一路。** 响应方（critic / verifier）那一路上同样的
    重复直到第十轮才被量出来——`data/sft_train_v23.jsonl` 渲染出的 310 处「对方
    内容」里 247 处是同 prompt 内逐字重复、每处约 263 字——由
    `test_request_context_drops_the_duplicate_quote` 守。两路的判据不同，不能合并。
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


def test_parse_failure_is_decomposed_into_two_modes():
    """`parse_rate` 必须拆成「无标签」与「空答案」两个读数（#23，纯埋点）。

    为什么非拆不可：`primary_parsed = has_answer_label(out) and bool(answer)` 是
    **合取**，聚合后的一个数无法回答「掉的是标签还是答案」。这不是洁癖——两种
    失败的下游后果和修法都不同：
      · 无标签 → 走「取文中最后一个数字」兜底，实测 23 个真实无标签 turn 里 6 个
        （26%）抽出垃圾数字（gold `\\frac{7}{32}` → `'32'`），**可解析**、与真票
        在票池里不可分，"过滤不可解析候选"闸不住它；
      · 空答案 → 空串进票池，实测两票（2×0.5=1.0）压过一票被 verifier 背书的
        正确答案（0.9×1）。
    v3 那 150 步只有一个 `parse`（0.95 → 最低 0.76），所以这两种到底各占多少，
    在这一步之前是**测不出来的**（`train.py` 只 dump `{"step": step}`，答案串从不
    落盘）。本测试钉住的正是"拆得开"这件事本身。
    """
    from agents.parsing import parse_reasoning, has_answer_label
    from training.metrics import rollout_metrics as T_metrics

    def ep(meta):
        return {"is_correct": True, "stopped": True, "raca_round_meta": meta}

    base = {"u": False, "forced": False, "p_primary": True, "p_end": True,
            "target": None, "gate_blocked": False}

    def rnd(no_label, empty):
        return {**base, "primary_parsed": not (no_label or empty),
                "no_label": no_label, "empty_answer": empty}

    # ① 同一个 parse_rate，两种完全不同的成因 —— 这就是拆开的全部理由。
    half_no_label = [rnd(True, False), rnd(False, False)]
    half_empty    = [rnd(False, True), rnd(False, False)]
    m_a = T_metrics([[ep(half_no_label)]])
    m_b = T_metrics([[ep(half_empty)]])
    assert approx(m_a["parse_rate"], 0.5) and approx(m_b["parse_rate"], 0.5), \
        "两种成因本应给出相同的 parse_rate，否则这条测试证明不了它分不开"
    assert approx(m_a["no_label_rate"], 0.5) and approx(m_a["empty_answer_rate"], 0.0)
    assert approx(m_b["no_label_rate"], 0.0) and approx(m_b["empty_answer_rate"], 0.5)

    # ② 两个标记不是 `primary_parsed` 的换名：无标签那一路答案**非空**
    #    （兜底抽出了数字），所以 parsed=False 而 empty_answer=False。
    #    把 empty_answer 写成 `not primary_parsed` 会在这里响。
    assert approx(m_a["empty_answer_rate"], 0.0), \
        "空答案率跟着无标签一起涨了——empty_answer 被写成 primary_parsed 的取反？"

    # ③ 标记必须与真实模型输出对得上，按执行器里那两行同样的方式推导。
    real = [
        # (输出, 期望 no_label, 期望 empty_answer)
        ("推理过程：算一下\n最终答案：42",        False, False),  # 正常
        ("推理过程：算一下\n最终答案：\n42",      False, False),  # 无空格 → 兜底捞回
        ("推理过程：算一下\n最终答案： \n42",     False, True),   # **一个空格** → 空串
        ("所以结果是 42",                        True,  False),  # 无标签，兜底有数字
        ("所以结论显然成立",                     True,  True),   # 无标签且无数字
    ]
    for out, exp_no_label, exp_empty in real:
        _, answer = parse_reasoning(out)
        no_label = not has_answer_label(out)
        empty = not answer
        assert (no_label, empty) == (exp_no_label, exp_empty), (
            f"{out!r} 的两个标记应为 {(exp_no_label, exp_empty)}，"
            f"实际 {(no_label, empty)}（answer={answer!r}）")
    # 第 3 例与第 2 例只差一个空格，却一个丢答案一个不丢。这两行钉的是**当前的
    # 缺陷行为，不是期望行为**（病灶记录）：`最终答案： \n42` 的答案就在下一行，
    # 却被整个丢掉。#22 修 `parse_reasoning` 时这两行会翻，翻了就说明修到了；
    # 不写进测试，下次改正则时没人会记得这条路径存在。
    assert parse_reasoning("推理过程：x\n最终答案： \n42")[1] == ""
    assert parse_reasoning("推理过程：x\n最终答案：\n42")[1] == "42"

    # ④ 旧 round record 没这两个键：只能读成 0.0（低报），不许凭空造失败率，
    #    也不许把 parse_rate 带坏。
    legacy = [{**base, "primary_parsed": False}, {**base, "primary_parsed": True}]
    m_old = T_metrics([[ep(legacy)]])
    assert approx(m_old["parse_rate"], 0.5)
    assert approx(m_old["no_label_rate"], 0.0) and approx(m_old["empty_answer_rate"], 0.0)

    # ⑤ 走完真实的三跳链路：round_records → compute_turn_data → round_meta →
    #    rollout_metrics。这一段不是形式主义——`round_meta` 是 `compute_turn_data`
    #    里**重新拼的白名单 dict**，不是 round_records 的透传。第一版实现就漏在
    #    这一跳：executor 记了、metrics 读不到，两个指标照样打印且永远 0.00，
    #    也就是一个**看起来在测量、实际什么都没测**的埋点。
    rr = [
        {**make_round(0, primary="4"), "primary_parsed": False,
         "no_label": True,  "empty_answer": False},   # 无标签，兜底抽到了数字
        {**make_round(1, primary=""),  "primary_parsed": False,
         "no_label": False, "empty_answer": True},    # 有标签但答案空
        {**make_round(2, primary="4"), "primary_parsed": True,
         "no_label": False, "empty_answer": False},   # 正常
    ]
    _, meta = compute_turn_data(rr, "4", True, 4, CFG)
    assert [m["no_label"] for m in meta] == [True, False, False], \
        f"no_label 没穿过 round_meta 那一跳：{[m.get('no_label') for m in meta]}"
    assert [m["empty_answer"] for m in meta] == [False, True, False], \
        f"empty_answer 没穿过 round_meta 那一跳：{[m.get('empty_answer') for m in meta]}"
    m_chain = T_metrics([[{"is_correct": True, "stopped": True,
                           "raca_round_meta": meta}]])
    assert approx(m_chain["parse_rate"], 1 / 3)
    assert approx(m_chain["no_label_rate"], 1 / 3)
    assert approx(m_chain["empty_answer_rate"], 1 / 3)
    # 并且两种失败确实各占一份，而不是同一份被数了两次
    assert approx(m_chain["no_label_rate"] + m_chain["empty_answer_rate"],
                  1 - m_chain["parse_rate"])


def test_parse_flags_are_wired_from_source_not_aliased():
    """源码级不变量：执行器里那两个标记必须各自独立算，不许挂在 `primary_parsed` 上。

    为什么只能用源码级：写这两行的位置在 `run_episodes_batch` 里，那个函数没有
    vLLM 引擎就跑不起来，**任何行为测试都到不了**。这和第六轮那两处裸切片是同一
    类问题——上一轮的教训就是"测不到的代码不会被行为测试保护"，所以照同样的办法
    用 AST 守。

    要守住的具体退化是把 `empty_answer` 写成 `not st["primary_parsed"]`：那样两个
    计数器就退化成同一个数（`1 - parse_rate`），本步的全部意义归零，而且**看不出
    来**——两个指标都还在报，只是永远相等。
    """
    import ast
    import pathlib

    path = pathlib.Path(__file__).with_name("agents").joinpath("agentic_executor.py")
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    rhs = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if (isinstance(tgt, ast.Subscript) and isinstance(tgt.slice, ast.Constant)
                and tgt.slice.value in ("no_label", "empty_answer")):
            rhs[tgt.slice.value] = ast.unparse(node.value)

    assert set(rhs) == {"no_label", "empty_answer"}, \
        f"执行器里没有同时给两个标记赋值：{sorted(rhs)}"
    assert "has_answer_label" in rhs["no_label"], \
        f"no_label 不是从 has_answer_label 算的：{rhs['no_label']!r}"
    for k, v in rhs.items():
        assert "primary_parsed" not in v, (
            f"{k} 挂在 primary_parsed 上了 —— 两个计数器会退化成 1−parse_rate，"
            f"且退化后依然两个都在报、看不出来：{v!r}")
    assert "answer" in rhs["empty_answer"] and "has_answer_label" not in rhs["empty_answer"], \
        f"empty_answer 应该只看答案本身空不空：{rhs['empty_answer']!r}"

    # 算出来还得真的带进 round record，否则 metrics 那边永远只能读到默认值 0.0。
    for key in ("no_label", "empty_answer"):
        assert f'"{key}":' in src, f"{key} 没进 round record，指标会永远读成 0.0"


class FakeQwen3:
    """按 Qwen3 模板的三个关键分支复刻（供模板探针的两条测试共用）。

    `inject_gen`  = `add_generation_prompt=True` 且 `enable_thinking=False`
                    时是否注入空 think 块（真实 Qwen3 会注入）。
    `inject_full` = 末条 assistant 消息是否也带上空 think 块（真实 Qwen3 的
                    `loop.last` 分支会带）。
    `full_flag_sensitive` = 上面那个块**是否也受 `enable_thinking=False` 影响**。
                    这一维是 08-28 才加的，专为探针的第 4 种渲染
                    （`sft_full_ef` = `train_sft.py:37` + `enable_thinking=False`）：
                    若不加这一维，`inject_full` 与 `enable_thinking` 无关 ⇒
                    `sft_full_ef` 永远等于 `sft_full` ⇒ `both_flag_ok` 永远等于
                    `only_38_ok`，"两行都补会打破前缀不变量"那条路径一次都走不到，
                    等于给 #25 的决策加了段没测过的代码。
    """

    def __init__(self, inject_gen, inject_full, full_flag_sensitive=False):
        self.inject_gen = inject_gen
        self.inject_full = inject_full
        self.full_flag_sensitive = full_flag_sensitive

    def apply_chat_template(self, messages, tokenize=False,
                           add_generation_prompt=False, enable_thinking=None):
        s = ""
        for m in messages:
            body = m["content"]
            suppressed = self.full_flag_sensitive and enable_thinking is False
            if m["role"] == "assistant" and self.inject_full and not suppressed:
                body = "<think>\n\n</think>\n\n" + body
            s += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
            if self.inject_gen and enable_thinking is False:
                s += "<think>\n\n</think>\n\n"
        return s

    # 按字符当 token（与 FakeTokenizer 同一套约定）
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids):
        return "".join(ids)


# 探针要吃一个真实形状的 turn（M1 之后 <interaction> 在尾部），两条测试共用一份，
# 免得各自维护一份、哪天形状漂了只改一处。
_PROBE_TURN = {
    "role_name": "proposer",
    "system": "你是解题者",
    "user": "问题：1+1\n当前状态：尚无信息",
    "response": "推理过程：显然\n最终答案：2\n<interaction>none</interaction>",
}


def test_sft_template_probe_separates_the_three_think_block_outcomes():
    """`check_sft_template.probe_template` 必须能分清三种结局，不能只会说"不对齐"。

    背景：AST 普查 19 处 `apply_chat_template`，9 处显式传
    `enable_thinking=False`（**口径：遍历全仓 .py 但排除 test_*.py**；含测试文件
    是 21 / 9 / 12，多出来的两处是本文件给假 tokenizer 的调用，与本议题无关。这个
    范围要跟 `check_sft_template.py` 模块 docstring A 段写的一致，否则两边数字对
    不上，下一个读者会以为其中一处过期），其中 5 处在真推理/数据链路上
    （executor:106 / grpo_trainer:237 / verify_sft_format:84 / measure_channels:96
    / generate_sft_v3:111），另 4 处是量具自身。10 处没传的里面只有真正训练的那处
    `train_sft.py:37-38` 是活着的缺口（其余是本脚本的探针臂和已废弃的诊断脚本）。
    这处不对称的**后果**取决于 Qwen3 模板怎么渲染末条 assistant 消息，而那要有
    tokenizer 才量得出来，本地开发机没有 —— 所以探针本身必须先被证明是准的，
    否则集群上那一行读数没人敢信。

    三种结局的后果差很远，混成一句"不对齐"等于没测：
      ① aligned                          —— 两处渲染一致，这处不对称无后果
      ② think_block_in_supervised_region —— 推理 prompt 仍是 SFT 序列的**逐 token
         前缀**，空 think 块落在了 labels != -100 那段里。注意后果**不是**"SFT 教
         模型再吐一个块"（那句话是 08-27 写错的，`prefix_ok=True` 就否掉了它：块
         在两侧出现的位置相同，推理时生成从块之后起，而那个位置被直接监督过）。
         真实后果只有错位监督 delta 个 token + loss 平均被稀释，RL 侧安全。
      ③ diverged                         —— 推理喂进去的前缀 SFT 从未见过
    """
    import io
    from contextlib import redirect_stdout

    from check_sft_template import probe_template

    turn = _PROBE_TURN

    cases = {
        # 生成侧不注入 ⇒ 两处 prompt 逐字节相同
        "aligned": FakeQwen3(inject_gen=False, inject_full=False),
        # 生成侧注入、末条 assistant 也带 ⇒ 块落进监督区
        "think_block_in_supervised_region": FakeQwen3(inject_gen=True, inject_full=True),
        # 只有生成侧注入 ⇒ 推理 prompt 根本不是 SFT 序列的前缀
        "diverged": FakeQwen3(inject_gen=True, inject_full=False),
    }
    got = {}
    for want, tok in cases.items():
        buf = io.StringIO()
        with redirect_stdout(buf):
            res = probe_template(tok, turn)
        got[want] = res["verdict"]
        assert res["verdict"] == want, \
            f"{want} 这一档被判成了 {res['verdict']}（三档判混了，读数就没有意义）"
        # 人读的那一行必须与返回值一致 —— 只算对不打对，等于集群上什么都没留下
        out = buf.getvalue()
        if want == "aligned":
            assert "对齐" in out and "不对齐" not in out, out
        elif want == "think_block_in_supervised_region":
            assert "监督区里" in out, out
        else:
            assert "不对齐" in out, out

    # 三档必须互不相同，否则上面三条断言可以被一个恒定返回值同时满足
    assert len(set(got.values())) == 3, got

    # ② 这一档最要紧的具体内容：监督区确实从空 think 块开始，而 response 本身没有它
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = probe_template(cases["think_block_in_supervised_region"], turn)
    assert res["supervised"].startswith("<think>"), \
        f"监督区没有以 think 块开头，那这一档的叙述就是错的：{res['supervised'][:40]!r}"
    assert "<think>" not in turn["response"], "response 本身不该带 think 块"
    # 推理 prompt 比 SFT prompt 长出的就是那个块
    assert res["delta_tokens"] == len("<think>\n\n</think>\n\n"), res["delta_tokens"]

    # ③ 走散那一档：前缀不成立，且不能被误判成 ②
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = probe_template(cases["diverged"], turn)
    assert not res["prefix_ok"] and not res["same_prompt"], res


def test_sft_template_probe_tells_apart_the_two_ways_to_fix_train_sft():
    """第 4 种渲染必须能分清「只补 :38」和「两行都补」的后果，#25 要照它决定改哪行。

    为什么需要这条：`train_sft.py:37-38` 两行都没传 `enable_thinking`。直觉上"两行
    都补"最整齐，但 Qwen3 的 `loop.last` 分支**可能也受这个参数影响**——若是，补完
    `:37` 后 `sft_full` 就不再含空 think 块，而推理侧的 `rl_prompt` 仍含（executor
    那处一直传 `enable_thinking=False`），于是 `prefix_ok` 会从 True 翻成 False：
    等于把现在这个无害的错位换成一次**真的走散**。本地无 tokenizer 量不出来，所以
    探针把两种补法的前缀不变量都算出来，让 #25 照读数改而不是照猜改。

    三种结局，按 (full_flag_matters, both_flag_ok) 区分，必须互不相同：
      A (F, T) —— :37 传不传都一样渲染 ⇒ 只补 :38 即可，补 :37 也无害
      B (T, F) —— **只能补 :38** ⇒ 两行都补会打破前缀不变量（最要紧的一档）
      C (T, T) —— 两种补法都保住前缀 ⇒ 可自由选
    """
    import io
    from contextlib import redirect_stdout

    from check_sft_template import probe_template

    turn = _PROBE_TURN
    cases = {
        # A：末条 assistant 的块不受 enable_thinking 影响（旧的两维假 tokenizer 就是这档）
        "A": (FakeQwen3(inject_gen=True, inject_full=True, full_flag_sensitive=False),
              False, True, "think_block_in_supervised_region", "加不加都一样"),
        # B：块受影响 ⇒ 补 :37 会把它从 sft_full 里拿掉，而 rl_prompt 仍带 ⇒ 前缀断
        "B": (FakeQwen3(inject_gen=True, inject_full=True, full_flag_sensitive=True),
              True, False, "think_block_in_supervised_region", "只能补 :38"),
        # C：生成侧本来就不注入 ⇒ rl_prompt 不含块，拿掉 sft_full 的块反而仍是前缀
        "C": (FakeQwen3(inject_gen=False, inject_full=True, full_flag_sensitive=True),
              True, True, "aligned", "都保住前缀"),
    }

    seen = {}
    for name, (tok, want_matters, want_both, want_verdict, want_phrase) in cases.items():
        buf = io.StringIO()
        with redirect_stdout(buf):
            res = probe_template(tok, turn)
        out = buf.getvalue()
        assert res["full_flag_matters"] is want_matters, \
            f"{name}: full_flag_matters={res['full_flag_matters']}，期望 {want_matters}"
        assert res["both_flag_ok"] is want_both, \
            f"{name}: both_flag_ok={res['both_flag_ok']}，期望 {want_both}"
        assert res["verdict"] == want_verdict, f"{name}: verdict={res['verdict']}"
        # 只算对不打对，等于集群上什么都没留下
        assert want_phrase in out, f"{name}: 人读的那一行里没有 {want_phrase!r}\n{out}"
        # only_38_ok 就是现状的 prefix_ok，别让谁悄悄把它重定义成别的东西
        assert res["only_38_ok"] is res["prefix_ok"], name
        seen[name] = (res["full_flag_matters"], res["both_flag_ok"])

    # 三档的读数组合必须互不相同，否则上面的断言可以被一个恒定返回值同时满足
    assert len(set(seen.values())) == 3, seen

    # B 这一档是唯一会拦住"两行都补"的，它必须**不给**修后监督区（前缀都不成立了，
    # 那个切片没有意义），而 A/C 必须给，且给出来的是 response 而不是 think 块——
    # 这正是修法要达成的效果：把块挪出监督区。
    for name in ("A", "C"):
        buf = io.StringIO()
        with redirect_stdout(buf):
            res = probe_template(cases[name][0], turn)
        assert res["supervised_ef"] is not None, name
        assert res["supervised_ef"].startswith("推理过程："), \
            f"{name}: 修后监督区没有从 response 起头：{res['supervised_ef'][:40]!r}"
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = probe_template(cases["B"][0], turn)
    assert res["supervised_ef"] is None, \
        f"B 档前缀不成立，不该给出修后监督区：{res['supervised_ef']!r}"
    # 而且 B 档现状的监督区确实是从块开始的（否则这一档的叙述就是错的）
    assert res["supervised"].startswith("<think>"), res["supervised"][:40]


def test_sft_template_probe_pins_the_train_sft_call_shape():
    """探针复刻的是 `train_sft.py` 那两行，形状漂了必须当场喊过期而不是照样出数。

    这条守的是探针最坏的失效方式：`train_sft.py` 改了调用形状、探针没跟上，于是
    它量的是一个**已经不存在**的训练流程，却照样打印一行看起来正常的读数。
    同时必须把"形状漂了"（该拦，exit 2）和"病灶被修好了"（该继续、只是退休）
    分开 —— 第一版用 grep 字面串，一旦真把 `enable_thinking=False` 补进
    `train_sft.py`，字面串立刻不匹配，**修 bug 反倒把作业弄挂**。
    """
    import pathlib

    from check_sft_template import _extract_calls, assert_mirrors_train_sft

    # 真实源码：两处都在，且病灶前提仍成立
    assert assert_mirrors_train_sft() is True

    src = pathlib.Path(__file__).with_name("train_sft.py").read_text(encoding="utf-8")
    found = _extract_calls(src)
    assert set(found) == {"full_text", "prompt_text"}, sorted(found)
    # 病灶本身：训练侧两处都没传 enable_thinking，而推理侧五处都传了
    for name, (_, kw) in found.items():
        assert "enable_thinking" not in kw, \
            f"{name} 已经传了 enable_thinking，本测试与探针的 A 段都该退休了"
    assert found["full_text"][0] == ["messages"]
    assert found["prompt_text"][0] == ["messages[:-1]"]

    # 反向：把 enable_thinking 补上（= 病灶被修），探针必须**不**退出，只提示退休
    tmp = pathlib.Path(__file__).with_name("_tmp_train_sft_fixed.py")
    tmp.write_text(src.replace(
        "apply_chat_template(messages, tokenize=False, add_generation_prompt=False)",
        "apply_chat_template(messages, tokenize=False, "
        "add_generation_prompt=False, enable_thinking=False)"), encoding="utf-8")
    try:
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            premise = assert_mirrors_train_sft(str(tmp))
        assert premise is False, "病灶被修好之后应当报 premise=False"
        assert "退休" in buf.getvalue(), buf.getvalue()
    finally:
        tmp.unlink()

    # 而形状漂了必须 exit 2（拦住，别出假读数）。两档都要盯：位置参数换了、以及
    # 多传一个关键字（`tools=` 之类足以改变渲染，而探针复刻的是没有它的版本）。
    drifts = {
        "位置参数": src.replace("apply_chat_template(messages[:-1], tokenize=False",
                                "apply_chat_template(messages, tokenize=False"),
        "多传关键字": src.replace(
            "apply_chat_template(messages[:-1], tokenize=False, "
            "add_generation_prompt=True)",
            "apply_chat_template(messages[:-1], tokenize=False, "
            "add_generation_prompt=True, tools=[])"),
    }
    for label, mutated in drifts.items():
        assert mutated != src, f"{label} 这一档的变异没生效，这条断言是空的"
        tmp2 = pathlib.Path(__file__).with_name("_tmp_train_sft_drift.py")
        tmp2.write_text(mutated, encoding="utf-8")
        try:
            raised = None
            try:
                assert_mirrors_train_sft(str(tmp2))
            except SystemExit as e:
                raised = e.code
            assert raised == 2, f"{label} 漂了却没有 exit 2（拿到 {raised}）"
        finally:
            tmp2.unlink()


def test_sft_template_census_splits_over_limit_three_ways():
    """B 段普查新加的三项输出必须真的算对，不能只在集群上才第一次跑到。

    三分法的后果差很远，混在一起等于没测（`train_sft.py:42` + `:96`）：
      超限   —— full > max_len，被 `[:max_length]` 砍掉尾巴
      零梯度 —— prompt 自己就 >= max_len ⇒ labels 全 -100 ⇒ `response_mask.sum()
                == 0` ⇒ 整条 turn 一点梯度都不产生，且**日志里不留痕**
      砍尾   —— 超限里除掉零梯度的那些，仍有梯度，只是丢尾巴；M1 把
                `<interaction>` 块搬到了尾部，所以每一条砍尾都等于一条"交互决策
                没被监督"的样本
    另外钉监督区长度（full − prompt）：它是 A 段那 delta 个错位 token 的分母，
    按角色打是因为 controller 只输出一行 decision、监督区最短，稀释比例的上界由
    它决定。这些都是本轮新加的打印，此前 `census_truncation` 一条测试都没有。
    """
    import io
    import json
    import os
    import tempfile
    from contextlib import redirect_stdout

    from check_sft_template import census_truncation

    tok = FakeQwen3(inject_gen=False, inject_full=False)

    def mk(role, s, u, r):
        return {"role_name": role, "system": "s" * s,
                "user": "u" * u, "response": "r" * r}

    # 假 tokenizer 下 full − prompt = len(response) + len("<|im_end|>\n") = len+11
    turns = [
        mk("controller", 10, 10, 5),     # 监督区 16，整条在阈内
        mk("controller", 10, 10, 105),   # 监督区 116 —— 让 controller 的中位≠最短
        mk("proposer", 10, 10, 1500),    # 监督区 1511；prompt 短、full 长 ⇒ 砍尾
        mk("critic", 400, 400, 5),       # 监督区 16；prompt 自己就超限 ⇒ 零梯度
    ]

    def lens(t):
        msgs = [{"role": "system", "content": t["system"]},
                {"role": "user", "content": t["user"]},
                {"role": "assistant", "content": t["response"]}]
        return (len(tok.apply_chat_template(msgs[:-1], tokenize=False,
                                            add_generation_prompt=True)),
                len(tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)))

    L = [lens(t) for t in turns]
    # 阈值取 critic 的 prompt 长度：这样 critic 恰好满足 prompt >= ml（零梯度），
    # proposer 的 full 远超阈而 prompt 远低于阈（砍尾），两条 controller 都在阈内。
    ml = L[3][0]
    assert L[0][1] <= ml and L[1][1] <= ml, L  # controller 不超限
    assert L[2][0] < ml < L[2][1], L           # proposer 砍尾
    assert L[3][0] >= ml and L[3][1] > ml, L   # critic 零梯度

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            census_truncation(tok, path, max_lens=(ml,))
        out = buf.getvalue()
    finally:
        os.unlink(path)

    # ── 三分法 ────────────────────────────────────────────────────────────
    line = [l for l in out.splitlines() if f"max_len={ml}" in l]
    assert len(line) == 1, out
    # 去掉 markdown 的 ** 强调号再匹配：不让断言挂在排版符号上
    line = line[0].replace("*", "")
    assert "超限 2" in line, f"超限该是 2（proposer+critic）：{line}"
    assert "零梯度 1" in line, f"零梯度该是 1（critic）：{line}"
    assert "砍尾 1" in line, f"砍尾该是 1（proposer）：{line}"
    # 三档各自的按角色拆分：超限那档是本轮才补的，此前只有零梯度那档有
    over_part, rest = line.split("其中", 1)
    dead_part, tail_part = rest.split("砍尾", 1)
    assert "'proposer': 1" in over_part and "'critic': 1" in over_part, over_part
    assert "controller" not in over_part, f"controller 没超限，不该出现：{over_part}"
    assert "'critic': 1" in dead_part and "proposer" not in dead_part, dead_part
    assert "'proposer': 1" in tail_part and "critic" not in tail_part, tail_part

    # ── 监督区长度 ────────────────────────────────────────────────────────
    # 期望值写死成字面数字，不由实现算出来——否则"中位数怎么取"这件事就测不到。
    # controller 两条的监督区是 16 和 116：**上**中位数（srt[n//2]）给 116，
    # 下中位数给 16。必须是 116，因为闸门用的就是 srt[len(srt)//2]，两处一致。
    want = {                    # role: (n, 中位, 最短, 最长)
        "controller": (2, 116, 16, 116),
        "proposer":   (1, 1511, 1511, 1511),
        "critic":     (1, 16, 16, 16),
    }
    for role, expect in want.items():
        rows = [l for l in out.splitlines() if l.strip().startswith(role)]
        assert len(rows) == 1, f"{role} 的监督区行没打出来或打了多行：{out}"
        nums = tuple(int(x) for x in rows[0].replace("n=", " ").split()
                     if x.lstrip("-").isdigit())
        assert nums == expect, \
            f"{role} 的 (n, 中位, 最短, 最长) 该是 {expect}，实得 {nums}：{rows[0]}"
    # 整体：监督区 [16, 116, 1511, 16] ⇒ 排序 [16, 16, 116, 1511] ⇒ 上中位 116
    overall = [l for l in out.splitlines() if "监督区 token 数" in l]
    assert len(overall) == 1, out
    assert overall[0].rstrip().endswith("116"), \
        f"整体监督区中位该是 116（上中位数，与闸门同）：{overall[0]}"


def test_census_and_gate_measure_over_and_dead_with_one_ruler():
    """`census_truncation` 与体检闸门对"超限/零梯度"必须是同一个谓词，逐字一致。

    为什么单独钉：`census_truncation` 的 docstring 里写了"口径必须与
    `submit_primus_sft.sh` 的闸门逐字一致"，但在这条测试之前**没有任何东西执行
    这句话**——那正是本项目已经踩过五次的"两把尺子"的标准形态：一句"请保持同步"
    的注释，加上两处会各自漂移的实现。闸门那边 `assert not dead` 会拦作业，探针
    这边只报数；两者若给出不同的数，人只会相信先看到的那一个。

    同时把 census 的两处 `apply_chat_template` 也钉到 `train_sft.py` 上——它是渲染
    chat template 的**第三处**（另两处：train_sft.py:37-38、闸门 heredoc），
    前两处已由 `test_sft_health_gate_shares_one_ruler_with_train_sft` 钉住。
    """
    import ast
    import pathlib
    import re

    from check_sft_template import _extract_calls

    root = pathlib.Path(__file__).parent

    def norm(expr):
        """把阈值变量名归一：闸门叫 max_len，census 里循环变量叫 ml。"""
        return re.sub(r"\b(max_len|ml)\b", "ML", expr)

    def grab(tree, names):
        got = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in names):
                got[node.targets[0].id] = norm(ast.unparse(node.value))
        return got

    # ── 闸门侧：从 heredoc 里抠出那段 python ──────────────────────────────
    sh = (root / "submit_primus_sft.sh").read_text(encoding="utf-8")
    m = re.search(r'^python - "\$\{MODEL_PATH\}" "\$\{MAX_LEN\}" <<', sh, re.M)
    assert m, "闸门 heredoc 的调用形状变了，本测试已过期"
    body = sh[m.end():]
    body = body[body.index("\n") + 1: body.index("\nEOF\n")]
    gate = grab(ast.parse(body), {"over", "dead"})
    assert set(gate) == {"over", "dead"}, f"闸门里 over/dead 的赋值变了：{sorted(gate)}"

    # ── 探针侧：只看 census_truncation 这个函数 ───────────────────────────
    probe_src = (root / "check_sft_template.py").read_text(encoding="utf-8")
    fn = next((n for n in ast.walk(ast.parse(probe_src))
               if isinstance(n, ast.FunctionDef) and n.name == "census_truncation"),
              None)
    assert fn, "check_sft_template.py 里找不到 census_truncation，本测试已过期"
    cen = grab(fn, {"over", "dead", "tail"})
    assert set(cen) == {"over", "dead", "tail"}, f"census 里的赋值变了：{sorted(cen)}"

    for k in ("over", "dead"):
        assert gate[k] == cen[k], (
            f"{k} 的口径漂了——闸门与探针会对同一份数据给出不同的数。\n"
            f"  闸门：{gate[k]}\n  探针：{cen[k]}")

    # `tail` 只在探针侧有（闸门不需要），但它必须恰好是 dead 的补集，否则
    # "超限 = 零梯度 + 砍尾"这个加法在打印里就不成立。
    assert cen["tail"] == cen["dead"].replace(">=", "<"), \
        f"tail 不是 dead 的补集：dead={cen['dead']}  tail={cen['tail']}"

    # ── 第三处渲染必须与 train_sft.py 同形 ────────────────────────────────
    cen_calls = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "attr", "") == "apply_chat_template"):
            cen_calls[node.targets[0].id] = (
                [ast.unparse(a) for a in node.value.args],
                {k.arg: ast.unparse(k.value) for k in node.value.keywords})
    assert set(cen_calls) == {"full", "prompt"}, \
        f"census 里 apply_chat_template 的赋值目标变了：{sorted(cen_calls)}"
    train = _extract_calls((root / "train_sft.py").read_text(encoding="utf-8"))
    for tname, cname in (("full_text", "full"), ("prompt_text", "prompt")):
        assert train[tname] == cen_calls[cname], (
            f"census 的 {cname} 与 train_sft.py 的 {tname} 漂开了——普查量的不是训练"
            f"实际吃的序列。train={train[tname]}  census={cen_calls[cname]}")


def test_sft_health_gate_shares_one_ruler_with_train_sft():
    """体检闸门与 `train_sft.py` 必须是同一把尺子，且 max_len 只有一个来源。

    背景：这个闸门原先用 `chars/2.2 > 1024` 这把代理尺量截断率，集群上用真
    tokenizer 一量发现偏乐观 3.7 倍——1024 下真实截断 5.73% 早已越过 2% 阈值，
    闸门却因为尺子偏软而放行。这是本项目第三次踩「两把尺子漂移」。

    修法是把闸门换成真 tokenizer。但这么一改，`submit_primus_sft.sh` 的 heredoc
    就成了**第三处**渲染 chat template 的地方（另两处是 train_sft.py:37-38 和
    check_sft_template.py）。第三处若与 train_sft.py 漂开，就是把刚扫掉的缺陷
    原样请回来，而且同样不会报错——闸门照样打印一个看起来正常的百分比。

    所以这条钉两件事：
      ① heredoc 里那两处 apply_chat_template 的实参形状与 train_sft.py 逐字相同
         （**含「都不传 enable_thinking」这一点**：那是 train_sft.py 的现存缺陷，
         闸门必须镜像它，否则量的就不是训练实际吃的序列；修法单独一步）；
      ② max_len 只有一个来源 —— 闸门与 +sft.max_len 都读 ${MAX_LEN}，不许任何
         一侧写字面量。原先两处各写一个 1024，改了一处忘了另一处不会报错。
    """
    import ast
    import pathlib
    import re

    from check_sft_template import _extract_calls

    root = pathlib.Path(__file__).parent
    sh = (root / "submit_primus_sft.sh").read_text(encoding="utf-8")

    # ── ② max_len 单一来源 ────────────────────────────────────────────────
    assert re.search(r"^MAX_LEN=\$\{MAX_LEN:-\d+\}", sh, re.M), \
        "submit_primus_sft.sh 里找不到 MAX_LEN 的定义（max_len 又散成字面量了？）"
    assert '+sft.max_len="${MAX_LEN}"' in sh, \
        "+sft.max_len 没有读 ${MAX_LEN}——训练参数与闸门会各自漂"
    assert not re.search(r"\+sft\.max_len=\d", sh), \
        "+sft.max_len 写成了字面量"
    gate_call = re.search(r'^python - "\$\{MODEL_PATH\}" "\$\{MAX_LEN\}" <<', sh, re.M)
    assert gate_call, "闸门 heredoc 没有把 MODEL_PATH / MAX_LEN 作为 argv 传进去"

    # ── ① 闸门与 train_sft.py 同一把尺 ───────────────────────────────────
    body = sh[gate_call.end():]
    body = body[body.index("\n") + 1: body.index("\nEOF\n")]
    tree = ast.parse(body)
    # 查代理尺要在 AST 上查，不能查文本：heredoc 里有一句注释正是在说明「不用
    # chars/2.2」，文本匹配会把那句注释当成尺子还在，一条永远失败的空断言。
    floats = {n.value for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    assert 2.2 not in floats, f"闸门代码里还留着 chars/2.2 代理尺（常量：{floats}）"
    assert "AutoTokenizer" in body, "闸门没有用真 tokenizer"

    gate = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "attr", "") == "apply_chat_template"):
            gate[node.targets[0].id] = (
                [ast.unparse(a) for a in node.value.args],
                {k.arg: ast.unparse(k.value) for k in node.value.keywords})
    assert set(gate) == {"full", "prompt"}, \
        f"闸门里 apply_chat_template 的赋值目标变了：{sorted(gate)}"

    train = _extract_calls((root / "train_sft.py").read_text(encoding="utf-8"))
    for tname, gname in (("full_text", "full"), ("prompt_text", "prompt")):
        assert train[tname] == gate[gname], (
            f"闸门的 {gname} 与 train_sft.py 的 {tname} 漂开了 —— 闸门量的不是训练"
            f"实际吃的序列。train={train[tname]}  gate={gate[gname]}")

    # ── 零梯度这一档必须被 assert 守住，不能只打印 ────────────────────────
    assert re.search(r"assert not dead", body), \
        "闸门只统计零梯度却不 assert，等于一个看起来在测量、实际不拦的埋点"


def test_request_context_drops_the_duplicate_quote():
    """响应方 prompt 里「对方内容」若与已给过的 `待审查解法` 同源，就不再重复一遍。

    实测规模（`data/sft_train_v23.jsonl` 渲染出的 310 处「对方内容」）：**247 处是
    同一个 prompt 内的逐字重复**（critic 132 / verifier 115），每处约 263 字，合计约
    6.5 万字符；而且被 `MAX_CHANNEL_CHARS` 截断的恰恰是这份多余的拷贝——同 prompt
    里那份走的是 `MAX_REASONING_CHARS`(1500)，更宽。所以"窗口不够"这个表象下面，
    近一半其实是"同一段话发了两遍，第二遍被砍了"。

    这条测试有三个断言，缺一个都不够：
      ① 重复被去掉（推理正文在 critic 的 prompt 里只出现一次）；
      ② **意图行必须留着**——`agents/parsing.py` 里剥块的理由正是"发起意图已由
         `request_context` 的 action/reason 显式表达"，把意图行也省掉等于把那条
         理由抽空；
      ③ 阳性对照：发起方**不是** proposer 时（critic→verifier），「对方内容」是
         critic 的批评而 `推理：` 是 proposer 的推理，两份不同，**必须保留**。
         实测这类有 63 处，去重写错方向就会把它们误删——那是丢真信息，比重复更坏。
    """
    reasoning = "第一步设 x 为未知数，第二步代入得 6*7"
    prop_out = f"推理过程：{reasoning}\n最终答案：41\n"

    # ── ①② proposer → critic：对方内容与待审查解法同源，应去重 ────────────
    # controller 必须先 continue：`stop_gate=False` 时第一句 stop 会让 episode 在
    # proposer 之前就终止，交互链一跳都不发生，断言全部空跑（本条第一版就是这么
    # 假绿的——`chain == []` 而 `count(reasoning) == 0` 也满足"只出现一次"以外的
    # 任何弱断言）。
    _CTRL = ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
             "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>"]
    script = {
        "controller": list(_CTRL),
        "proposer": [
            prop_out + INTER.format(a="request", t="critic"),
            # critic 标错会**机械触发** proposer 修正（`target="proposer"` 写死），
            # 占掉 hop2，所以这里必须有第二条 proposer 脚本。本条第一版漏了它，
            # 报的是 FakeEngine 的 `pop from empty list` —— 一个和被测逻辑毫无关系
            # 的错误信息，查起来比断言失败慢得多。
            "推理过程：fix\n最终答案：42\n" + INTER.format(a="none", t="none"),
        ],
        "critic": ["错误分析：第二步把 6*7 算成了 41\n" + INTER.format(a="none", t="none")],
        "verifier": ["分数: 0.9\n验证说明：ok\n" + INTER.format(a="none", t="none")] * 4,
    }
    eng = FakeEngine(script)
    _mk_executor(eng, stop_gate=False).run_episodes_batch(["6*7=?"], ["42"])

    cp = [p for role, p in eng.prompts if role == "critic"][0]
    assert cp.count(reasoning) == 1, \
        f"推理在 critic 的 prompt 里出现了 {cp.count(reasoning)} 次（去重没生效）"
    assert "对方内容：" not in cp, "重复的引述应整行省掉"
    assert "发起了交互" in cp and "理由：" in cp, \
        "意图行被一起省掉了——剥块的理由就建立在它还在上面"

    # ── ③ 阳性对照：critic → verifier，两份内容不同，必须保留 ──────────────
    # 这里 critic 必须输出「无错误」：`critic_found_errors` 为真时
    # `if target == "critic" and flagged` 会把它自己写的 `request verifier`
    # **覆盖**成写死的 proposer 修正，于是根本拿不到"发起方是 critic"的场景。
    # 本条第一版就栽在这上面——写了 request verifier 却被硬触发吃掉，测的是另一件事。
    script2 = {
        "controller": list(_CTRL),
        "proposer": [prop_out + INTER.format(a="request", t="critic")],
        "critic": ["错误分析：无错误，推导可以再核一遍第二步\n"
                   + INTER.format(a="request", t="verifier")],
        "verifier": ["分数: 0.3\n验证说明：确有问题\n" + INTER.format(a="none", t="none")] * 4,
    }
    eng2 = FakeEngine(script2)
    _mk_executor(eng2, stop_gate=False, max_hops=3).run_episodes_batch(["6*7=?"], ["42"])
    vp = [p for role, p in eng2.prompts if role == "verifier"][0]
    assert "对方内容：" in vp, \
        "发起方是 critic 时「对方内容」与「推理：」是两份不同的东西，去重把真信息删了"
    # 必须断到**具体内容**而不只是「对方内容：」这个标签：标签在而内容被截成空串
    # 同样是丢信息，而那种失效方式恰恰不会动标签。
    assert "推导可以再核一遍第二步" in vp, "critic 的实质意见没送到 verifier"

    # ── ④ 空串守卫：`last[0]` 解析为空时不许判成"已经给过了" ────────────────
    # 空串是**任何**字符串的子串，所以判据里少写 `_seen and` 这一半，就会在推理
    # 解析失败的 turn 上把发起方的真内容全部误删——而那种 turn 恰恰是最需要下游
    # 看到对方原文的（自己手里那份是空的）。上面 ③ 抓不到这一条：那里 `推理：`
    # 非空，空串分支根本走不到。变异验证里去掉 `_seen and` 只有这一段会响。
    script3 = {
        "controller": list(_CTRL),
        # 只输出块 → `parse_reasoning` 得到空串 → 黑板 trace 的推理是 ""
        "proposer": [INTER.format(a="request", t="critic")],
        "critic": ["错误分析：无错误，但请复核\n" + INTER.format(a="request", t="verifier")],
        "verifier": ["分数: 0.5\n验证说明：信息不足\n" + INTER.format(a="none", t="none")] * 4,
    }
    eng3 = FakeEngine(script3)
    _mk_executor(eng3, stop_gate=False, max_hops=3).run_episodes_batch(["6*7=?"], ["42"])
    # 前置断言：确认 fixture 真的造出了"推理为空"的场景，否则这一段测不到空串分支
    # （不要写 `assert ... or True` 那种恒真式——那正是本仓库反复在抓的假绿）。
    from agents.parsing import parse_reasoning
    assert parse_reasoning(INTER.format(a="request", t="critic"))[0] == "", \
        "fixture 失效：proposer 只输出块时推理应解析为空串"
    vp3 = [p for role, p in eng3.prompts if role == "verifier"][0]
    assert "无错误，但请复核" in vp3, \
        "推理解析为空时把发起方内容误删了——空串是任何串的子串，判据缺了空值守卫"

    # ── ⑤ 超长推理：`_seen` 末尾带 CLIP_MARK，比较前必须剥掉 ────────────────
    # `parse_reasoning` 对超 `MAX_REASONING_CHARS`(1500) 的推理会追加 `CLIP_MARK`，
    # 而那个标记**不在发起方原文里** → 子串判断恒假 → 去重静默失效（不报错、只是
    # prompt 又长了一倍）。实测这类 turn 占 8/599(1.3%)，量不大但失效方式是静默的，
    # 而且恰恰发生在 prompt 已经最长的那些 turn 上——正是最不该浪费预算的时候。
    from agents.parsing import CLIP_MARK, MAX_REASONING_CHARS
    long_r = "第一步推导" * (MAX_REASONING_CHARS // 4)     # 远超 1500 字
    assert len(long_r) > MAX_REASONING_CHARS
    script4 = {
        "controller": list(_CTRL),
        "proposer": [
            f"推理过程：{long_r}\n最终答案：41\n" + INTER.format(a="request", t="critic"),
            "推理过程：fix\n最终答案：42\n" + INTER.format(a="none", t="none"),
        ],
        "critic": ["错误分析：第二步错了\n" + INTER.format(a="none", t="none")],
        "verifier": ["分数: 0.9\n验证说明：ok\n" + INTER.format(a="none", t="none")] * 4,
    }
    eng4 = FakeEngine(script4)
    _mk_executor(eng4, stop_gate=False).run_episodes_batch(["6*7=?"], ["42"])
    cp4 = [p for role, p in eng4.prompts if role == "critic"][0]
    assert CLIP_MARK in cp4, "fixture 失效：这段推理没触发 MAX_REASONING_CHARS 截断"
    assert "对方内容：" not in cp4, \
        "推理被 CLIP_MARK 截断时去重失效了——比较前没剥掉标记，子串判断恒假"


def test_vote_counterfactual_arms_do_not_touch_the_production_path():
    """2×2 反事实臂是旁挂观测；生产的 `final_answer` / `is_correct` 一位都不许变。

    这条是本轮改动的**硬约束**。四臂的存在意义是"不必真打开 `correction_in_vote`
    就知道该不该打开"，可一旦哪个臂的 `exclude` 串到生产那一路，acc 的变化就会被
    误读成开关的效果——那比没有反事实更坏。

    构造要小心**平票**：只有一轮时 `excl` 池是 `{'5':1}`、`incl` 是 `{'5':1,'42':1}`，
    `Counter.most_common` 平票按插入序取第一个，于是两臂都判错、断言一点劲都吃不到
    （本条第一版正是这样，被自己那句"两个臂必须真的分歧"的前置断言抓住了——这类
    前置断言值得多写，它保护的是断言的**有效性**而不是被测代码）。
    所以改成两轮：错答案分别是 5 和 7，两次修正都给 42。于是
    `excl = {'5':1,'7':1}`（平票取 5，错），`incl = {'5':1,'7':1,'42':2}`（取 42，对）。
    """
    _W = "推理过程：{r}\n最终答案：{a}\n"

    def _run(corr_in_vote):
        script = {
            "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
                           "<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
                           "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>"],
            # 顺序 = 轮1 primary → 轮1 修正(hop2) → 轮2 primary → 轮2 修正(hop2)
            "proposer": [
                _W.format(r="x", a="5") + INTER.format(a="request", t="critic"),
                _W.format(r="fix", a="42") + INTER.format(a="none", t="none"),
                _W.format(r="y", a="7") + INTER.format(a="request", t="critic"),
                _W.format(r="fix", a="42") + INTER.format(a="none", t="none"),
            ],
            "critic": ["错误分析：算错了\n" + INTER.format(a="none", t="none")] * 2,
            "verifier": ["分数: 0.9\n验证说明：ok\n" + INTER.format(a="none", t="none")] * 4,
        }
        ex = _mk_executor(FakeEngine(script), stop_gate=False,
                          correction_in_vote=corr_in_vote)
        return ex.run_episodes_batch(["6*7=?"], ["42"])[0]

    off = _run(False)
    for k in ("is_correct_uni_excl", "is_correct_wt_excl",
              "is_correct_uni_incl", "is_correct_wt_incl"):
        assert k in off, f"四臂缺 {k}"
    # 两个臂必须真的分歧，否则下面的断言不吃劲
    assert off["is_correct_uni_excl"] is not off["is_correct_uni_incl"], \
        "excl 与 incl 判定相同——这个 fixture 测不到任何东西，先修 fixture"
    # 关闭时生产 == 排除臂
    assert off["is_correct"] == off["is_correct_uni_excl"], \
        "correction_in_vote=False 时生产路径应等于排除臂"

    on = _run(True)
    # 打开时生产 == 进池臂，而**四臂的语义一个都不许跟着翻**
    assert on["is_correct"] == on["is_correct_uni_incl"], \
        "correction_in_vote=True 时生产路径应等于进池臂"
    for k in ("is_correct_uni_excl", "is_correct_wt_excl",
              "is_correct_uni_incl", "is_correct_wt_incl"):
        assert on[k] == off[k], (
            f"{k} 随生产开关变了——四臂必须方向恒定，否则 d_corr 的符号会跟着翻，"
            f"而图上完全看不出来")
    # 旧名钉在排除臂上（wandb 历史曲线的含义不许静默漂移）
    for r in (off, on):
        assert r["is_correct_uniform"] == r["is_correct_uni_excl"]
        assert r["is_correct_weighted"] == r["is_correct_wt_excl"]


def test_correction_funnel_counts_both_directions():
    """修正漏斗必须双向计数：flip（错→对）与 unflip（对→错）不能塌成一个数。

    v3 step150 的 `fnl=561/566/76` 只有单向的 76，而 `correction_in_vote` 那条注释
    记的是净效应为负——两者同时成立只能推出反向 **>76**，具体多少现有落盘一个字节
    都查不到。于是"打开开关后 acc 变 0.01"分不清是"帮 76 毁 70"还是"帮 200 毁 194"，
    而这两种情形下一步该做的事完全相反。

    另断一条容易漏的语义：**没有修正 turn 的轮上两个计数都必须是 0**。`p_end` 在
    无修正时缺省等于 `p_prim`，所以两个合取天然为假；写成别的形式（比如用
    `corrected_answer is None` 当哨兵）就会在这里出错。
    """
    from agents.raca_rewards import compute_turn_data
    from training.metrics import rollout_metrics

    def _rec(prim, corr):
        return {
            "u": True, "forced": False, "target": "critic",
            "primary_tid": 0, "primary_answer": prim,
            "corrected_answer": corr, "primary_parsed": True,
            "ctrl_tid": None, "gate_blocked": False,
            "critic_turns": [{"tid": 1, "flagged": True, "reviewed_answer": prim,
                              "correction_followed": corr is not None}],
            "verifier_turns": [], "sigma": "verify",
            "correction_turns": ([{"tid": 2, "answer": corr}] if corr else []),
        }

    # 三轮：错→对（flip）、对→错（unflip）、无修正（两者都不算）
    recs = [_rec("5", "42"), _rec("42", "5"), _rec("42", None)]
    _, meta = compute_turn_data(recs, "42", True, 4, CFG,
                               stop_ctrl_tid=None, stop_sigma="verify")
    assert [m["flip"] for m in meta] == [True, False, False], \
        f"flip 方向错了：{[m['flip'] for m in meta]}"
    assert [m["unflip"] for m in meta] == [False, True, False], \
        f"unflip 方向错了：{[m['unflip'] for m in meta]}"

    st = rollout_metrics([[{"is_correct": True, "raca_round_meta": meta,
                            "raca_turn_data": {}, "stopped": True}]])
    assert st["funnel_flip"] == 1 and st["funnel_unflip"] == 1, \
        f"漏斗聚合塌了：flip={st['funnel_flip']} unflip={st['funnel_unflip']}"
    assert st["funnel_corr"] == 2, "两次修正应都计入 corr"


def test_funnel_unflip_survives_the_round_meta_whitelist():
    """`unflip` 要走完 round_records → compute_turn_data → round_meta → metrics 四跳。

    #23 正是在第三跳栽的：`round_meta` 是 `compute_turn_data` **重新拼的白名单
    dict、不是 round_records 的透传**，漏带一手的后果是指标照样打印、但永远 0.00——
    比不加埋点更坏，因为它长得和真读数一模一样。所以这条不测方向（上一条测了），
    只测"这个键真的活着穿过了白名单"。
    """
    import inspect

    from agents import raca_rewards
    src = inspect.getsource(raca_rewards.compute_turn_data)
    assert '"unflip"' in src, \
        "round_meta 的白名单里没有 unflip —— metrics 会读到永远 0.00 的假读数"

    from agents.raca_rewards import compute_turn_data
    from training.metrics import rollout_metrics

    # 用真实产物再把键删掉，模拟"旧 round record"——比手搓字典更贴近真实退化场景
    # （手搓的还容易漏掉 metrics 需要的必填键，那样测的就是 fixture 而不是代码）。
    rec = {
        "u": True, "forced": False, "target": "critic",
        "primary_tid": 0, "primary_answer": "5", "corrected_answer": "42",
        "primary_parsed": True, "ctrl_tid": None, "gate_blocked": False,
        "critic_turns": [{"tid": 1, "flagged": True, "reviewed_answer": "5",
                          "correction_followed": True}],
        "verifier_turns": [], "sigma": "verify",
        "correction_turns": [{"tid": 2, "answer": "42"}],
    }
    _, meta = compute_turn_data([rec], "42", True, 4, CFG,
                                stop_ctrl_tid=None, stop_sigma="verify")
    old = [{k: v for k, v in meta[0].items() if k != "unflip"}]
    st = rollout_metrics([[{"is_correct": True, "raca_turn_data": {},
                            "stopped": True, "raca_round_meta": old}]])
    assert st["funnel_unflip"] == 0, "缺键时应低报为 0，不能虚报"
    assert st["funnel_flip"] == 1, "同一条记录里的 flip 仍应读得到"


def test_three_hop_chain_can_reach_verifier():
    """`max_hops: 3` 打开的那条新路径：critic 标错 → proposer 修正 → verifier 打分。

    这条测试的存在本身是个警告：改到第十轮之前，**58 条测试里没有一条覆盖 3 跳链**
    （`_mk_executor` 默认 max_hops=2，只有 max_hops=0 的消融），也就是说加预算之后
    第一次真正跑这条路径会是在集群上。

    要记住的限定：第 3 跳**没有机制把守**。第 2 跳里 critic 标错触发 proposer 修正
    是写死的（`target="proposer"`），但第 3 跳要求**修正后的 proposer 自己写出**
    `request verifier`。所以这条测试的脚本里那个块是刻意写出来的——它测的是"路径
    通不通"，不是"模型会不会走"。会不会走由跑起来之后的 `hop=` 读数回答。
    """
    script = {
        # 先 continue 再 stop —— 第一句 stop 会让 episode 在 proposer 之前就终止
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
                       "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>"],
        "proposer": [
            "推理过程：x\n最终答案：5\n" + INTER.format(a="request", t="critic"),
            # 修正 turn 自己发起第 3 跳（没有机制会替它发）
            "推理过程：fix\n最终答案：42\n" + INTER.format(a="request", t="verifier"),
        ],
        "critic": ["错误分析：第二步错了\n" + INTER.format(a="none", t="none")],
        "verifier": ["分数: 0.9\n验证说明：修正后正确\n" + INTER.format(a="none", t="none")] * 3,
    }
    eng = FakeEngine(script)
    ex = _mk_executor(eng, stop_gate=False, max_hops=3)
    ex.run_episodes_batch(["6*7=?"], ["42"])
    chain = [r for r in eng.calls if r != "controller"]
    assert chain == ["proposer", "critic", "proposer", "verifier"], \
        f"3 跳链没走通：{chain}"

    # 对照：同一个脚本在 max_hops=2 下第 3 跳必须被预算拦掉
    eng2 = FakeEngine({k: list(v) for k, v in script.items()})
    _mk_executor(eng2, stop_gate=False, max_hops=2).run_episodes_batch(["6*7=?"], ["42"])
    chain2 = [r for r in eng2.calls if r != "controller"]
    assert chain2 == ["proposer", "critic", "proposer"], \
        f"max_hops=2 下不该走到第 3 跳：{chain2}"


def test_hop_depth_counter_matches_the_actual_chain():
    """跳深计数器是 `max_hops` 2→3 的**唯一**验收读数，所以它自己得先准。

    只统计真发出去的请求（空批次不记），并按响应方角色分开——`max_hops: 3` 想要的
    具体事件是"第 3 跳到达 verifier"（修正后的答案在轮内被打分），只看总数分不出
    它到的是 verifier 还是又一次 critic。
    """
    script = {
        # 先 continue 再 stop —— 第一句 stop 会让 episode 在 proposer 之前就终止，
        # 计数器就恒为空，断言反而"通过"不了但也测不到东西。
        "controller": ["<meta-plan>\ndecision: continue\nreason: r\n</meta-plan>",
                       "<meta-plan>\ndecision: stop\nreason: r\n</meta-plan>"],
        "proposer": [
            "推理过程：x\n最终答案：5\n" + INTER.format(a="request", t="critic"),
            "推理过程：fix\n最终答案：42\n" + INTER.format(a="request", t="verifier"),
        ],
        "critic": ["错误分析：第二步错了\n" + INTER.format(a="none", t="none")],
        "verifier": ["分数: 0.9\n验证说明：ok\n" + INTER.format(a="none", t="none")] * 3,
    }
    eng = FakeEngine(script)
    ex = _mk_executor(eng, stop_gate=False, max_hops=3)
    ex.run_episodes_batch(["6*7=?"], ["42"])
    hd = ex.n_hop_depth
    assert hd[1] == 1 and hd[2] == 1 and hd[3] == 1, f"跳深总数不对：{dict(hd)}"
    assert hd["1:critic"] == 1, "第 1 跳应到 critic"
    assert hd["2:proposer"] == 1, "第 2 跳应是机械触发的 proposer 修正"
    assert hd["3:verifier"] == 1, "第 3 跳应到 verifier —— 这正是加预算想买到的事件"
    assert 4 not in hd, "max_hops=3 却记到了第 4 跳"

    # 每批次必须重置，否则 train.py 读到的是累计值而不是本步值
    ex.run_episodes_batch([], [])
    assert not ex.n_hop_depth, "n_hop_depth 没有按批次重置"


def test_absorb_counters_keeps_batch_and_cumulative_semantics_apart():
    """分片计数器聚合：批次级用赋值、累计级用累加，两者不能混。

    背景：`train.py` 多引擎时走分片 rollout，每步为每个分片新建临时 executor，
    而 step 行读 `trainer.executor`。08-28 两跑（7 个引擎）因此四个诊断读数全程
    是假的 0 —— `hop=` 一次没打印（`max_hops` 2→3 的唯一验收指标失效）、
    `clip_prompt=0` 不能证明 prompt 没超预算（窗口 600 的第一号风险项）。

    这条测试钉的是**最容易写错的那一点**：两类计数器语义不同。
      - 批次级（`run_episodes_batch` 开头重置）→ 赋值。写成 `+=` 会变成整轮累计，
        只增不减，被误读成"越来越糟"。
      - 累计级（只在 `__init__` 归零）→ 累加。写成赋值会每步覆盖、丢掉历史，
        而"整轮有没有超过预算过"只有累计值答得了。
    所以要连着 absorb 两次，才能把这两种错法都暴露出来 —— 只 absorb 一次的话，
    赋值和累加在数值上看不出区别。
    """
    from collections import Counter

    from agents.agentic_executor import AgenticExecutor

    class _Fake:
        def __init__(self, gate, self_t, split_f, split_why, gap, decode_f,
                     lp_f, clip, hops):
            self.n_gate_unlocked = gate
            self.n_self_target = self_t
            self.n_credit_split_failed = split_f
            self.n_credit_split_failures = Counter(split_why)
            self.n_credit_boundary_tokens = gap
            self.n_credit_decode_fallback = decode_f
            self.n_logprob_mismatch = lp_f
            self.n_prompt_clipped = clip
            self.n_hop_depth = Counter(hops)

    dst = object.__new__(AgenticExecutor)
    dst.n_gate_unlocked = 0
    dst.n_self_target = 0
    dst.n_credit_split_failed = 0
    dst.n_credit_split_failures = Counter()
    dst.n_credit_boundary_tokens = 0
    dst.n_credit_decode_fallback = 0
    dst.n_logprob_mismatch = 0
    dst.n_prompt_clipped = 0
    dst.n_hop_depth = Counter()

    shards = [
        _Fake(2, 1, 2, {"text_after_block": 2}, 4, 3, 3, 3,
              {1: 10, 2: 4, "1:critic": 7}),
        _Fake(5, 0, 4, {"text_after_block": 1, "visible_id_mismatch": 3},
              6, 2, 5, 1, {1: 6, 3: 2, "3:verifier": 2}),
    ]

    dst.absorb_counters(shards)
    assert dst.n_gate_unlocked == 7 and dst.n_self_target == 1
    assert dst.n_credit_split_failed == 6, "credit split 失败数没有跨分片求和"
    assert dst.n_credit_split_failures == {
        "text_after_block": 3, "visible_id_mismatch": 3}, \
        "credit split 原因分布没有跨分片合并"
    assert dst.n_credit_boundary_tokens == 10
    assert dst.n_credit_decode_fallback == 5
    assert dst.n_logprob_mismatch == 8, "logprob mismatch 没有跨分片求和"
    assert dst.n_prompt_clipped == 4, "累计级应累加"
    assert dst.n_hop_depth[1] == 16 and dst.n_hop_depth[2] == 4
    assert dst.n_hop_depth[3] == 2, "跳深要按键相加，不能只取最后一个分片"
    assert dst.n_hop_depth["1:critic"] == 7
    assert dst.n_hop_depth["3:verifier"] == 2, "按角色分的键不能丢"

    # 第二步（模拟下一个 step）：批次级必须**重新赋值**，累计级必须继续累加
    dst.absorb_counters(shards)
    assert dst.n_gate_unlocked == 7, \
        f"批次级被累加了（{dst.n_gate_unlocked}）—— 会被误读成越来越糟"
    assert dst.n_credit_split_failed == 6, "split failure 被错误地跨批次累加了"
    assert dst.n_credit_split_failures["text_after_block"] == 3
    assert dst.n_credit_split_failures["visible_id_mismatch"] == 3
    assert dst.n_credit_boundary_tokens == 10
    assert dst.n_credit_decode_fallback == 5
    assert dst.n_logprob_mismatch == 8, "logprob mismatch 被错误地跨批次累加了"
    assert dst.n_hop_depth[1] == 16, "跳深没有按批次重置"
    assert dst.n_prompt_clipped == 8, \
        f"累计级被覆盖了（{dst.n_prompt_clipped}）—— 丢掉了历史"

    # 空分片列表：批次级归零，累计级不动（对应"这一步没有任何分片"）
    dst.absorb_counters([])
    assert dst.n_gate_unlocked == 0 and not dst.n_hop_depth
    assert dst.n_credit_split_failed == 0
    assert not dst.n_credit_split_failures
    assert dst.n_credit_boundary_tokens == 0
    assert dst.n_credit_decode_fallback == 0
    assert dst.n_logprob_mismatch == 0
    assert dst.n_prompt_clipped == 8


def test_sharded_rollout_path_absorbs_counters():
    """源码级不变量：`train.py` 的分片路径必须把分片计数器聚合回 trainer.executor。

    为什么只能用源码级：分片路径要求 `vllm_engine.engines` 存在且 >1，没有真 vLLM
    引擎任何行为测试都到不了 —— 这正是那四个计数器能在集群上静默失效整整两跑的
    原因（本地单引擎路径读 `trainer.executor`，恰好是对的，所以本地永远看不出问题）。
    第三次遇到"测不到的代码"，手法照旧：跑不到就用源码守，并做变异验证。

    **为什么走 AST 而不是原文 grep**：第一版写的是 `assert "absorb_counters" in src`，
    变异验证当场打脸 —— 把真实调用删掉换成 `pass`，断言**照样通过**，因为
    `_run_chunk` 里我自己写的注释提到了 `AgenticExecutor.absorb_counters`。
    这与第四轮 `test_no_bare_lt_lookahead_in_parsers` 栽的是同一个坑（被自己的
    文档绊倒），已经是第三次。AST 天然排除注释与 docstring，只看会真正执行的代码。
    """
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent / "train.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)

    # ① 必须有真实的 absorb_counters 调用（注释里提到不算）
    absorb = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call)
              and getattr(n.func, "attr", "") == "absorb_counters"]
    assert absorb, \
        "分片路径没有真正调用 absorb_counters —— hop/clip_prompt/selfT 会恒为 0"

    # ② 聚合必须发生在**读计数器之前**，顺序错了照样读到聚合前的值
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "getattr"
             and len(n.args) >= 2
             and isinstance(n.args[1], ast.Constant)
             and n.args[1].value == "n_hop_depth"]
    assert reads, "找不到读取 n_hop_depth 的位置，本不变量的前提变了"
    assert min(a.lineno for a in absorb) < min(r.lineno for r in reads), \
        "absorb_counters 在读取跳深之后 —— 顺序错了，读到的仍是聚合前的 0"

    # ③ `_run_chunk` 的 return 必须是二元组且第二项是 executor 本身，
    #    否则外面拿不到分片实例，聚合无从谈起
    fn = [n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "_run_chunk"]
    assert fn, "分片路径的 _run_chunk 不见了，本不变量的前提变了"
    rets = [n for n in ast.walk(fn[0]) if isinstance(n, ast.Return)]
    assert len(rets) == 1 and isinstance(rets[0].value, ast.Tuple), \
        "_run_chunk 应恰好返回一个二元组（结果, executor）"
    second = rets[0].value.elts[1]
    assert isinstance(second, ast.Name) and second.id == "ex", \
        f"_run_chunk 返回的第二项不是 executor（实际 {ast.dump(second)[:60]}）"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"All {len(fns)} v2 tests passed.")
