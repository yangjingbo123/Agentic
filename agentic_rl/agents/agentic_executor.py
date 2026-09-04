"""RACA v2 执行器：controller 仅终止决策，交互是唯一路由机制。

流程（§2.1）：每轮
  1. σ 机械推导 → 2. Controller continue/stop（stop 闸门：需黑板存在 verifier 分数）
  3. Proposer 固定起点：解题 + 交互决策 {none|request|challenge}×{critic|verifier}
  4. 交互链（≤ max_hops 跳响应）：响应方以标准角色格式回应、追加写入（不覆盖）；
     critic 可 request proposer 修正，修正输出作为新 trace 进入多数投票
  5. ε 强制注入：proposer 未发起时以 ε_t 概率强制一次 critic 审查（冷启动保护）

奖励计算委托给 agents/raca_rewards.compute_turn_data（纯 Python，可单测）。
"""

import random
from collections import Counter

from agents.grader import math_equal   # re-export：evaluate.py 从此模块导入
from agents.parsing import (
    CLIP_MARK,
    MAX_CHANNEL_CHARS,
    ROLE_NAMES,
    critic_found_errors,
    has_answer_label,
    parse_decision,
    parse_interaction,
    parse_reasoning,
    parse_score,
    strip_interaction,
    trailing_interaction_span,
)
from agents.raca_rewards import compute_turn_data
from envs.blackboard import Blackboard, Message, MessageType
from llm.prompt_templates import PromptTemplates


class AgenticExecutor:
    def __init__(self, model, tokenizer, config, vllm_engine=None, eval_mode: bool = False):
        self.model = model              # v2 rollout 不再直接用训练模型前向，仅保留引用
        self.tokenizer = tokenizer
        self.cfg = dict(config)
        self.max_tokens = config.get("max_tokens", 512)
        self.max_rounds = config.get("max_rounds", 4)
        self.max_hops = config.get("max_hops", 2)        # 0 = 消融A：禁用交互
        self.stop_gate = config.get("stop_gate", True)   # stop 需存在 verifier 分数
        # v3.1：闸门解锁。想停但黑板无分数时，本轮末独立补一次 verifier
        # 打分（不占 hop 预算、与 proposer 选不选交互无关）。详见 3.5 步注释。
        self.gate_unlock = config.get("gate_unlock", True)
        # ε 强制注入时选 verifier 的概率（其余选 critic）。v2.0 固定注 critic，
        # 导致 verifier 零训练数据且 stop 闸门永不解锁。
        self.eps_verifier_share = config.get("eps_verifier_share", 0.5)
        self.vllm_engine = vllm_engine
        # eval_mode：greedy 解码 + 不做 ε 强制注入（评测学到的策略本身）
        self.eval_mode = eval_mode
        self.temperature = 0.0 if eval_mode else 1.0
        # v2.3：投票模式。weighted = verifier 分数加权（交互产出直接参与聚合，
        # 交互与投票从替代关系变互补关系）；uniform = 朴素多数投票（消融）。
        self.vote_mode = config.get("vote_mode", "uniform")
        # v3 测量（§19）：修正正确率两次 ≤ 裸重采样 → 修正票退出投票池
        # （仍进黑板供上下文/σ/r_int 因果信用，只是不计票）。
        self.correction_in_vote = config.get("correction_in_vote", True)
        # v3 测量（§19）：旧错解+批评的上下文对下一轮重答有锚定伤害
        # （通道② Δ=−0.085/−0.055）；False = FLAW 不进 primary prompt。
        self.flaw_in_primary = config.get("flaw_in_primary_prompt", True)
        # 新版将 proposer 的解题/交互优势路由到各自 token span；关闭时回退到
        # 历史的整 turn 标量优势，便于做严格消融和紧急回滚。
        self.token_credit = config.get("token_credit", True)
        # ── prompt 长度保险（v3.1） ────────────────────────────────────────
        # v3 实测 step 151：decoder prompt 5036 > max_model_len 4096 直接崩作业。
        # 已在 parsing/blackboard 侧堵住已知无界点，但任何嵌入自由文本的 prompt
        # 都可能再出现新的增长源，故在唯一出口 make_prompt 处兜底：超限则中段
        # 截断（保头保尾——头是任务指令，尾是格式要求），并计数暴露。
        # 宁可这一条 prompt 降质，也不能让 200 步的作业在第 151 步整体丢失。
        _model_len = config.get("vllm_max_model_len", 4096)
        self.max_prompt_tokens = config.get(
            "max_prompt_tokens", 0) or max(256, _model_len - self.max_tokens - 64)
        self.n_prompt_clipped = 0
        self.n_gate_unlocked = 0    # 本批次闸门解锁次数（健康指标：应随训练下降）
        self.n_self_target = 0      # proposer 自指 target（被归一为 none）的次数
        # Proposer primary 的文本块边界无法与 vLLM 原始 token_ids 精确对齐的次数。
        # 这类结构化 turn 整体 fail-closed 跳过 PG（仍可在其它统计中出现），因此
        # 必须可见；否则 tokenizer 一旦整体不匹配，双通道会静默停止训练。
        self.n_credit_split_failed = 0
        self.n_credit_split_failures = Counter()  # 失败原因分布（定位格式 vs tokenizer）
        # vLLM 返回的 response_ids / log_probs 长度不一致。扁平数组仍需补位保持后续
        # turn 对齐，但该 turn 必须标成无效并在 PPO 前整条跳过，不能把补的 0 当真值。
        self.n_logprob_mismatch = 0
        # 交互链按**跳深**分布（第十轮，`max_hops` 2→3 的配套读数）。
        # 为什么非要这个数：第 1 跳（proposer→critic/verifier）由 proposer 的块决定，
        # 第 2 跳里 critic 标错→proposer 修正是**机械触发**（写死 "proposer"），但
        # 第 3 跳（修正→verifier）**没有机制**，它要求修正后的 proposer 自己写出
        # `request verifier`。而这个仓库已经在同一个坑里栽过一次——`:377` 注释记着
        # "v2.1 实测 eff≈0：修正跳由**未学会的行为**把守，correction_turns 几乎恒空
        # → q≈0 → 发起恒负期望 → int_rate 塌"，所以才把它改成机械触发。
        # 于是 `max_hops: 3` 买到的只是**可能性**，不是保证。这个计数器就是判据：
        # depth3 若接近 0，说明该照 critic 那条的样子把它也做成机械触发（行为改动，
        # 单独一步）；若明显非 0，才说明加预算本身就够了。
        self.n_hop_depth = Counter()
        self._rng = random.Random()

    # ── batch entry point（N episodes 并行，逐 turn-slot 批量调 vLLM） ───────
    def run_episodes_batch(self, questions: list, correct_answers: list,
                           eps_force: float = 0.0) -> list:
        """运行 N 个 episode。eps_force：本批次强制注入 critic 的概率（训练时
        由 train.py 按线性衰减 schedule 传入；eval 默认 0）。"""
        if self.vllm_engine is None:
            raise RuntimeError("RACA v2 rollout 需要 vLLM 引擎（no_vllm 路径已移除）")
        if self.eval_mode:
            eps_force = 0.0
        self.n_gate_unlocked = 0    # 每批次重置；train.py 读到的就是本步值
        self.n_self_target = 0      # 同上
        self.n_credit_split_failed = 0  # 同上（双通道 token 边界）
        self.n_credit_split_failures.clear()
        self.n_logprob_mismatch = 0     # 同上（response/logprob 对齐）
        self.n_hop_depth.clear()    # 同上（跳深分布）

        n = len(questions)
        blackboards    = [Blackboard() for _ in range(n)]
        all_messages   = [[] for _ in range(n)]
        turn_ids_list  = [[] for _ in range(n)]
        log_probs_list = [[] for _ in range(n)]
        turn_counters  = [0] * n
        round_records  = [[] for _ in range(n)]
        corr_answers   = [[] for _ in range(n)]   # 修正票（可能被投票排除）
        stop_ctrl_tids = [None] * n
        stop_sigmas    = [None] * n
        active         = list(range(n))

        def next_tid(i):
            tid = turn_counters[i]
            turn_counters[i] += 1
            return tid

        def make_prompt(system, user):
            user = clip_user(system, user)
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )

        def clip_user(system, user):
            """prompt 长度保险：按 token 精确计数，超限取头尾各半。"""
            budget = self.max_prompt_tokens - len(
                self.tokenizer.encode(system, add_special_tokens=False))
            if budget < 64:
                return user
            ids = self.tokenizer.encode(user, add_special_tokens=False)
            if len(ids) <= budget:
                return user
            head = budget // 2
            self.n_prompt_clipped += 1
            if self.n_prompt_clipped == 1:
                print(f"[warn] prompt 超限已启用中段截断（{len(ids)}>{budget} tokens）"
                      f"；若 clip 计数持续增长说明仍有文本无界点", flush=True)
            return (self.tokenizer.decode(ids[:head])
                    + "\n…（中段已省略）…\n"
                    + self.tokenizer.decode(ids[-(budget - head):]))

        def interaction_token_span(text, token_ids):
            """把末尾 interaction 字符边界映射到原始 vLLM IDs。

            返回 ``((start, end), None)`` 或 ``(None, reason)``。失败原因必须保留，
            否则 ``splitF`` 升高时无法区分模型格式续写和 tokenizer round-trip 问题。
            """
            span = trailing_interaction_span(text)
            if span is None:
                close_tag = "</interaction>"
                close_start = text.rfind(close_tag)
                if close_start < 0:
                    return None, "no_close_tag"
                close_end = close_start + len(close_tag)
                if text[close_end:].strip():
                    return None, "text_after_block"
                open_tag = "<interaction>"
                open_start = text.rfind(open_tag, 0, close_start)
                if open_start < 0:
                    return None, "no_open_tag"
                return None, "malformed_tail_block"

            char_start, _ = span
            ids = list(token_ids)

            def visible_match(full_ids):
                full_ids = list(full_ids)
                if ids[:len(full_ids)] != full_ids:
                    return False, "visible_id_mismatch"
                extras = ids[len(full_ids):]
                special = set(getattr(self.tokenizer, "all_special_ids", []) or [])
                if not all(tok in special for tok in extras):
                    return False, "extra_non_special"
                return True, None

            try:
                encoded = self.tokenizer(
                    text, add_special_tokens=False, return_offsets_mapping=True)
                full_ids = encoded["input_ids"]
                offsets = encoded["offset_mapping"]
                if full_ids and isinstance(full_ids[0], list):
                    full_ids, offsets = full_ids[0], offsets[0]
                matched, _ = visible_match(full_ids)
                if matched and len(offsets) == len(full_ids):
                    for idx, (lo, hi) in enumerate(offsets):
                        if lo == char_start:
                            return (idx, len(full_ids)), None
                        if lo < char_start < hi:
                            return None, "boundary_inside_token"
            except (TypeError, KeyError, NotImplementedError, AttributeError):
                pass

            try:
                full_ids = list(self.tokenizer.encode(
                    text, add_special_tokens=False))
                prefix_ids = list(self.tokenizer.encode(
                    text[:char_start], add_special_tokens=False))
            except (TypeError, AttributeError):
                return None, "tokenizer_error"

            matched, reason = visible_match(full_ids)
            if not matched:
                return None, reason
            if len(prefix_ids) >= len(full_ids):
                return None, "empty_interaction_span"
            if full_ids[:len(prefix_ids)] != prefix_ids:
                return None, "prefix_not_stable"
            return (len(prefix_ids), len(full_ids)), None

        def record(i, role, system, user, result, tid, *, prompt_text=None,
                   split_credit=False):
            text, turn_lps, token_ids = result
            token_ids = list(token_ids)
            n_resp = len(token_ids)
            logprob_aligned = len(turn_lps) == n_resp
            if not logprob_aligned:
                self.n_logprob_mismatch += 1
            # 扁平存储必须保持每个 token 都有占位，否则后续 turn_ids 整体错位；
            # 但补出的 0 只用于维持索引，message 标记会让训练器整 turn 跳过。
            aligned = list(turn_lps) + [0.0] * max(0, n_resp - len(turn_lps))
            log_probs_list[i].extend(aligned[:n_resp])
            turn_ids_list[i].extend([tid] * n_resp)
            message = {
                "role_name":    role,
                "turn_id":      tid,
                "system":       system,
                "user":         user,
                # 保存 rollout 真正使用的完整 prompt。user 可能在 make_prompt() 内
                # 被截断；训练时重新从原始 system/user 渲染会改变条件上下文，导致
                # old/new logprob 不再对应同一序列。旧数据没有此字段时训练器再回退重建。
                "prompt_text":  prompt_text,
                "response":     text,
                "response_ids": token_ids,
                "logprob_aligned": logprob_aligned,
            }
            if split_credit:
                span, split_error = interaction_token_span(text, token_ids)
                message["interaction_span"] = span
                if span is None:
                    message["interaction_span_error"] = split_error
                    self.n_credit_split_failed += 1
                    self.n_credit_split_failures[split_error or "unknown"] += 1
            all_messages[i].append(message)
            return text

        def responder_prompt(i, target, initiator, action, reason, init_out, forced):
            """响应方 prompt：标准角色格式 + 请求上下文（forced 无请求上下文）。"""
            bb = blackboards[i]
            last = bb.traces[-1] if bb.traces else ("", "")
            if target == "critic":
                sys = PromptTemplates.critic_system()
                usr = f"待审查解法：{last[0]}\n答案：{last[1]}\n当前状态：{bb.to_text()}"
            elif target == "verifier":
                sys = PromptTemplates.verifier_system()
                usr = f"待验证答案：{last[1]}\n推理：{last[0]}\n当前状态：{bb.to_text()}"
            else:  # proposer 修正
                sys = PromptTemplates.proposer_system()
                # 去重：flaw 窗口放宽到 `MAX_CHANNEL_CHARS` 后，黑板的「发现问题」
                # 与下面 `proposer_correction_user` 的 `initiator_output` 在 critic
                # 硬触发路径上是**逐字节相同**的两份拷贝（同一个 `shown`、同样的
                # 窗口），白占一整个 `MAX_CHANNEL_CHARS` 的预算还让 prompt 自我重复。
                # 这里按内容比对来判定，而不是推断"initiator 是不是 critic"——critic
                # 未标错却主动请求修正时 flaws[-1] 是更早的另一条，那份信息是真的、
                # 不能扔。
                # 用常量而非字面量：这个比较**必须**与两处截断同宽，否则就是第三把
                # 尺子。比较宽度取小了，两份只在前缀相同、后文不同的内容会被判成同
                # 一条（误删真信息）；取大了则会因为其中一份已被截断而判成不同（去
                # 重失效，白付预算）。写成两个恰好相等的字面量，等于把这条耦合交给
                # 记性——已经栽过五次了。**第十轮把常量从 300 改成 600 时这里不用动
                # 一个字，这正是当初写成常量换来的。**
                dup = (bool(bb.flaws)
                       and bb.flaws[-1]["content"][:MAX_CHANNEL_CHARS]
                       == init_out[:MAX_CHANNEL_CHARS])
                usr = PromptTemplates.proposer_correction_user(
                    questions[i], ROLE_NAMES.get(initiator, initiator),
                    init_out, bb.to_text(include_flaws=not dup))
            if not forced and target != "proposer":
                # 去重（第十轮）：上面 critic / verifier 那两路的 `usr` 开头已经带了
                # `last[0]`（`待审查解法：` / `推理：`），而 `request_context` 的
                # 「对方内容」在**发起方就是产出 last 的那个 proposer** 时是同一段
                # 文本的第二份拷贝——且前者上限是 `MAX_REASONING_CHARS`(1500)、比
                # 后者的 `MAX_CHANNEL_CHARS` 宽，所以被截断的恰恰是那份多余的。
                # 实测（`data/sft_train_v23.jsonl` 渲染出的 310 处「对方内容」）：
                # **247 处可判为重复**（critic 132 / verifier 115），每处白占约 263
                # 字，合计约 6.5 万字符；**63 处是 verifier 独有**，那是发起方不是
                # proposer 的情形（如 critic→verifier），此时「对方内容」是 critic 的
                # 批评而 `推理：` 是 proposer 的推理，两份不同，扔了就是丢真信息。
                #
                # 所以判据**按内容比对，不按角色推断**——与上面 `dup` 同一条理由。
                # 比的是 `last[0]`（已经在 prompt 里的那份）是否被 `init_out` 覆盖：
                # `init_out` 是未解析的完整输出（含 `推理过程：` 标签与答案行），
                # `last[0]` 是它过 `parse_reasoning` 之后的推理正文，所以方向只能是
                # 「后者是前者的子串」，反过来写恒为假。空串不算重复（`last[0]` 为空
                # 时任何 `in` 都真，会把独有内容误删——这是本条最容易写错的地方）。
                # 一个会让这条去重静默失效的边界：`last[0]` 是 `parse_reasoning` 的
                # 产物，推理超 `MAX_REASONING_CHARS`(1500) 时它**末尾带 `CLIP_MARK`**，
                # 而那个标记不在 `init_out` 里 → 子串判断恒假 → 去重白做。实测这类
                # turn 是 8/599（1.3%），量不大但失效方式是静默的，所以比较前剥掉。
                _seen = last[0][:-len(CLIP_MARK)] \
                    if last[0].endswith(CLIP_MARK) else last[0]
                quoted = init_out if not (_seen and _seen in init_out) else None
                usr += PromptTemplates.request_context(
                    ROLE_NAMES.get(initiator, initiator), action, reason, quoted)
            return sys, usr, (last[1] if bb.traces else "")

        for _ in range(self.max_rounds):
            if not active:
                break

            # ── σ 机械推导（进入本轮前的黑板状态） ────────────────────────
            sigma_map = {i: blackboards[i].derive_sigma() for i in active}

            # ── 1. controller（batch）：continue / stop ────────────────────
            ctrl_info = []
            for i in active:
                sys = PromptTemplates.controller_system()
                usr = f"问题：{questions[i]}\n当前状态：{blackboards[i].to_text()}"
                prompt = make_prompt(sys, usr)
                ctrl_info.append((i, next_tid(i), sys, usr, prompt))

            ctrl_res = self.vllm_engine.generate_batch(
                [{"role": "controller", "prompt": prompt,
                  "temperature": self.temperature}
                 for _, _, _, _, prompt in ctrl_info]
            )

            still_active = []
            ep_st = {}   # 本轮活跃 episode 的 round 状态
            for (i, tid, sys, usr, prompt), res in zip(ctrl_info, ctrl_res):
                text = record(i, "controller", sys, usr, res, tid,
                              prompt_text=prompt)
                gate_blocked = False
                if parse_decision(text) == "stop":
                    # stop 闸门：黑板存在 verifier 分数才允许终止（§2.1 约束）
                    if (not self.stop_gate) or blackboards[i].scores:
                        stop_ctrl_tids[i] = tid
                        stop_sigmas[i] = sigma_map[i]
                        continue
                    gate_blocked = True   # 闸门拦截 → 强制 continue
                still_active.append(i)
                ep_st[i] = {
                    "sigma":            sigma_map[i],
                    "ctrl_tid":         tid,
                    "gate_blocked":     gate_blocked,
                    "critic_turns":     [],
                    "verifier_turns":   [],
                    "correction_turns": [],
                    "corrected_answer": None,
                    "pending":          [],
                }
            active = still_active
            if not ep_st:
                continue

            # ── 2. proposer 固定起点（batch） ──────────────────────────────
            prop_info = []
            for i in active:
                sys = PromptTemplates.proposer_system()
                usr = (f"问题：{questions[i]}\n当前状态："
                       f"{blackboards[i].to_text(include_flaws=self.flaw_in_primary)}")
                prompt = make_prompt(sys, usr)
                prop_info.append((i, next_tid(i), sys, usr, prompt))

            prop_res = self.vllm_engine.generate_batch(
                [{"role": "proposer", "prompt": prompt,
                  "temperature": self.temperature}
                 for _, _, _, _, prompt in prop_info]
            )

            for (i, tid, sys, usr, prompt), res in zip(prop_info, prop_res):
                out = record(i, "proposer", sys, usr, res, tid,
                             prompt_text=prompt, split_credit=self.token_credit)
                reasoning, answer = parse_reasoning(out)
                blackboards[i].add_message(Message(0, MessageType.TRACE, (reasoning, answer)))
                # 展示副本：交给别的角色看的文本一律剥掉 <interaction> 块。
                # `out` 本身（= record 存下的训练目标）绝不能动。
                shown = strip_interaction(out)

                st = ep_st[i]
                st["primary_tid"] = tid
                st["primary_answer"] = answer
                # 格式健康：是否输出了答案标签（而非靠抽末尾数字兜底）。
                # 格式崩了 reward 再高也是假的，所以进入监控指标。
                st["primary_parsed"] = has_answer_label(out) and bool(answer)
                # 同时分开记两个条件。`primary_parsed` 是**合取**，聚合成
                # `parse_rate` 之后就再也拆不开了——v3 那 150 步只能读出「两种失败
                # 合计 ≤ 24%」（parse 首步 0.95、均值 0.86、最低 0.76），而这两种
                # 失败的后果完全不同，混在一个数里等于什么都没测到：
                #   ① 无标签 → 走「取文中最后一个数字」兜底。实测 23 个真实无标签
                #      turn 里 6 个（26%）抽出的是垃圾数字（gold `\frac{7}{32}`
                #      抽成 `'32'`、`45,045` 抽成 `'045'`），这种票**可解析、与真票
                #      在票池里完全不可分**。
                #   ② 空答案 → 空串进票池。实测两票空串（加权各 0.5，合 1.0）压过
                #      一票被 verifier 背书的正确答案（0.9×1），weighted 与 uniform
                #      **都**被投成 `''`；而 `to_text` 的 `if a` 过滤又把它藏起来，
                #      黑板会说「已有3个解法」却只列 1 个答案。
                # 两条都要修（#22），但先得知道运行时各占多少——现有落盘一个字节都
                # 没有（`train.py` 只 dump `{"step": step}`，答案串从不写盘），所以
                # 这两个计数器是唯一的测量入口。本步**只加读数、不改任何判定**。
                st["no_label"] = not has_answer_label(out)
                st["empty_answer"] = not answer

                action, target, reason = parse_interaction(out)
                # 自指归一化：`parse_interaction` 已保证 target ∈ ROLE_NAMES 或
                # 整体退回 ("none","none","")，所以下面 hop 循环里
                # `target not in ROLE_NAMES` 实际不可达，**唯一可达的丢弃条件是
                # target == initiator**（proposer 自己写 `target: proposer`）。
                # 该情形下 hop 被 continue 掉、交互从未执行，但若仍按 u=True 记账：
                #   ① INTERACTION 落黑板 → 别的角色以为发生过一次交互
                #   ② int_rate 计入一次未发生的交互
                #   ③ r_int 按「发起了求助」定价 → 为没发生的事件付奖励
                # 三项污染 eff / sel / 漏斗，即用来判断"是否学会何时求助"的那些数。
                # 因此 u 的判定必须与 hop 的执行条件共用同一个谓词：自指等价于没发起，
                # 归一为 none 后自然落入下面的 ε 强制注入分支（与解析失败同等对待）。
                if action != "none" and target == "proposer":
                    self.n_self_target += 1
                    action, target, reason = "none", "none", ""
                u = action != "none" and self.max_hops > 0
                forced = False
                if u:
                    blackboards[i].add_message(Message(
                        0, MessageType.INTERACTION,
                        {"from": "proposer", "action": action,
                         "target": target, "reason": reason}))
                elif self.max_hops > 0:
                    # ── 强制注入（§2.4 冷启动保护 + §2.1 闸门解锁） ─────────
                    # v2.0 只注 critic，verifier 没有任何兜底调用通道：一旦
                    # int_rate→0，黑板永远拿不到 verifier 分数 → stop_gate 拦下
                    # 所有 stop → 每个 episode 跑满 max_rounds（实测连锁失效）。
                    # 因此：本轮 controller 想停但被闸门拦下时优先注 verifier（直接
                    # 解锁终止路径），否则按 ε 概率在 critic/verifier 间随机选。
                    if st["gate_blocked"]:
                        forced, action, target, reason = True, "request", "verifier", ""
                    elif self._rng.random() < eps_force:
                        tgt = "verifier" if self._rng.random() < self.eps_verifier_share \
                              else "critic"
                        forced, action, target, reason = True, "request", tgt, ""
                st["u"] = u
                st["forced"] = forced
                # target 对 forced 轮也落盘（v2.3：q_spont/q_forced 拆分需要）
                st["target"] = target if (u or forced) else None
                if u or forced:
                    st["pending"].append(("proposer", shown, action, target, reason, forced))

            # ── 3. 交互链（≤ max_hops 跳，跨 episode 按深度批量） ──────────
            for _hop in range(self.max_hops):
                batch_req = []
                for i, st in ep_st.items():
                    if not st["pending"]:
                        continue
                    initiator, init_out, action, target, reason, forced = st["pending"].pop(0)
                    # 防御性不变量（不再是正常路径）。三个 append 点都已保证
                    # target 合法且非自指：proposer 起点做了自指归一化、critic 标错
                    # 硬触发写死 "proposer"、响应方下一跳带 `t2 != target` 守卫。
                    # 因此这里 continue 掉任何一条都意味着上游回归，而不是模型行为。
                    if target == initiator or target not in ROLE_NAMES:
                        continue
                    sys, usr, reviewed = responder_prompt(
                        i, target, initiator, action, reason, init_out, forced)
                    prompt = make_prompt(sys, usr)
                    batch_req.append((i, next_tid(i), target, sys, usr, reviewed, prompt))
                if not batch_req:
                    break
                # 跳深计数（从 1 起，与"第几跳"的口头说法一致）。放在 `break` 之后、
                # `generate_batch` 之前：只统计**真的发出去**的请求，空批次不记。
                # 同时按响应方角色分开记，因为 `max_hops: 3` 想要的具体事件是
                # "第 3 跳到达 verifier"（修正后的答案在轮内被打分），只看总数分不出
                # 它到的是 verifier 还是又一次 critic。
                self.n_hop_depth[_hop + 1] += len(batch_req)
                for _, _, _t, _, _, _, _ in batch_req:
                    self.n_hop_depth[f"{_hop + 1}:{_t}"] += 1

                res_all = self.vllm_engine.generate_batch(
                    [{"role": t, "prompt": prompt,
                      "temperature": self.temperature}
                     for _, _, t, _, _, _, prompt in batch_req]
                )
                for (i, tid, target, sys, usr, reviewed, prompt), res in zip(
                        batch_req, res_all):
                    out = record(i, target, sys, usr, res, tid, prompt_text=prompt)
                    # 展示副本（不影响 record 存下的训练目标）：黑板文本会嵌进
                    # controller / critic / verifier 每一个 prompt，而 flaw 窗口只有
                    # `_MAX_FLAW_CHARS`；块占着窗口就是 v3 "critic 说了等于没说"
                    # 的根源。剥离对接收方零信息损失（发起意图由 request_context 表达）。
                    shown = strip_interaction(out)
                    st = ep_st[i]
                    bb = blackboards[i]
                    flagged = False

                    if target == "critic":
                        flagged = critic_found_errors(out)
                        if flagged:
                            bb.add_message(Message(1, MessageType.FLAW, {"content": shown}))
                        st["critic_turns"].append({
                            "tid": tid, "flagged": flagged,
                            "reviewed_answer": reviewed,
                            "correction_followed": False,
                        })
                    elif target == "verifier":
                        score = parse_score(out)
                        bb.add_message(Message(
                            2, MessageType.SCORE,
                            (reviewed, score if score is not None else 0.5)))
                        st["verifier_turns"].append({
                            "tid": tid, "score": score, "reviewed_answer": reviewed,
                        })
                    else:  # proposer 修正：新 trace 进黑板（上下文/σ/因果通路）
                        reasoning, answer = parse_reasoning(out)
                        bb.add_message(Message(0, MessageType.TRACE, (reasoning, answer)))
                        st["correction_turns"].append({"tid": tid, "answer": answer})
                        st["corrected_answer"] = answer
                        corr_answers[i].append(answer)
                        # 本轮内此前的 critic flag 得到了修正响应 → 因果窗口在本轮
                        for ct in st["critic_turns"]:
                            ct["correction_followed"] = True

                    # 响应方的下一跳（受 hop 预算约束）。
                    # v2.2：critic 标错 → 机械触发 proposer 修正，不再依赖 critic
                    # 自己学会输出 <interaction>。v2.1 实测 eff≈0：修正跳由“未学会
                    # 的行为”把守，correction_turns 几乎恒空 → q≈0 → 发起恒负期望
                    # → int_rate 塌。与 gate_blocked 强注 verifier 同一设计哲学：
                    # 关键因果通路由机制保证，角色只学“判断”，不学“走流程”。
                    a2, t2, r2 = parse_interaction(out)
                    if target == "critic" and flagged:
                        bb.add_message(Message(
                            list(ROLE_NAMES).index(target), MessageType.INTERACTION,
                            {"from": target, "action": "request",
                             "target": "proposer", "reason": r2}))
                        st["pending"].append((target, shown, "request", "proposer", r2, False))
                    elif a2 != "none" and t2 != target:
                        bb.add_message(Message(
                            list(ROLE_NAMES).index(target), MessageType.INTERACTION,
                            {"from": target, "action": a2, "target": t2, "reason": r2}))
                        st["pending"].append((target, shown, a2, t2, r2, False))

            # ── 3.5 闸门解锁批次（v3.1，不占 hop 预算） ────────────────
            # v2.0 写的解锁通道在上文 elif（proposer 未自发起）分支里，当时
            # int_rate→0 所以总能触发。v3 零成本冷启动把 int_rate 救到 0.78后，
            # 78% 的轮次走 if u: 分支，解锁反而被遮蔽；其中 75% 又选 critic
            # （tgtC=0.75）→ 黑板永远拿不到分数 → stop 被拦（gate 10→193）
            # → 跑满 max_rounds → prompt 膨胀至 step 151 崩。前一个修复的成功
            # 造成了后一个修复的失效，故改为与 proposer 选择无关的独立批次：
            # 想停但没钥匙，系统就发一把。
            # 该 turn 只领 verifier 自己的校准奖励（r_int 由主 turn 的 u/forced
            # 定，不受影响），因此不污染因果归因，且顺带抬高 verifier 覆盖率。
            if self.gate_unlock and self.max_hops > 0:
                unlock_info = []
                for i, st in ep_st.items():
                    if not st["gate_blocked"] or blackboards[i].scores:
                        continue
                    bb = blackboards[i]
                    last = bb.traces[-1] if bb.traces else ("", "")
                    sys = PromptTemplates.verifier_system()
                    usr = (f"待验证答案：{last[1]}\n推理：{last[0]}\n"
                           f"当前状态：{bb.to_text()}")
                    prompt = make_prompt(sys, usr)
                    unlock_info.append((i, next_tid(i), sys, usr, last[1], prompt))
                if unlock_info:
                    unlock_res = self.vllm_engine.generate_batch(
                        [{"role": "verifier", "prompt": prompt,
                          "temperature": self.temperature}
                         for _, _, _, _, _, prompt in unlock_info])
                    for (i, tid, sys, usr, reviewed, prompt), res in zip(
                            unlock_info, unlock_res):
                        out = record(i, "verifier", sys, usr, res, tid,
                                     prompt_text=prompt)
                        score = parse_score(out)
                        blackboards[i].add_message(Message(
                            2, MessageType.SCORE,
                            (reviewed, score if score is not None else 0.5)))
                        ep_st[i]["verifier_turns"].append({
                            "tid": tid, "score": score, "reviewed_answer": reviewed,
                        })
                    self.n_gate_unlocked += len(unlock_info)

            # ── 4. round record 落盘 ───────────────────────────────────────
            for i, st in ep_st.items():
                round_records[i].append({
                    "sigma":            st["sigma"],
                    "ctrl_tid":         st["ctrl_tid"],
                    "gate_blocked":     st["gate_blocked"],
                    "primary_tid":      st["primary_tid"],
                    "primary_answer":   st["primary_answer"],
                    "primary_parsed":   st["primary_parsed"],
                    "no_label":         st["no_label"],
                    "empty_answer":     st["empty_answer"],
                    "corrected_answer": st["corrected_answer"],
                    "u":                st["u"],
                    "forced":           st["forced"],
                    "target":           st["target"],
                    "critic_turns":     st["critic_turns"],
                    "verifier_turns":   st["verifier_turns"],
                    "correction_turns": st["correction_turns"],
                })

        # ── finalise ─────────────────────────────────────────────────────────
        results = []
        for i in range(n):
            # 生产路径：这三行是唯一决定交出去的答案的地方，下面的反事实臂**一行都
            # 不许碰它**。四臂是旁挂的观测；哪个臂的 exclude 串到这里来，就等于悄悄
            # 改了判定，而 acc 的变化会被误读成开关的效果。测试钉住了这条。
            exclude = None if self.correction_in_vote else corr_answers[i]
            final_answer = self._majority_vote(blackboards[i], exclude)
            is_correct = math_equal(final_answer, correct_answers[i])
            # 消融对照（§19.3 判据①）：同一批 episode 上多种计票各自的正确性，
            # 免二次 rollout 即可在线量化聚合方式的 Δacc。所有臂都**无条件**计算，
            # 使读数方向恒定而不随生产开关变化——v3.1 回退 uniform 后若只算生产那
            # 一路，d_vote 会恒等于 0，重开 weighted 的判据就此失明。
            #
            # 第十轮扩成 2×2：`correction_in_vote` 与 `vote_mode` 都是**黑板的后处理
            # 函数**，同一批 rollout 上四种组合全算得出，`_majority_vote` 是纯本地
            # 计数、零采样开销。于是：
            #   d_vote = wt_excl − uni_excl   （钉在排除臂，与 v3 基线同口径）
            #   d_corr = uni_incl − uni_excl  （钉在 uniform 臂，同理）
            # **不必真打开 `correction_in_vote` 就能知道该不该打开**——这正是上面
            # 那条"读数方向恒定"的原则第二次派上用场。
            _excl, _incl = corr_answers[i], None
            uni_excl = math_equal(self._majority_vote(
                blackboards[i], _excl, mode="uniform"), correct_answers[i])
            wt_excl = math_equal(self._majority_vote(
                blackboards[i], _excl, mode="weighted"), correct_answers[i])
            uni_incl = math_equal(self._majority_vote(
                blackboards[i], _incl, mode="uniform"), correct_answers[i])
            wt_incl = math_equal(self._majority_vote(
                blackboards[i], _incl, mode="weighted"), correct_answers[i])
            turn_data, round_meta = compute_turn_data(
                round_records[i], correct_answers[i], is_correct,
                self.max_rounds, self.cfg,
                stop_ctrl_tid=stop_ctrl_tids[i],
                stop_sigma=stop_sigmas[i] or "verify",
            )
            # 票池埋点：offline f 臂测出 k=4 等权投票值 +10 点，但那是 k 份独立
            # 采样；这里的票是同一条 episode 逐轮产生、后轮能看见前轮答案。若
            # controller 学会「答案稳了就停」，池子会塌成重复票，投票退化为重复
            # 确认第一个答案——此时 acc 高是 greedy 本身准，与聚合无关。n_distinct
            # 是唯一能分辨这两种情形的量，缺了它就无法判断加大 k 是不是杠杆。
            pool = (self._vote_pool(blackboards[i], exclude)
                    if blackboards[i].traces else Counter())
            n_votes = sum(pool.values())
            top2 = pool.most_common(2)
            results.append({
                "messages":        all_messages[i],
                "turn_ids":        turn_ids_list[i],
                "log_probs":       log_probs_list[i],
                "raca_turn_data":  turn_data,
                "raca_round_meta": round_meta,
                "final_answer":    final_answer,
                "is_correct":      is_correct,
                # 2×2 反事实臂（第十轮）。excl = 修正票不进池，incl = 进池。
                "is_correct_uni_excl": uni_excl,
                "is_correct_wt_excl":  wt_excl,
                "is_correct_uni_incl": uni_incl,
                "is_correct_wt_incl":  wt_incl,
                # 旧名保留，供 wandb 历史曲线连续。**刻意钉在「排除臂」上而不是跟着
                # 生产 `correction_in_vote` 走**：跟着走的话，开关一翻这两条曲线的
                # 含义就静默变了，而图上完全看不出来——那正是本仓库反复栽的
                # 「两把尺子」的种子。要读生产路径请用 `is_correct`。
                "is_correct_uniform":  uni_excl,
                "is_correct_weighted": wt_excl,
                "stopped":         stop_ctrl_tids[i] is not None,
                "n_votes":         n_votes,
                "n_distinct":      len(pool),
                # 首位领先幅度（占总票数）。=1.0 意味着全票一致，投票没做任何事。
                "vote_margin":     ((top2[0][1] - (top2[1][1] if len(top2) > 1 else 0))
                                    / n_votes) if n_votes else 0.0,
            })
        return results

    # ── 分片计数器聚合（v3.2 第十三轮） ─────────────────────────────────────
    def absorb_counters(self, others) -> None:
        """把分片 executor 的诊断计数器聚合到本实例。

        为什么需要它：`train.py` 在多 vLLM 引擎时走**分片 rollout**，每步为每个
        分片新建一个临时 `AgenticExecutor`，而 step 行读的是 `trainer.executor`
        —— 一个从未跑过 rollout 的实例。于是 `n_hop_depth` / `n_gate_unlocked` /
        `n_self_target` / `n_credit_split_failed` / `n_logprob_mismatch` /
        `n_prompt_clipped` 若不聚合，都会在集群上**结构性恒为 0**。
        08-28 两跑（`v32_m1` / `v32_open`，实测 7 个引擎）四个读数全程是假的 0：
        `gate=363→0` 里那个 0 没有意义；`hop=` 一次都没打印，于是 `max_hops` 2→3
        的**唯一**验收指标失效；`clip_prompt` 为 0 **不能**证明 prompt 没超预算，
        而那正是窗口 600 的第一号风险项。
        这与 #23 的 `round_meta` 白名单是同一类病：**指标照样打印、永远是 0。**

        **两类计数器语义不同，聚合方式必须不同——这是本方法最容易写错的地方：**
        - 批次级（`run_episodes_batch` 开头会重置）：`n_gate_unlocked`、
          `n_self_target`、`n_credit_split_failed`、`n_credit_split_failures`、
          `n_logprob_mismatch`、`n_hop_depth` → **赋值**为各分片之和。
        - 累计级（只在 `__init__` 归零、跨步累加）：`n_prompt_clipped` → **累加**。

        一律写 `+=` 会让批次级的数变成整轮累计（只增不减，被误读成"越来越糟"）；
        一律写赋值会让 `clip_prompt` 每步被覆盖，丢掉历史 —— 而"整轮有没有超过
        预算过"这个问题只有累计值答得了。
        """
        others = list(others)
        self.n_gate_unlocked = sum(getattr(o, "n_gate_unlocked", 0) for o in others)
        self.n_self_target = sum(getattr(o, "n_self_target", 0) for o in others)
        self.n_credit_split_failed = sum(
            getattr(o, "n_credit_split_failed", 0) for o in others)
        self.n_credit_split_failures.clear()
        for o in others:
            self.n_credit_split_failures.update(
                getattr(o, "n_credit_split_failures", None) or {})
        self.n_logprob_mismatch = sum(
            getattr(o, "n_logprob_mismatch", 0) for o in others)
        self.n_prompt_clipped += sum(
            getattr(o, "n_prompt_clipped", 0) for o in others)
        self.n_hop_depth.clear()
        for o in others:
            self.n_hop_depth.update(getattr(o, "n_hop_depth", None) or {})

    # ── 单条入口（batch 路径的薄包装，评测脚本兼容） ─────────────────────────
    def run_episode(self, question: str, correct_answer: str) -> dict:
        return self.run_episodes_batch([question], [correct_answer])[0]

    # ── evaluate.py 兼容 ─────────────────────────────────────────────────────
    def _critic_found_errors(self, critic_output: str) -> bool:
        return critic_found_errors(critic_output)

    def _vote_pool(self, blackboard: Blackboard, exclude_answers=None) -> Counter:
        """实际参与计票的票池。计票与埋点共用这一个真相源，避免两处逻辑漂移。

        v3（§19）：修正票退出投票池（修正正确率两次测量 ≤ 裸重采样，留在池内
        是在稀释投票质量）；全排空时回退全量计票。
        """
        counts = Counter(ans for _, ans in blackboard.traces)
        if exclude_answers:
            counts = counts - Counter(exclude_answers)
            if not counts:
                counts = Counter(ans for _, ans in blackboard.traces)
        return counts

    def _majority_vote(self, blackboard: Blackboard, exclude_answers=None,
                       mode=None) -> str:
        if not blackboard.traces:
            return ""
        counts = self._vote_pool(blackboard, exclude_answers)
        if (mode or self.vote_mode) != "weighted" or not blackboard.scores:
            return counts.most_common(1)[0][0]
        # 加权投票（v2.3 §18 通道①）：被 verifier 验证过的答案用其平均分数
        # 加权，未验证的用先验 0.5；票权 = 票数 × 权重。verifier 的校准奖励
        # 1−|v−p| 本就在训练它，加权投票把这个能力直接兑换成 acc。
        scored = {a for a, _ in blackboard.scores}
        weight = {
            ans: n * (blackboard.get_avg_score(ans) if ans in scored else 0.5)
            for ans, n in counts.items()
        }
        return max(weight, key=weight.get)
