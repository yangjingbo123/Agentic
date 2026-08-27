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


_INTER_BLOCK = re.compile(r"\s*<interaction>.*?</interaction>\s*", re.S)


def strip_interaction(text: str, *, trim: bool = True) -> str:
    """剥掉 `<interaction>` 块——**仅用于把一个角色的输出展示给另一个角色**。

    动机（v3.2 实测）：块的长度固定 68 字符，而 `blackboard.to_text()` 给 critic
    意见留的窗口只有 80 字符，`request_context` / `proposer_correction_user` 是
    300 字符。块占着窗口 → critic 真正报的错传到下游只剩十几个字。M1 把块移到
    尾部只是让截断先吃正文；要真正腾出信道，必须把块整个拿掉——它对**接收方**
    零信息量（发起意图已由 `request_context` 的 action/reason 显式表达）。

    **绝不能用在 `record()` 存下的 `response` 上**：那是训练目标，与 token_ids /
    log_probs 逐位对齐，改一个字符就会让 per-token ratio 错位。这里只处理展示副本。

    `trim=False` 供 `data/prepare_sft.py` 清洗 SFT 的 `user` 字段用：那是**已经
    套好模板**的整段 prompt，再 strip 一次会连模板自带的首尾空白一起改掉。两处
    共用同一个正则字面量是有意的——RL 侧的 prompt 现在是「模板 + 剥过块的输出」，
    SFT 的 user 必须落到逐字节相同的形状，否则又是一处静默的训练/推理漂移。
    """
    out = _INTER_BLOCK.sub("", text)
    return out.strip() if trim else out


# 无界文本上限（v3.1）。v3 实测：parse 0.95→0.80 后，兜底抽取的 answer 可能
# 是整段文本，reasoning 缺失时更是直接返回全部输出；两者都会进 responder
# prompt 与黑板文本，随轮数累积后撑破 max_model_len（step 151 实测 5036>4096）。
# 另：answer 是投票池的键，长文本 answer 彼此各不相同→各占一票稀释投票。
MAX_ANSWER_CHARS = 64      # 数学答案（含 LaTeX）远不到此长度；超出即视为解析失败
MAX_REASONING_CHARS = 1500  # 完整推理保留上限（够容纳正常多步解题）

# 标签的容错写法。抽成常量是为了让 reasoning 的**前瞻**与 answer 的**匹配**用
# 同一套字面量——两处写歧了会造成「reasoning 吃掉答案段」这种极难发现的错位。
_RSN_LABEL = r"推理过程[：:]"
_ANS_LABEL = r"(?:最终答案|Final Answer)[：:]"

# 块的开标签。**必须用它，不能用裸 `<`**（v3.2 第三轮修的实测缺陷）。
# reasoning 的前瞻原先写 `(?=最终答案[：:]|<|$)`，配 `re.S` + 非贪婪 `.+?`，于是
# 推理在**第一个 `<`** 处就停——而数学里 `<` 是不等号。实测 v2+v3 的 582 个
# proposer turn 中 32 个（5.5%）中招，中位丢掉 77% 的推理（411 字 → 83 字），
# 最坏 316 字只剩 10 字（`首先解第一段：若 x`）。这段残文正是
# `traces[-1][0]`，会原样进 critic 的 `待审查解法：` 与 verifier 的 `推理：`——
# 也就是说 critic 有 5.5% 的概率在**审查一个被截在第一个不等号处的片段**，还要
# 回答"有没有计算错误"。这与 flaw 窗口 80 是同一类病灶（静默的信道截断），且
# 自 v2（`524ddfa`）就在，与 M1 无关。
# 换成块的开标签后语义才是原本想表达的那个，且数学文本不可能包含它。
_INTER_OPEN = r"<interaction"


def parse_reasoning(text: str) -> tuple[str, str]:
    """proposer 输出 → (推理过程, 最终答案)。两路返回值均有硬上限。

    v3.2（M1）：`最终答案：` 的正则加了 `(?=<interaction|\\n|$)` 前瞻。M1 把
    `<interaction>` 块从开头移到了**答案之后**，若模型把块写在同一行
    （`最终答案：4<interaction>...`），原来的 `(.+)` 会把块吃进答案——长度常
    不足 64 字符，于是这个带标签的垃圾串会被当成合法答案进投票池，各自占一票。
    三个分支缺一不可：`\\n` 覆盖块换行写（最常见），`<interaction` 覆盖同行写，
    `$` 覆盖答案就是输出末尾。注意光写 `$` 不够：未开 re.M 时 `$` 只匹配串尾，
    非贪婪的 `.+?` 又跨不过换行，会导致整个匹配失败。前瞻用块的**开标签**而
    非裸 `<`，理由见 `_INTER_OPEN` 的注释（裸 `<` 会撞上不等号，实测截掉 5.5%
    的 proposer 推理）。

    v3.2（第二轮）：标签容错扩到 **半角冒号 + 英文别名**。这不是防御性编程，
    是修实测缺陷——SFT 数据里就有 `最终答案: 24`（半角）这种写法，原正则只认
    全角 `：`，于是整条答案落进「取文中最后一个数字」的兜底，`45,045` 会被抽成
    `045` 这类假答案进投票池占一票。**静默污染比解析失败更坏**，所以这里放宽。
    同理只给「答案」加英文别名、不给「推理过程」加：答案解析失败 → 垃圾投票
    （污染），而推理解析失败 → 整段输出当推理（仅冗长，无害），按同一条原则
    只在污染侧放宽。`critic_found_errors` 刻意不加英文别名，理由见该函数。
    """
    reasoning = re.search(rf"{_RSN_LABEL}(.+?)(?={_ANS_LABEL}|{_INTER_OPEN}|$)", text, re.S)
    answer = re.search(rf"{_ANS_LABEL}(.+?)(?={_INTER_OPEN}|\n|$)", text)
    # 兜底路径专用的无块副本。前瞻只在**标签存在**时挡住块；标签缺失时旧代码
    # 直接 `return text`，块就跟着整段输出被当成推理返回，而这个串正是
    # `traces[-1][0]`——会原样出现在 critic 的 `待审查解法：` 与 verifier 的
    # `推理：` 里。实测 SFT 数据 599 个 proposer turn 中 17 个（2.8%）如此；运行
    # 时只会更多，因为 v3 的 `parse_rate` 一度掉到 0.80，掉的正是标签。
    # 与前瞻是互补而非重复：前瞻管「有标签的正常输出」，这里管「无标签的兜底」，
    # 两条路的失效条件不同。先 `strip_interaction` 剥规范块，再按开标签截一刀，
    # 是为了连**没写闭标签**的残块一起挡掉（那种 `_INTER_BLOCK` 匹配不上）。
    fallback_src = re.split(_INTER_OPEN, strip_interaction(text), maxsplit=1)[0]
    if not answer:
        nums = re.findall(r"-?\d+\.?\d*", fallback_src)
        ans_str = nums[-1] if nums else ""
    else:
        ans_str = answer.group(1).strip()
        # 「最终答案：」后又接着展开论述 → 不是答案，当解析失败处理，
        # 而非截断后当真（截断会造出一个“看上去像答案”的假票污染投票池）。
        if len(ans_str) > MAX_ANSWER_CHARS:
            nums = re.findall(r"-?\d+\.?\d*", ans_str)
            ans_str = nums[-1] if nums else ""
    reasoning_str = reasoning.group(1).strip() if reasoning else fallback_src
    return (reasoning_str[:MAX_REASONING_CHARS], ans_str)


def has_answer_label(text: str) -> bool:
    """输出里是否真的写了答案标签（而非靠「取末尾数字」兜底）。

    存在的意义是让 `primary_parsed` 这个格式健康指标与 `parse_reasoning` **共用
    同一套标签字面量**。此前 executor 里硬编码 `"最终答案：" in out`，只认全角，
    放宽解析后就会出现「解析成功但指标报格式崩」的自相矛盾读数——而这个指标正是
    用来判断"reward 高是不是假的"，它自己不准就没有意义。
    """
    return bool(re.search(_ANS_LABEL, text))


def parse_score(text: str) -> float | None:
    """verifier 输出 → [0,1] 分数；解析失败返回 None。

    v3.2：标签容错扩到 `分数：`（全角）与 `Score:`（英文）。原正则只认半角
    `分数:`，而实测 764 条 verifier SFT turn 里有 15 条整条是英文（`Score: 1.0`），
    解析失败后 executor 会回落 **0.5 先验**——这不是中性默认，而是把「没验证过」
    和「验证了但拿不到分」混成同一个值，在 weighted vote 下系统性压制该答案。
    失败模式是静默污染，故放宽。
    """
    m = re.search(r"(?:分数|Score)[：:]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text)
    if not m:
        return None
    return max(0.0, min(1.0, float(m.group(1))))


def critic_found_errors(critic_output: str) -> bool:
    """鲁棒判断 critic 是否报了错（沿用 v1.x Fix 3 的解析逻辑）。

    **刻意不加英文别名**（与 `parse_reasoning` / `parse_score` 不同）：这里的
    判据是一对**互补**的启发式——先看有没有「无错误」，再看「错误分析」段是否
    非空。只给后者加英文别名而前者认不出 "No errors found"，会让每一条英文
    critic 输出都判成 flag（假阳性），直接污染 `r^critic` 的四格矩阵与 flag→corr
    漏斗。失败模式在这里是「保守不 flag」（良性），所以宁可失败也不放宽。
    整条英文的 critic turn 改为在 SFT 派生步剔除（`data/prepare_sft.py`）。
    """
    if "无错误" in critic_output or "无错" in critic_output:
        return False
    err_match = re.search(r"错误分析[：:]\s*(.+?)(?=<|$)", critic_output, re.S)
    if err_match:
        err_text = err_match.group(1).strip()
        return bool(err_text) and "无错误" not in err_text and "无错" not in err_text
    # 无「错误分析」段（响应未按格式）——保守不判 flag
    return False
