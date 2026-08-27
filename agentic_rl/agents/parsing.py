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
# 数学答案（含 LaTeX）的长度上限；超出即视为**这一行不是答案**。
#
# v3.2（第六轮）：64 → 192。64 是拍的，实测它偏紧到会毁掉合法答案：
# ① 模型侧：v2+v3 SFT 的 580 个 proposer turn 里命中 1 次，那一次是
#    `\begin{pmatrix} 2 & 0 & 7 \\ 3 & 5 & -1 \\ -8 & -2 & 4 \end{pmatrix}`
#    （68 字，完全正确），被兜底抽成 `'4'` —— 一个正确答案被记成错，且进投票池
#    占一票。② gold 侧：`math_train_rl` 5265 题有 14 题 gold 超 64（最长 159，
#    形如 `\left( -\infty, -\frac{1}{2} \right] \cup \left[ \frac{1}{2}, \infty \right)`），
#    `math_test` 3669 题有 6 题（最长 81）。也就是说这条上限本身就在和数据集矛盾。
# 192 覆盖全部已观测 gold（159）并留余量；答案长度 p99 = 46，放宽不会改动
# 99% 的样本。
#
# 放宽是安全的，因为 64 从来不是它看上去在防的那道防线：
# ① 「同行写 `最终答案：4<interaction>...`」由 `_INTER_OPEN` 前瞻挡（见
#    `parse_reasoning`），而且正则的 `\n` 前瞻已把答案限死在一行内，靠长度去
#    拦这种垃圾串本来就拦不住——原注释自己写着那种串"长度常不足 64 字符"。
# ② 「答案后接着展开论述」是它唯一真正针对的情形，实测 580 个 turn 里 0 例。
#
# 注意：这个数与 `Blackboard._MAX_ANSWER_CHARS`（展示上限，仍是 64）**职责不同，
# 不要互相看齐**。这里管「什么算答案」（投票与判分的输入），那里管 prompt 预算
# （答案列表 × 轮数是乘数项）。但两者的**大小关系**有后果：本轮放宽之前两个数
# 恰好都是 64，黑板那两处 `a[:64]` 因此永远切不到东西；放宽后它们才真正开始
# 截断，所以同一轮把那两处改成了带标记的 `clip_text`。
MAX_ANSWER_CHARS = 192
MAX_REASONING_CHARS = 1500  # 完整推理保留上限（够容纳正常多步解题）

# ── 信道截断的统一出口（v3.2 第四轮） ────────────────────────────────────
# 本仓库反复栽在同一类病灶上（到这一轮已第五例：flaw 窗口 80、块未剥、reasoning
# 前瞻裸 `<`、无标签兜底泄漏、`critic_found_errors` 裸 `<`）。复盘后的结论是：
# **坏的不是「截断」，是「截断不可见」。** 截断本身必要——prompt 预算有限，
# `_MAX_FLAW_CHARS` / `MAX_REASONING_CHARS` 都得存在。真正致命的是接收方拿到残
# 文之后**无法分辨「对方说完了」和「对方还有话被砍了」**，于是照常给出一个自信
# 的判断：critic 对着截在 `C\sqrt{` 处的 LaTeX 回答"有没有计算错误"，proposer
# 对着半句话去"修正"。错误由此凭空产生，且全链路无人报错。
#
# 所以所有面向**另一个角色**的截断都必须走这里，带一个可见标记。这不是防御性
# 编程：标记改变的是接收方的可判定性——看到标记它就知道该说"信息不全"而不是
# "这步是错的"。
#
# 刻意不加计数器：`to_text` 每个 turn 被调用多次（controller / critic / verifier
# / 修正各一次），在里面计数得到的是调用次数而非事件数，是个会误导人的读数。
# 要量截断发生率就 grep 落盘 prompt 里的这个标记——比计数器更好，因为它还能告诉
# 你是哪个 turn。`agentic_executor.clip_user` 的 `n_prompt_clipped` 是另一回事：
# 那里每个 prompt 恰好过一次，计数是准的。
CLIP_MARK = "…（后文已截断）"

# 展示给另一个角色的自由文本窗口。三处共用同一个数（黑板 `发现问题`、
# `request_context` 的 `对方内容`、`proposer_correction_user` 的引述），只留一个
# 量级要记。300 的取值理由见 `envs/blackboard.py:_MAX_FLAW_CHARS`。
MAX_CHANNEL_CHARS = 300


def clip_text(text: str, limit: int) -> str:
    """按字符截断并**留下可见标记**。所有跨角色的文本截断都必须走这里。

    标记不占 `limit` 预算（`text[:limit] + 标记`），因此「保证送达 limit 个字符
    正文」这条语义不因加标记而缩水；多出的十来个字符对 3008 token 的 prompt
    预算可以忽略。
    """
    return text if len(text) <= limit else text[:limit] + CLIP_MARK


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
        # 超过上限 → 这一行不像答案（多半是「最终答案：」后又展开论述），退回
        # 「取其中最后一个数字」。注意这**不是**解析失败处理：早先的注释写着
        # "当解析失败处理，而非截断后当真"，但代码做的恰恰是后者的变体——抽出的
        # 数字仍会当成一张正常选票。留着这个行为是权衡后的结果，不是疏忽：
        # 改成返回 `""` 看着更诚实，实际更坏——空串是 `_vote_pool` 的合法键，
        # 加权投票里未被 verifier 验证的候选拿 0.5 先验，于是「两次解析失败」
        # 权重 2×0.5=1.0 会压过「一次被验证的正确答案」0.9×1=0.9，直接把
        # 空串投成最终答案。要先给票池加「不可解析候选不计票」才谈得上改这里。
        if len(ans_str) > MAX_ANSWER_CHARS:
            nums = re.findall(r"-?\d+\.?\d*", ans_str)
            ans_str = nums[-1] if nums else ""
    reasoning_str = reasoning.group(1).strip() if reasoning else fallback_src
    # 推理超限**带标记**截断。实测 599 个 proposer turn 中 8 个（1.3%）超 1500，
    # 最长 1753 字；旧代码在第 1500 个字符处硬切，实测切口落在 LaTeX 中间
    # （`C\sqrt{`）。这个串正是 `traces[-1][0]` → critic 的 `待审查解法：`，于是
    # critic 要对一段被截在半个公式处的推导回答"有没有计算错误"——它没有任何
    # 线索知道后面还有 253 字。标记就是那个线索。
    return (clip_text(reasoning_str, MAX_REASONING_CHARS), ans_str)


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

    v3.2（第四轮）：前瞻由裸 `<` 改为 `_INTER_OPEN`。这是裸 `<` 病灶的第五例，
    写法上和 `parse_reasoning` 那条一模一样（当时只扫了 `parse_reasoning`，漏了
    这里），但**行为后果与那条完全不同，必须说清，否则会误导下一个人**：

    这里改前改后**判定完全等价**，不是"实测没翻转"而是**结构上不可能翻转**。
    先说证据：长度 ≤4 的字母表穷举 11110 个串、真实 critic turn 2228 条，两版
    判定不同的**一个都没有**。再说机制，两条缺一不可：① `.+?` 至少吃 1 个字符，
    而 `\\s*` 已贪婪吃掉前导空白，所以 `err_text` 恒非空——`bool(err_text)` 这个
    分支永远为真，前瞻停在哪都不影响它；② 输出里任何位置的「无错误」都被**上面
    第一个分支**在整段上拦掉了，轮不到截断后的 `err_text` 去判。
    换言之被裸 `<` 截短的只是 `err_text` 的**长度**，而这个函数只看它空不空。

    （这条更正是**变异测试逼出来的**：原先这里写着"有一条可达的假阴性路径——
    critic 用尖括号做强调时 err_text 被截成空串"。把修复回滚成裸 `<` 之后，
    专门为这条路径写的
    `test_critic_flag_survives_angle_bracket_emphasis` 照样通过——一个抓不到
    自己那个变异的测试，说明被测的那个失败模式根本不存在。先写测试再回滚验证，
    比"想一想觉得有道理"可靠得多。）

    所以本次改动的理由**只有一条**：留着它，整个仓库就还有一个裸 `<`，
    `test_no_bare_lt_lookahead_in_parsers` 那条不变量就立不起来。这一类已经
    犯了五次，靠记性显然不行，得靠一个会失败的测试；而那个测试要能成立，
    前提是仓库里一个裸 `<` 都不剩。为一条不变量做的一致性改动是正当的，
    但它**不是 bug 修复**，不该记进"修了几个缺陷"里。
    """
    if "无错误" in critic_output or "无错" in critic_output:
        return False
    err_match = re.search(rf"错误分析[：:]\s*(.+?)(?={_INTER_OPEN}|$)", critic_output, re.S)
    if err_match:
        err_text = err_match.group(1).strip()
        return bool(err_text) and "无错误" not in err_text and "无错" not in err_text
    # 无「错误分析」段（响应未按格式）——保守不判 flag
    return False

