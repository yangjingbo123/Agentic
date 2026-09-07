"""SFT 起跑前的 tokenizer 级预检（只加载 tokenizer，不碰 GPU，秒级）。

存在的理由：这两个读数**只能在有 tokenizer 的机器上取到**，本地开发机拿不到，
而它们决定了一次 3 epoch 的 SFT 到底训了什么。

── A. 模板对齐：`enable_thinking` 的那处不对称 ──────────────────────────
AST 普查（遍历全仓 .py，按 `Call.func.attr == "apply_chat_template"` 计，
**排除 test_*.py**）：**19 处调用，9 处显式传 `enable_thinking=False`，10 处没传。**

范围一定要写清，否则这行数字会被反复误判成过期：**含**测试文件是 21 / 9 / 12
——测试里给假 tokenizer 的两次调用也会被 `ast.walk` 数进来，那两处与本议题无关。
复现（08-28 实跑过，输出 `19 9 10`；把 `and not f.startswith('test_')` 去掉即得
含测试口径的 `21 9 12`）：

    python3 -c "
    import ast,os
    c=[x for r,_,fs in os.walk('.') if '.git' not in r and '__pycache__' not in r
       for f in fs if f.endswith('.py') and not f.startswith('test_')
       for x in ast.walk(ast.parse(open(os.path.join(r,f)).read()))
       if isinstance(x,ast.Call) and getattr(x.func,'attr','')=='apply_chat_template']
    w=[x for x in c if any(k.arg=='enable_thinking' for k in x.keywords)]
    print(len(c),len(w),len(c)-len(w))"

（这条命令**必须自己跑过再写进来**。此前这里写的那版最后一行以 `;` 开头接在列表
推导之后，Python 直接 SyntaxError——一条跑不起来的"复现命令"比没有更坏，它让读者
以为这个数字有人核对过。）

（此前这里写的是"6 处 / 5 处"，那是人眼扫出来的，漏了一半；行号也全偏移了。
本次新增第 4 种渲染 `sft_full_ef` 后 18→19、8→9。）

9 处传参中，真正跑在推理/数据链路上的是这 5 处：

    agents/agentic_executor.py:106   rollout 取样（vLLM prompt）
    training/grpo_trainer.py:237     RL 前向（算 new_lps）
    verify_sft_format.py:84          SFT 格式验收
    measure_channels.py:96           离线测量
    generate_sft_v3.py:111           **生成 v3 SFT 数据本身**

另 4 处是量具自身，不影响训练：本脚本的 `rl_prompt` 与 `sft_full_ef` 两臂，以及
`diag_fix.py:24` 与 `:50`（一次性诊断脚本）。

10 处没传的分布同样要说清，否则容易误以为"只有 train_sft.py 没传"：
    本脚本的 `sft_full` / `sft_prompt` 两臂 + `census_truncation` 里的两处
                                           刻意镜像 SFT 侧，不传是对的
    diag_fix.py:23                         该脚本的对照臂（与 :24 成对）
    diagnose_loss.py:54 / diagnose_loss2.py:50 / diagnose_nan.py:46
                                           三个已废弃的诊断脚本，不在任何链路上
    train_sft.py:37 / :38                  **唯一一处活着的缺口——真正训练的那处**

（本脚本自身的位置一律按变量名指代，不写行号：自指的行号每改一次本文件就漂一次，
而上面那五个外部行号有 `assert_mirrors_train_sft` 与测试里的 AST 不变量兜着。）

`train_sft.py:37-38` 两行都没传，取 Jinja 默认（Qwen3 模板里 `enable_thinking`
未定义 ⇒ 不注入空 think 块）。

机制：Qwen3 的模板对**最后一条 assistant 消息**有专门分支（`loop.last` ⇒ 渲染成
`<think>\n{reasoning}\n</think>\n\n{content}`），而 `add_generation_prompt=True`
那一侧只有在 `enable_thinking is False` 时才注入空块。于是 `train_sft.py` 的
    full_text   = 模板(sys,user,assistant)  ← 含空 think 块（loop.last 分支）
    prompt_text = 模板(sys,user, gen=True)  ← **不**含（没传 enable_thinking）
两者在 assistant 头之后分叉，`labels` 的监督区从空 think 块开始。

**2026-08-28 上机实测：verdict = think_block_in_supervised_region，且
`prefix_ok = True`，`delta_tokens = 4`。**（此前这里写的"短 5 个 token"是猜的。）

判据取最干净的那个不变量：推理时喂给模型的是 `rl_prompt`，那么 SFT 见过的整条
序列就必须以 `rl_prompt` 起头。`rl_prompt` 不是 `sft_full` 的前缀 ⇒ 两条路分叉。

`prefix_ok = True` 这个读数**否掉了**此前写在这里的一句错话（"SFT 教模型先吐一个
空 think 块再答"）。理由：前缀成立意味着那 4 个 token 在 SFT 序列里出现的位置与
推理时**逐 token 相同**；推理时模型被要求续写的起点是块**之后**的 `推理过程：`，
而那个位置在 SFT 里被直接监督过。模型在块所在的那个位置上根本没有发挥机会，
所以不会"再吐一个块"。

真实代价只有两项，都不改变分布：
    ① 错位监督 4 个 token：SFT 在 `assistant\n` 这个位置上教了那个块，而推理时
       这 4 个 token 属于 prompt、不需要模型产出；
    ② loss 稀释：`train_sft.py` 对监督区取平均，这 4 个 token 掺进分母，等于把
       学习率悄悄乘了一个略小于 1 的系数（整体 <3.6%，监督区最短的 controller
       上 <8.7%；`clip_grad_norm_` 还会部分抵消）。
`prefix_ok` 把损害封在 SFT 内部——RL 的 rollout prompt 仍在分布内。

第 4 种渲染（`sft_full_ef`，同 `:37` 但传 `enable_thinking=False`）是给 #25 选
修法用的：`:38` 补上参数肯定对齐，但**两行都补**有可能让 `loop.last` 也不再注入
空块，那 `rl_prompt`（含块）就不再是 `sft_full`（不含块）的前缀，`prefix_ok` 会
从 True 翻成 False——把现在这个无害的错位换成一次真的走散。本地无 tokenizer
量不出来，所以脚本同时打 `only_38_ok` 与 `both_flag_ok`，让 #25 照读数改。

本脚本**只报数、不拦**（exit 0）。原因是：这处不对称是既有的，当前
KL 参考快照 `sft_v3` 与本次 SFT 用的是同一个 `train_sft.py`，**两边一样偏**，
所以它不影响"M1 + 重放"这次的归因；而现在改掉它等于同一轮动两件事，反而毁掉
对照。先拿读数，修法单独一步。

── B. 截断普查：把 chars/2.2 的估计换成真 token 数 ────────────────────
`submit_primus_sft.sh` 里的体检曾用 `(len(system)+len(user)+len(response))/2.2`
估 token，这是个代理指标。它偏乐观：1024 下代理给 1.54%，真值 5.73%，**偏了
3.7 倍**——于是当阈值是 2% 时，闸门会在实际越阈的情况下放行。该处已在
`96c7384` 换成真 tokenizer，本脚本与它同口径。

并且要把"超限"拆成两种，因为后果完全不同（`train_sft.py:42` + `:96`）：
    ① response 被砍尾    → 仍有梯度，只是丢尾巴（M1 之后丢的是 <interaction> 块）
    ② prompt 本身就超限  → `labels` 全 -100 → `response_mask.sum()==0` → **整条
       turn 一点梯度都不产生**，等于白算一遍前向

**2026-08-28 真 token 实测（v23 全量 2864 turn，中位 426、最长 1797）：**
    max_len=1024  超限 164（5.73%），其中②零梯度 32（1.12%），①砍尾 132（4.61%）
    max_len=1536  超限   8（0.28%），其中②零梯度  0，        ①砍尾   8（0.28%）
此前这里写的"44 条超限、其中 8 条属②"是代理尺子的读数，**已作废**；同理
"1024 截断率 1.5%、抬到 1536 只剩 3 条"也作废。现用的 `MAX_LEN = 1536`
即由上表选定（②必须为 0，①留 2% 阈）。

用法：
    python check_sft_template.py --model_path <Qwen3-8B 路径>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# train_sft.py 里那两行的调用形状。本脚本必须逐字复刻它们，否则量的是别的东西。
# 复刻靠人眼是会漂的，所以下面 `assert_mirrors_train_sft` 直接读 train_sft.py 源码
# 核对；train_sft.py 一改而这里没跟上，预检会当场说自己过期，而不是给出一个
# 看起来正常的假读数。
#
# 走 AST 取关键字集合、而不是 grep 字面串：第一版就是 grep 字面串，结果有个真缺陷
# —— 等哪天真把 `enable_thinking=False` 补进 train_sft.py（也就是病灶被修好），
# 字面串立刻不匹配，预检会把它当成"自己过期"而 exit 2，于是**修 bug 反倒把作业
# 弄挂**。必须能分开两件事：调用形状漂了（该拦），和病灶被修了（该退休）。
_EXPECT = {
    "full_text":   (["messages"],       {"tokenize": "False",
                                         "add_generation_prompt": "False"}),
    "prompt_text": (["messages[:-1]"],  {"tokenize": "False",
                                         "add_generation_prompt": "True"}),
}


def _extract_calls(src):
    """从 train_sft.py 源码里取出 full_text / prompt_text 两处调用的实参形状。"""
    import ast
    found = {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Name) and tgt.id in _EXPECT):
            continue
        if not (isinstance(node.value, ast.Call)
                and getattr(node.value.func, "attr", "") == "apply_chat_template"):
            continue
        found[tgt.id] = ([ast.unparse(a) for a in node.value.args],
                         {k.arg: ast.unparse(k.value) for k in node.value.keywords})
    return found


def assert_mirrors_train_sft(path="train_sft.py"):
    """确认本脚本复刻的两处调用与 train_sft.py 现状一致。

    返回 True 表示 A 段的病灶前提仍然成立（train_sft.py 仍未传 enable_thinking）。
    """
    src = open(path, encoding="utf-8").read()
    found = _extract_calls(src)
    missing = [k for k in _EXPECT if k not in found]
    if missing:
        print(f"!! 本预检已过期：train_sft.py 里找不到 {missing} 的 "
              f"apply_chat_template 赋值，读数不可信，请先同步本脚本", file=sys.stderr)
        sys.exit(2)

    premise = True
    for name, (want_args, want_kw) in _EXPECT.items():
        args, kw = found[name]
        if args != want_args:
            print(f"!! 本预检已过期：{name} 的位置参数是 {args}，本脚本复刻的是 "
                  f"{want_args}", file=sys.stderr)
            sys.exit(2)
        for k, v in want_kw.items():
            if kw.get(k) != v:
                print(f"!! 本预检已过期：{name} 的 {k}={kw.get(k)}，本脚本复刻的是 "
                      f"{k}={v}", file=sys.stderr)
                sys.exit(2)
        # 多传一个关键字也算漂：`tools=` / `add_special_tokens=` 之类足以改变渲染，
        # 而本脚本复刻的是没有它们的那个版本。`enable_thinking` 是唯一的例外，
        # 它走下面"病灶被修了"的分支。
        extra = set(kw) - set(want_kw) - {"enable_thinking"}
        if extra:
            print(f"!! 本预检已过期：{name} 多传了 {sorted(extra)}，渲染结果可能"
                  f"与本脚本复刻的不同", file=sys.stderr)
            sys.exit(2)
        if "enable_thinking" in kw:
            premise = False
    if not premise:
        print("== 注意：train_sft.py 已经显式传 enable_thinking，A 段的病灶前提"
              "不再成立。读数仍照打（用来确认修对了），但这一段可以退休了 ==")
    return premise


def _common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def probe_template(tok, turn):
    """A 段：四种渲染逐字节对照。

    前三种量的是**现状**（`train_sft.py` 两行都没传 `enable_thinking`）；
    第四种 `sft_full_ef` 量的是**修法**——它与 `:37` 唯一的差别就是补上
    `enable_thinking=False`，用来回答 #25 的那个岔路：只补 `:38`（prompt 侧）
    还是两行都补。两者未必都保住 `prefix_ok`，见模块 docstring A 段。
    """
    messages = [
        {"role": "system",    "content": turn["system"]},
        {"role": "user",      "content": turn["user"]},
        {"role": "assistant", "content": turn["response"]},
    ]
    # 逐字复刻 train_sft.py:37-38
    sft_full = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=False)
    sft_prompt = tok.apply_chat_template(messages[:-1], tokenize=False,
                                         add_generation_prompt=True)
    # 逐字复刻 agents/agentic_executor.py:106-110（推理侧真实形状）
    rl_prompt = tok.apply_chat_template(messages[:-1], tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
    # 只为 #25 选修法：同 :37，但补上 enable_thinking=False。**不参与判定**
    sft_full_ef = tok.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=False,
                                          enable_thinking=False)

    print("=" * 72)
    print("A. 模板对齐（train_sft.py 与推理侧）")
    print("=" * 72)
    print(f"  SFT   prompt 尾 40 字：{sft_prompt[-40:]!r}")
    print(f"  推理  prompt 尾 40 字：{rl_prompt[-40:]!r}")
    same_prompt = sft_prompt == rl_prompt
    print(f"  两者逐字节相同：{same_prompt}")
    if not same_prompt:
        n = _common_prefix_len(sft_prompt, rl_prompt)
        print(f"    分叉于第 {n} 字符；SFT 侧多出 {sft_prompt[n:]!r}，"
              f"推理侧多出 {rl_prompt[n:]!r}")

    full_ids = tok.encode(sft_full, add_special_tokens=False)
    sft_pids = tok.encode(sft_prompt, add_special_tokens=False)
    rl_pids = tok.encode(rl_prompt, add_special_tokens=False)
    # 这就是 SFT 真正教模型输出的东西（train_sft.py:42 的监督区）
    supervised = tok.decode(full_ids[len(sft_pids):])
    print(f"\n  SFT 监督区（labels != -100 那段）头 60 字：{supervised[:60]!r}")
    print(f"  该 turn 的 response 头 60 字：            {turn['response'][:60]!r}")

    # 核心不变量：推理时模型被喂 rl_prompt，SFT 见过的整条序列就该以它起头
    prefix_ok = full_ids[:len(rl_pids)] == rl_pids
    delta = len(rl_pids) - len(sft_pids)
    print(f"\n  token 数：sft_full={len(full_ids)} sft_prompt={len(sft_pids)} "
          f"rl_prompt={len(rl_pids)}（rl − sft = {delta:+d}）")
    print(f"  不变量「rl_prompt 是 sft_full 的前缀」：{prefix_ok}")
    if prefix_ok and same_prompt:
        verdict = "aligned"
        print("  → 对齐。A 段这处不对称在本 tokenizer 上无后果。")
    elif prefix_ok:
        verdict = "think_block_in_supervised_region"
        print("  → 分叉在 prompt 侧，但 rl_prompt 仍是 sft_full 的前缀：空 think 块"
              "\n     落在监督区里。前缀成立意味着这几个 token 在 SFT 序列中的位置与"
              "\n     推理时逐 token 相同——推理时生成从块**之后**的「推理过程：」起，"
              "\n     而那个位置在 SFT 里被直接监督过，所以模型不会「再吐一个块」。"
              f"\n     代价只有两项：错位监督 {delta} 个 token（它们在推理时属于"
              "\n     prompt、不需要模型产出），以及 loss 取平均时被这几个 token 稀释"
              "\n     （相当于学习率乘了个略小于 1 的系数）。prefix_ok 把损害封在 SFT"
              "\n     内部，RL 的 rollout prompt 仍在分布内。")
    else:
        verdict = "diverged"
        print("  → **不对齐**：推理喂进去的前缀 SFT 从未见过。这一条比上一条更重，"
              "\n     说明两条路在 assistant 头之后就走散了。")

    # ── 第 4 种渲染：给 #25 选修法（只报数，不参与上面的 verdict）────────
    full_ef_ids = tok.encode(sft_full_ef, add_special_tokens=False)
    full_flag_matters = sft_full_ef != sft_full
    only_38_ok = prefix_ok                                   # 只补 :38 的效果
    both_flag_ok = full_ef_ids[:len(rl_pids)] == rl_pids      # 两行都补的效果
    print("\n  #25 选修法（第 4 种渲染 sft_full_ef = :37 + enable_thinking=False）：")
    print(f"    :37 传不传这个参数会改变渲染吗：{full_flag_matters}"
          f"（token 数 {len(full_ids)} → {len(full_ef_ids)}）")
    print(f"    只补 :38   ⇒ rl_prompt 仍是 sft_full 的前缀：{only_38_ok}")
    print(f"    两行都补   ⇒ rl_prompt 仍是 sft_full 的前缀：{both_flag_ok}")
    if not full_flag_matters:
        print("    → :37 加不加都一样，#25 只补 :38 即可（加了也无害）。")
    elif both_flag_ok:
        print("    → 两种补法都保住前缀，#25 可自由选（建议两行都补，形状一致）。")
    else:
        print("    → **只能补 :38**：两行都补会让 sft_full 不再含块，前缀不变量翻成"
              "\n       False，等于把现在这个无害的错位换成一次真的走散。")
    sup_ef = tok.decode(full_ef_ids[len(rl_pids):]) if both_flag_ok else None
    if sup_ef is not None:
        print(f"    两行都补后的监督区头 60 字：{sup_ef[:60]!r}")
        print(f"    该 turn 的 response 头 60 字：{turn['response'][:60]!r}")

    return {"verdict": verdict, "same_prompt": same_prompt,
            "prefix_ok": prefix_ok, "supervised": supervised,
            "sft_prompt": sft_prompt, "rl_prompt": rl_prompt,
            "delta_tokens": delta,
            "full_flag_matters": full_flag_matters,
            "only_38_ok": only_38_ok, "both_flag_ok": both_flag_ok,
            "supervised_ef": sup_ef}


def census_truncation(tok, path, max_lens=(1024, 1536, 2048)):
    """B 段：真 token 数的截断普查 + 零梯度拆分 + 监督区长度。

    **口径必须与 `submit_primus_sft.sh` 的闸门逐字一致**（该处 heredoc 里的
    `over` / `dead` / `byrole` 三个定义）：
        over = [r for r in rows if r[2] > max_len]              # 超限（砍尾或更坏）
        dead = [r for r in over if r[1] >= min(r[2], max_len)]  # 零梯度
    闸门那边 `assert not dead` + `assert trunc < 0.02`，本脚本只报数不拦。谁改一边
    都要改另一边，否则又是一次"两把尺子"——量具与闸门对同一件事给不同的数。
    """
    print("\n" + "=" * 72)
    print(f"B. 截断普查（真 token 数，数据 {path}）")
    print("=" * 72)
    rows = []          # (role, len(prompt_ids), len(full_ids))
    for line in open(path, encoding="utf-8"):
        for t in json.loads(line).get("turns", []):
            messages = [
                {"role": "system",    "content": t["system"]},
                {"role": "user",      "content": t["user"]},
                {"role": "assistant", "content": t["response"]},
            ]
            full = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=False)
            prompt = tok.apply_chat_template(messages[:-1], tokenize=False,
                                             add_generation_prompt=True)
            rows.append((t["role_name"],
                         len(tok.encode(prompt, add_special_tokens=False)),
                         len(tok.encode(full, add_special_tokens=False))))
    n = len(rows)
    chars_ruler = None

    def byrole(rs):
        d = {}
        for r in rs:
            d[r[0]] = d.get(r[0], 0) + 1
        return d

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2] if s else 0

    print(f"  {n} 个 turn，full token 数：中位 "
          f"{median([r[2] for r in rows])}，最长 {max(r[2] for r in rows)}")

    # 监督区长度 = full − prompt，也就是 train_sft.py:42 里 labels != -100 那段。
    # 它是 A 段那 delta 个错位 token 的**分母**：稀释比例 = delta / 监督区长度。
    # 按角色打是因为四个角色差得远——controller 只输出一行 decision，监督区最短，
    # 同样几个 token 在它身上占比最大，所以整体上界要由它决定，不能只看整体中位。
    sup_all = median([r[2] - r[1] for r in rows])
    print(f"  监督区 token 数（full − prompt，即 labels != -100 那段）：中位 {sup_all}")
    for role in sorted({r[0] for r in rows}):
        rs = [r for r in rows if r[0] == role]
        sup = [r[2] - r[1] for r in rs]
        print(f"    {role:11s} n={len(rs):5d}  中位 {median(sup):5d}  "
              f"最短 {min(sup):5d}  最长 {max(sup):5d}")

    for ml in max_lens:
        over = [r for r in rows if r[2] > ml]
        dead = [r for r in over if r[1] >= min(r[2], ml)]
        # 用与 dead 相反的谓词，而不是 `r not in dead`：rows 里是元组，成员判断走
        # 值比较，此刻恰好正确（是否 dead 完全由元组自身决定），但那是巧合。
        tail = [r for r in over if r[1] < min(r[2], ml)]
        print(f"  max_len={ml}: 超限 {len(over)} ({len(over) / max(n, 1):.2%})"
              + (f" {byrole(over)}" if over else "")
              + f"，其中**零梯度** {len(dead)}"
              + (f" {byrole(dead)}" if dead else "")
              + f"，砍尾 {len(tail)}"
              + (f" {byrole(tail)}" if tail else ""))
        if ml == 1024:
            chars_ruler = (len(over), len(dead))
    # 与本地那把代理尺子对一下，代理偏多少要留档
    if chars_ruler:
        print(f"\n  本地 chars/2.2 代理尺子在 1024 上给的是 44 超限 / 8 零梯度，"
              f"真值 {chars_ruler[0]} / {chars_ruler[1]}。")
        print("  两者差多少不重要，重要的是往后所有 max_len 结论都该引真值这一行。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--config", default="configs/llm/qwen3_8b.yaml")
    ap.add_argument("--data", default="data/sft_train_v23.jsonl")
    ap.add_argument("--skip-census", action="store_true",
                    help="只跑 A 段（B 段要遍历全量数据，约数十秒）")
    args = ap.parse_args()

    assert_mirrors_train_sft()

    model_path = args.model_path
    if not model_path:
        for line in open(args.config, encoding="utf-8"):
            if "model_path" in line:
                model_path = line.split(":", 1)[1].strip().strip('"').strip("'")
    assert model_path, f"未能从 {args.config} 解析 model_path（或用 --model_path）"

    from transformers import AutoTokenizer
    print(f"tokenizer = {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)

    # 取一条**真实** proposer turn：它是 M1 之后带 <interaction> 尾块的那一类，
    # 也是格式最要紧的角色。合成串量不出真实长度。
    turn = None
    for line in open(args.data, encoding="utf-8"):
        for t in json.loads(line).get("turns", []):
            if t["role_name"] == "proposer":
                turn = t
                break
        if turn:
            break
    assert turn, f"{args.data} 里没有 proposer turn"

    res = probe_template(tok, turn)
    if not args.skip_census:
        census_truncation(tok, args.data)

    print("\n" + "=" * 72)
    if res["verdict"] == "aligned":
        print("✓ 模板对齐；本脚本只报数，不拦作业。")
    elif res["verdict"] == "think_block_in_supervised_region":
        # 不用「!!」：prefix_ok=True 说明这一档对 RL 侧无害，摆成告警会误导下一个读者
        print(f"◐ 空 think 块落在监督区里，代价 {res['delta_tokens']} 个 token 的错位"
              f"监督 + loss 稀释。")
        print(f"   监督区开头 = {res['supervised'][:24]!r}")
        print("   **RL 侧安全**：rl_prompt 是 sft_full 的逐 token 前缀，推理时生成从")
        print("   块之后的「推理过程：」起，而那个位置被直接监督过——不会「再吐一个块」。")
        print(f"   #25 选修法：只补 :38 ⇒ 前缀 {res['only_38_ok']}；"
              f"两行都补 ⇒ 前缀 {res['both_flag_ok']}"
              f"（:37 传参改变渲染：{res['full_flag_matters']}）")
    else:
        print("!! SFT 见过的序列不以推理 prompt 起头（两条路走散）：")
        print(f"   SFT   prompt 尾 = {res['sft_prompt'][-24:]!r}")
        print(f"   推理  prompt 尾 = {res['rl_prompt'][-24:]!r}")
        print("   本脚本**刻意不拦**作业：这处不对称是既有的，KL 参考快照 sft_v3 与")
        print("   本次用的是同一个 train_sft.py，两边一样偏，不影响 M1 + 重放这次的")
        print("   归因；现在改掉反而是同一轮动两件事。修法单独一步（见记忆文档 §7）。")
    print("=" * 72)


if __name__ == "__main__":
    main()
