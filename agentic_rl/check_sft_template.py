"""SFT 起跑前的 tokenizer 级预检（只加载 tokenizer，不碰 GPU，秒级）。

存在的理由：这两个读数**只能在有 tokenizer 的机器上取到**，本地开发机拿不到，
而它们决定了一次 3 epoch 的 SFT 到底训了什么。

── A. 模板对齐：`enable_thinking` 的那处不对称 ──────────────────────────
全仓 6 处 `apply_chat_template`，5 处显式传 `enable_thinking=False`：

    agents/agentic_executor.py:109   rollout 取样（vLLM prompt）
    training/grpo_trainer.py:241     RL 前向（算 new_lps）
    verify_sft_format.py:87          SFT 格式验收
    measure_channels.py:99           离线测量
    generate_sft_v3.py:114           **生成 v3 SFT 数据本身**

只有 `train_sft.py:37-38`（也就是真正训练的那一处）两行都没传，取 Jinja 默认
（Qwen3 模板里 `enable_thinking` 未定义 ⇒ 不注入空 think 块）。

为什么可能坏：Qwen3 的模板对**最后一条 assistant 消息**有专门分支
（`loop.last` ⇒ 渲染成 `<think>\n{reasoning}\n</think>\n\n{content}`）。若该分支
生效，则 `train_sft.py` 的
    full_text   = 模板(sys,user,assistant)  ← 含空 think 块
    prompt_text = 模板(sys,user, gen=True)  ← **不**含空 think 块
两者在 assistant 头之后就分叉，于是 `labels` 的监督区从空 think 块开始
——SFT 教模型"先吐一个空 think 块再答"，而推理时那个块**已经在 prompt 里了**。
若该分支不生效，则监督区正好是 response，只是 SFT 的 prompt 尾比推理时短 5 个
token。两种情形的后果差很远，而**分不清就等于没测**。

判据取最干净的那个不变量：推理时喂给模型的是 `rl_prompt`，那么 SFT 见过的整条
序列就必须以 `rl_prompt` 起头。`rl_prompt` 不是 `sft_full` 的前缀 ⇒ 两条路分叉。

本脚本**只报数、不拦**（exit 0）。原因是：这处不对称是既有的，当前
KL 参考快照 `sft_v3` 与本次 SFT 用的是同一个 `train_sft.py`，**两边一样偏**，
所以它不影响"M1 + 重放"这次的归因；而现在改掉它等于同一轮动两件事，反而毁掉
对照。先拿读数，修法单独一步。

── B. 截断普查：把 chars/2.2 的估计换成真 token 数 ────────────────────
`submit_primus_sft.sh` 里的体检用 `(len(system)+len(user)+len(response))/2.2`
估 token，这是个代理指标；"1024 截断率 1.5%、抬到 1536 只剩 3 条"这条结论一直
建在这个代理上。有 tokenizer 就该量真的。

并且要把"超限"拆成两种，因为后果完全不同（`train_sft.py:42` + `:96`）：
    ① response 被砍尾    → 仍有梯度，只是丢尾巴（M1 之后丢的是 <interaction> 块）
    ② prompt 本身就超限  → `labels` 全 -100 → `response_mask.sum()==0` → **整条
       turn 一点梯度都不产生**，等于白算一遍前向
本地按 chars/2.2 估出来是 44 条超限、其中 8 条属②，这里给出真值。

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
    """A 段：三种渲染逐字节对照。"""
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
    print(f"\n  token 数：sft_full={len(full_ids)} sft_prompt={len(sft_pids)} "
          f"rl_prompt={len(rl_pids)}（rl − sft = {len(rl_pids) - len(sft_pids):+d}）")
    print(f"  不变量「rl_prompt 是 sft_full 的前缀」：{prefix_ok}")
    if prefix_ok and same_prompt:
        verdict = "aligned"
        print("  → 对齐。A 段这处不对称在本 tokenizer 上无后果。")
    elif prefix_ok:
        verdict = "think_block_in_supervised_region"
        print("  → 分叉，但推理 prompt 仍是 SFT 序列的前缀：也就是空 think 块落在了"
              "\n     监督区里。SFT 在教模型再吐一个块，而推理时块已在 prompt 中。")
    else:
        verdict = "diverged"
        print("  → **不对齐**：推理喂进去的前缀 SFT 从未见过。这一条比上一条更重，"
              "\n     说明两条路在 assistant 头之后就走散了。")
    return {"verdict": verdict, "same_prompt": same_prompt,
            "prefix_ok": prefix_ok, "supervised": supervised,
            "sft_prompt": sft_prompt, "rl_prompt": rl_prompt,
            "delta_tokens": len(rl_pids) - len(sft_pids)}


def census_truncation(tok, path, max_lens=(1024, 1536, 2048)):
    """B 段：真 token 数的截断普查 + 零梯度拆分。"""
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
    print(f"  {n} 个 turn，full token 数：中位 "
          f"{sorted(r[2] for r in rows)[n // 2]}，最长 {max(r[2] for r in rows)}")
    for ml in max_lens:
        over = [r for r in rows if r[2] > ml]
        dead = [r for r in over if r[1] >= min(r[2], ml)]
        byrole = {}
        for r in dead:
            byrole[r[0]] = byrole.get(r[0], 0) + 1
        print(f"  max_len={ml}: 超限 {len(over)} ({len(over) / max(n, 1):.2%})，"
              f"其中**零梯度** {len(dead)}"
              + (f" {byrole}" if byrole else ""))
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
    else:
        if res["verdict"] == "think_block_in_supervised_region":
            print("!! SFT 的监督区从空 think 块开始，而推理时那个块已经在 prompt 里：")
            print(f"   监督区开头 = {res['supervised'][:24]!r}")
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
