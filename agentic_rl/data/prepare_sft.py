"""SFT 数据派生：M1 块重排 + 剔除关键字段解析不出来的 turn。

（原名 `apply_m1_reorder.py`。加了第二个阶段后改名，因为脚本已不只做 M1。）

## 阶段一：`<interaction>` 块从开头移到末尾（M1）

为什么机械搬移而不是重新生成数据：M1 只改**块的位置**，重新调用 LLM 生成会
同时改掉推理内容、错误分析、分数分布，届时 sel/eff 的任何变化都无法归因到 M1。
所以这里做的是纯机械搬移——块内容、实质内容、题目、答案，一个字符都不改。

同时必须重写 `system` 字段：SFT 数据里的 system 与 `PromptTemplates` 是逐字节
相同的快照（已校验 2914/2914 全等），模板一改而数据不改，SFT 教的格式就和 RL
推理时给的格式不一致，模型会在两套 prompt 之间漂移。

## 阶段二：剔除关键字段解析不出来的 turn

既有缺陷（非 M1 引入，v2/v3 生成时就在）。实测 2914 个 turn 中 48 个的关键字段
拿不到：proposer 20（缺「最终答案：」）、critic 13（缺「错误分析/无错误」）、
verifier 15（缺「分数：」）、controller 0。**其中 45 个整条是英文**，形如
`Score: 1.0\nVerification: Using the identity ...`。

为什么是「剔除」而不是另两种修法：
- 翻译回中文不是机械操作，无法像阶段一那样逐字节自证，且会改掉实质内容；
- 只补标签别名会造出「中文标签 + 英文正文」的杂交样本，SFT 照样在教模型输出
  英文，只是让解析器看不出来——比现在更坏；
- 48/2914 = 1.6%，删掉的信息量可以忽略，而每条坏样本都在直接教错格式。

**判据用严格中文标签，与运行时解析器的宽容别名是刻意的分工**：
`agents/parsing.py` 在推理侧接受 `Score:` / `Final Answer:` / 半角冒号，是因为
模型真写了英文时，解析出来总比回落 0.5 先验或抓「文本里最后一个数字」要好；而
训练数据这一侧必须严格，因为 SFT 的目标就是把中文格式教进去。宽容用于兜底，
严格用于监督。

判据跑在**重排之后**：M1 把块移到答案后面，`parse_reasoning` 新加的
`(?=<|\n|$)` 前瞻才生效；顺序反了会把 `最终答案: 24`（半角冒号）这类本可救回
的样本误杀。

## 阶段三：把 `user` 字段里的 `<interaction>` 块剥掉

v3.2 起 RL 侧展示给下游角色的文本一律先过 `strip_interaction`（块对接收方零信息
量，却固定占 68 字符的窗口预算）。但 SFT 数据的 `user` 是当年**带着块**套模板生
成的：实测 199 个 turn 的 user 里含块（v2 的 controller 104 / critic 60 /
verifier 30 / proposer 2，v3 只 3 个）。不清洗就是一处训练/推理漂移——SFT 教模型
在「上游发言里带块」的 prompt 下作答，而 RL 推理时给的 prompt 里没有块。

清洗用 `strip_interaction(user, trim=False)`：与 RL 侧共用同一个正则字面量，
`trim=False` 是因为 user 是**已套好模板**的整段文本，再 strip 一次会改掉模板自带
的首尾空白。剥完的形状与「模板 + 剥过块的输出」逐字节一致。

用法：
    python3 data/prepare_sft.py                  # 派生 v2 与 v3，写 *_m1.jsonl
    python3 data/prepare_sft.py --check-only     # 只统计与校验，不写文件

产物 `data/sft_train_v2_m1.jsonl` / `data/sft_train_v3_m1.jsonl`，由
`submit_primus_sft.sh` 现场生成并拼接，刻意不入库（见 .gitignore 注释）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.parsing import parse_reasoning, strip_interaction  # noqa: E402
from llm.prompt_templates import PromptTemplates  # noqa: E402

# 只匹配「开头的」整块（含其后的空白），确保搬移而非复制
_HEAD_BLOCK = re.compile(r"\A\s*(<interaction>.*?</interaction>)\s*", re.S)

_SYSTEM = {
    "controller": PromptTemplates.controller_system,
    "proposer":   PromptTemplates.proposer_system,
    "critic":     PromptTemplates.critic_system,
    "verifier":   PromptTemplates.verifier_system,
}

# ---- 阶段二：关键字段的严格中文判据 -----------------------------------------
# 每个角色一条：这是该角色的输出里**下游真正会去读**的那个字段。读不到，这条
# turn 对训练就是纯噪声（甚至是反向监督）。半角冒号一律容忍——它不影响语言，
# 且运行时解析器也已容忍；容忍的只是标点，不是语言。
_CTRL_DECISION = re.compile(r"decision:\s*(?:continue|stop)")
_PROP_ANSWER = re.compile(r"最终答案[：:]")
_CRIT_VERDICT = re.compile(r"错误分析[：:]|无错误|无错")
_VERI_SCORE = re.compile(r"分数[：:]\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def keep_turn(role: str, resp: str) -> tuple[bool, str]:
    """重排后的 turn 是否合格。返回 (保留?, 不合格原因)。

    注意 proposer 是两条判据的合取：标签在、且答案抽得出来。只判标签不够——
    `最终答案：` 后面接着展开长篇论述时，`parse_reasoning` 会因超过
    `MAX_ANSWER_CHARS` 而回落，抽不到数字就返回空串；那种样本教出来的是「写了
    标签但不给答案」，进投票池是一张空票。
    """
    if role == "controller":
        return (bool(_CTRL_DECISION.search(resp)), "缺 decision")
    if role == "proposer":
        if not _PROP_ANSWER.search(resp):
            return False, "缺「最终答案：」"
        if not parse_reasoning(resp)[1]:
            return False, "答案抽不出来"
        return True, ""
    if role == "critic":
        return (bool(_CRIT_VERDICT.search(resp)), "缺「错误分析/无错误」")
    if role == "verifier":
        return (bool(_VERI_SCORE.search(resp)), "缺「分数：」")
    raise ValueError(f"未知角色：{role}")


def reorder_response(resp: str) -> tuple[str, bool]:
    """块在开头 → 移到末尾；否则原样返回。返回 (新文本, 是否改动)。"""
    m = _HEAD_BLOCK.match(resp)
    if not m:
        return resp, False
    block = m.group(1)
    body = resp[m.end():].rstrip()
    if not body:
        # 只有块没有实质内容：搬到末尾等于原样，且会造出空 response，保持不动
        return resp, False
    return f"{body}\n{block}", True


def convert(path_in: str, path_out: str | None) -> dict:
    stats = {"rows": 0, "rows_dropped": 0, "turns": 0, "moved": 0,
             "system_rewritten": 0, "no_block": 0, "dropped": 0,
             "user_cleaned": 0, "roles": {}}
    out_lines = []
    for line in open(path_in, encoding="utf-8"):
        row = json.loads(line)
        stats["rows"] += 1
        kept = []
        for t in row["turns"]:
            stats["turns"] += 1
            role = t["role_name"]
            r = stats["roles"].setdefault(
                role, {"moved": 0, "no_block": 0, "dropped": 0,
                       "user_cleaned": 0, "why": {}})

            # 顺序要紧：先重排，再判合格。见模块 docstring「判据跑在重排之后」。
            new_resp, moved = reorder_response(t["response"])
            t["response"] = new_resp

            new_sys = _SYSTEM[role]()
            if t["system"] != new_sys:
                stats["system_rewritten"] += 1
            t["system"] = new_sys

            # 阶段三：user 里的块也要剥掉，否则 SFT 的 prompt 形状与 RL 不一致
            new_user = strip_interaction(t["user"], trim=False)
            if new_user != t["user"]:
                stats["user_cleaned"] += 1
                r["user_cleaned"] += 1
            t["user"] = new_user

            ok, why = keep_turn(role, new_resp)
            if not ok:
                stats["dropped"] += 1
                r["dropped"] += 1
                r["why"][why] = r["why"].get(why, 0) + 1
                continue

            if moved:
                stats["moved"] += 1
                r["moved"] += 1
            else:
                stats["no_block"] += 1
                r["no_block"] += 1
            kept.append(t)

        if not kept:
            # 整条 episode 都不合格。保留空 turns 会让 train_sft.py 拿到零样本的
            # episode，不如整行去掉；行号对不上无妨，verify() 是按内容重算的。
            stats["rows_dropped"] += 1
            continue
        row["turns"] = kept
        out_lines.append(json.dumps(row, ensure_ascii=False))
    if path_out:
        with open(path_out, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
    return stats


def verify(path_in: str, path_out: str) -> None:
    """逐 turn 对照原文件：只允许块位置、system、user 剥块、整条剔除四种变化。

    这是本脚本的核心保障。这里**独立重算**一遍该保留哪些 turn（而不是相信
    convert 的输出顺序），再与产物逐字节对齐：把新 response 的尾部块摘掉、旧
    response 的头部块摘掉，剩下的实质内容必须完全一致（空白差异只允许末尾
    strip）；两处的块文本本身也必须完全一致。行数与存活 turn 数也要对上——这样
    「剔除逻辑写错、多删或少删」会在这里当场炸掉，而不是变成一次静默的数据缩水。

    `user` 一侧同理是逐字节判的：产物必须恰好等于「原 user 剥掉块」，多改一个
    字符就失败。
    """
    tail = re.compile(r"\n(<interaction>.*?</interaction>)\s*\Z", re.S)
    n = 0
    with open(path_out, encoding="utf-8") as fb:
        out_rows = [json.loads(x) for x in fb if x.strip()]
    bi = 0
    for la in open(path_in, encoding="utf-8"):
        a = json.loads(la)
        # 预期存活的 turn：重排后过一遍同一个判据
        expect = [ta for ta in a["turns"]
                  if keep_turn(ta["role_name"], reorder_response(ta["response"])[0])[0]]
        if not expect:
            continue
        assert bi < len(out_rows), "产物行数少于预期（有该保留的 episode 被丢了）"
        b = out_rows[bi]
        bi += 1
        assert a["question"] == b["question"], "question 被改动"
        assert a["answer"] == b["answer"], "answer 被改动"
        assert len(expect) == len(b["turns"]), (
            f"存活 turn 数不符：预期 {len(expect)}，产物 {len(b['turns'])}")
        for ta, tb in zip(expect, b["turns"]):
            n += 1
            assert ta["role_name"] == tb["role_name"], "role_name 被改动"
            assert strip_interaction(ta["user"], trim=False) == tb["user"], \
                "user prompt 除了剥块之外还被改动了"
            ma = _HEAD_BLOCK.match(ta["response"])
            if ma is None:
                assert ta["response"] == tb["response"], \
                    f"无块的 turn 被改动：{ta['role_name']}"
                continue
            mb = tail.search(tb["response"])
            assert mb is not None, f"块没落到末尾：{tb['response'][-80:]!r}"
            assert ma.group(1) == mb.group(1), "块内容被改动"
            body_a = ta["response"][ma.end():].rstrip()
            body_b = tb["response"][:mb.start()]
            assert body_a == body_b, (
                f"实质内容被改动\n旧：{body_a[:120]!r}\n新：{body_b[:120]!r}")
    assert bi == len(out_rows), f"产物多出 {len(out_rows) - bi} 行"
    print(f"  ✓ 逐 turn 校验通过（{n} turn）："
          f"只有块位置、system、user 剥块、剔除四种变化")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=["data/sft_train_v2.jsonl", "data/sft_train_v3.jsonl"])
    ap.add_argument("--suffix", default="_m1")
    ap.add_argument("--check-only", action="store_true")
    # 剔除率上限，**按单个输入文件**判。实测：v2 44/1765 = 2.49%（全部剔除都在
    # v2），v3 0/1149 = 0%。所以阈值要按 v2 定，0.03 只剩 0.5pp 余量太紧。判据写
    # 错（比如某个角色的标签改了名）会表现为剔除率暴涨，这里当场停住，而不是安静
    # 地交出一份缩水的训练集。
    ap.add_argument("--max-drop-ratio", type=float, default=0.05)
    args = ap.parse_args()

    total_moved = total_dropped = total_cleaned = 0
    for path_in in args.inputs:
        stem, ext = os.path.splitext(path_in)
        path_out = None if args.check_only else f"{stem}{args.suffix}{ext}"
        st = convert(path_in, path_out)
        total_moved += st["moved"]
        total_dropped += st["dropped"]
        total_cleaned += st["user_cleaned"]
        ratio = st["dropped"] / max(st["turns"], 1)
        print(f"{path_in} → {path_out or '(dry-run)'}")
        print(f"  行={st['rows']}（整行剔除 {st['rows_dropped']}） "
              f"turn={st['turns']} 搬移={st['moved']} 无块={st['no_block']} "
              f"剔除={st['dropped']}（{ratio:.2%}） "
              f"user 剥块={st['user_cleaned']} "
              f"system 重写={st['system_rewritten']}")
        for role, r in sorted(st["roles"].items()):
            why = "，".join(f"{k}×{v}" for k, v in sorted(r["why"].items()))
            print(f"    {role:11s} 搬移={r['moved']:5d} 无块={r['no_block']:5d} "
                  f"剔除={r['dropped']:3d} user 剥块={r['user_cleaned']:4d}"
                  + (f"（{why}）" if why else ""))
        assert ratio <= args.max_drop_ratio, (
            f"剔除率 {ratio:.2%} 超过上限 {args.max_drop_ratio:.2%}——"
            f"先查 keep_turn 的判据是不是和数据格式脱节了，不要直接放宽阈值")
        if path_out:
            verify(path_in, path_out)
    print(f"\n合计搬移 {total_moved} 个 turn，剔除 {total_dropped} 个，"
          f"user 剥块 {total_cleaned} 个。")


if __name__ == "__main__":
    main()
