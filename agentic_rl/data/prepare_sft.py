"""SFT 数据派生：M1 块重排 + v2 prompt 重放 + 剔除关键字段解析不出来的 turn。

（原名 `apply_m1_reorder.py`。加了后面几个阶段后改名，因为脚本已不只做 M1。）

## 阶段一：`<interaction>` 块从开头移到末尾（M1）

为什么机械搬移而不是重新生成数据：M1 只改**块的位置**，重新调用 LLM 生成会
同时改掉推理内容、错误分析、分数分布，届时 sel/eff 的任何变化都无法归因到 M1。
所以这里做的是纯机械搬移——块内容、实质内容、题目、答案，一个字符都不改。

同时必须重写 `system` 字段：SFT 数据里的 system 与 `PromptTemplates` 是逐字节
相同的快照（已校验 2914/2914 全等），模板一改而数据不改，SFT 教的格式就和 RL
推理时给的格式不一致，模型会在两套 prompt 之间漂移。

## 阶段二：把 v2 的 `user` 重放成 RL 形状（`replay_v2_prompts.py`）

v2 那批数据是当初让 API 整段编出来的，**连 `user` 一起编**——于是 SFT 教模型去
读一句人话摘要，而 RL 上线后给的是 `Blackboard.to_text()` 的结构化转储。实测
v2 的 1765 个 turn 里只有 5 个含 `当前状态：`，v3 与 RL 侧都是 100%。最要紧的是
**884 条 controller turn 全在 v2**，也就是 controller 一次都没在带黑板的 prompt
上被 SFT 过。重放按 turn 顺序把更早的 response 喂进一块真 `Blackboard`，再用
executor 里同样的字面量渲染 `user`；`response` 一字不动。可行性证据、六种拓扑
的可达性论证、以及一处刻意的不忠实（561 条 controller turn 的位置）都写在
`replay_v2_prompts.py` 的 docstring 里。

顺序是被逼死的：必须在阶段一**之后**（黑板喂的是 `parse_reasoning(response)`，
其前瞻按块在尾部设计），且必须在剔除**之前**（重放要看完整 turn 列表，先剔就会
让后续 turn 看到一份 RL 不会呈现的黑板状态）。

## 阶段三：剔除关键字段解析不出来的 turn

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
`(?=<interaction|\n|$)` 前瞻才生效；顺序反了会把 `最终答案: 24`（半角冒号）这类
本可救回的样本误杀。

## 阶段四：把 `user` 字段里的 `<interaction>` 块剥掉

v3.2 起 RL 侧展示给下游角色的文本一律先过 `strip_interaction`（块对接收方零信息
量，却固定占 68 字符的窗口预算）。但 SFT 数据的 `user` 是当年**带着块**套模板生
成的：实测 199 个 turn 的 user 里含块（v2 的 controller 104 / critic 60 /
verifier 30 / proposer 2，v3 只 3 个）。不清洗就是一处训练/推理漂移——SFT 教模型
在「上游发言里带块」的 prompt 下作答，而 RL 推理时给的 prompt 里没有块。

那 199 是**加阶段二之前**量的。重放把 v2 的 `user` 整个换掉，所以这一步现在实际
只剩 v3 的 3 个要清（跑出来就是 v2 剥块 0、v3 剥块 3）；v2 那 196 个不是没管，是
在阶段二连同整段 prompt 一起重建掉了。

清洗用 `strip_interaction(user, trim=False)`：与 RL 侧共用同一个正则字面量，
`trim=False` 是因为 user 是**已套好模板**的整段文本，再 strip 一次会改掉模板自带
的首尾空白。剥完的形状与「模板 + 剥过块的输出」逐字节一致。对重放过的行这一步
必然是空操作（重放本来就用剥过块的文本拼 prompt），代码里对此有断言。

用法：
    python3 data/prepare_sft.py                  # 派生 v2 与 v3，写 *_m1.jsonl
    python3 data/prepare_sft.py --check-only     # 只统计与校验，不写文件

产物 `data/sft_train_v2_m1.jsonl` / `data/sft_train_v3_m1.jsonl`，由
`submit_primus_sft.sh` 现场生成并拼接，刻意不入库（见 .gitignore 注释）。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.parsing import parse_reasoning, strip_interaction  # noqa: E402
from data.replay_v2_prompts import (  # noqa: E402
    Unreplayable,
    needs_replay,
    replay_row,
)
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
    stats = {"rows": 0, "rows_dropped": 0, "rows_replayed": 0,
             "rows_unreplayable": 0, "why_unreplayable": {},
             "turns": 0, "moved": 0,
             "system_rewritten": 0, "no_block": 0, "dropped": 0,
             "user_cleaned": 0, "user_replayed": 0, "roles": {}}
    out_lines = []
    for line in open(path_in, encoding="utf-8"):
        row = json.loads(line)
        stats["rows"] += 1

        # ── 阶段一：整行的 response 重排 + system 重写 ──────────────────────
        # 必须在重放之前跑完**整行**：黑板喂的是 parse_reasoning(response)，而它的
        # `最终答案：` 前瞻按块在尾部设计。
        moved_flags = []
        for t in row["turns"]:
            new_resp, moved = reorder_response(t["response"])
            t["response"] = new_resp
            moved_flags.append(moved)
            new_sys = _SYSTEM[t["role_name"]]()
            if t["system"] != new_sys:
                stats["system_rewritten"] += 1
            t["system"] = new_sys

        # ── 阶段二：把旧形状的 user 重建成 RL 形状 ─────────────────────────
        # 顺序要紧：重放**必须看到完整的 turn 列表**。先过 keep_turn 再重放，被剔
        # 掉那个 turn 的产出就不会进黑板，后续 turn 看到的状态与 RL 真实呈现的不
        # 一致——那等于用一份自造的状态去教模型，正是这个阶段在修的病。
        replayed = False
        if needs_replay(row):
            try:
                replay_row(row)
            except Unreplayable as e:
                stats["rows_unreplayable"] += 1
                why = str(e)
                stats["why_unreplayable"][why] = \
                    stats["why_unreplayable"].get(why, 0) + 1
                continue                    # 整行剔除，不猜
            replayed = True
            stats["rows_replayed"] += 1

        # ── 阶段三/四：user 剥块 + 剔除关键字段解析不出来的 turn ────────────
        kept = []
        for t, moved in zip(row["turns"], moved_flags):
            stats["turns"] += 1
            role = t["role_name"]
            r = stats["roles"].setdefault(
                role, {"moved": 0, "no_block": 0, "dropped": 0,
                       "user_cleaned": 0, "user_replayed": 0, "why": {}})
            if replayed:
                stats["user_replayed"] += 1
                r["user_replayed"] += 1

            new_user = strip_interaction(t["user"], trim=False)
            if new_user != t["user"]:
                # 重放出来的 user 是用**剥过块的** response 拼的，这里不该再剥出
                # 东西来。真剥掉了说明重放漏了一处 strip_interaction，那块就会
                # 顶着 68 字符的窗口预算进 SFT——正是 v3.2 刚修掉的那个病。
                assert not replayed, (
                    f"重放产生的 user 里仍有 <interaction> 块（{role}）："
                    f"{t['user'][:120]!r}")
                stats["user_cleaned"] += 1
                r["user_cleaned"] += 1
            t["user"] = new_user

            ok, why = keep_turn(role, t["response"])
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
    """独立重推一遍整条派生链，再与产物逐字节对齐。

    这是本脚本的核心保障，关键在于「独立」：这里从**原始行的深拷贝**重跑
    重排 → 重放 → 剔除，而不是相信 convert 的中间结果。最要防的一类错就是
    **阶段顺序写反**——若 convert 先剔 turn 再重放，被剔掉那个 turn 的产出不会
    进黑板，后续 user 里的黑板段就与这里从完整 turn 列表推出的不同，比较当场
    失败。这种错不会抛异常，只会静默地交出一份「状态是自造的」训练集。

    response 一侧仍是老判据：把新 response 的尾部块摘掉、旧 response 的头部块摘
    掉，剩下的实质内容必须完全一致；块文本本身也必须完全一致。

    user 一侧分两种：重放过的行额外要求每条 user 都含黑板段、确实与手写原文不同、
    且不残留 `<interaction>` 块；没重放的行则必须恰好等于「原 user 剥掉块」。
    """
    tail = re.compile(r"\n(<interaction>.*?</interaction>)\s*\Z", re.S)
    n_replayed = n_plain = 0
    with open(path_out, encoding="utf-8") as fb:
        out_rows = [json.loads(x) for x in fb if x.strip()]
    bi = 0
    for la in open(path_in, encoding="utf-8"):
        a = json.loads(la)

        # —— 独立重推：深拷贝 → 重排 → 重放/剥块 → 剔除 ——
        full = copy.deepcopy(a)
        for tf in full["turns"]:
            tf["response"] = reorder_response(tf["response"])[0]
        row_replayed = needs_replay(a)
        if row_replayed:
            try:
                replay_row(full)
            except Unreplayable:
                continue                    # convert 也整行剔除了，跳过
        for tf in full["turns"]:
            tf["user"] = strip_interaction(tf["user"], trim=False)
        # (原始 turn, 期望 turn) 配对：response 判据要拿原文比，user 判据要拿期望比
        expect = [(ta, tf) for ta, tf in zip(a["turns"], full["turns"])
                  if keep_turn(tf["role_name"], tf["response"])[0]]
        if not expect:
            continue

        assert bi < len(out_rows), "产物行数少于预期（有该保留的 episode 被丢了）"
        b = out_rows[bi]
        bi += 1
        assert a["question"] == b["question"], "question 被改动"
        assert a["answer"] == b["answer"], "answer 被改动"
        assert len(expect) == len(b["turns"]), (
            f"存活 turn 数不符：预期 {len(expect)}，产物 {len(b['turns'])}")

        for (ta, tf), tb in zip(expect, b["turns"]):
            assert ta["role_name"] == tb["role_name"], "role_name 被改动"

            # —— user ——
            assert tf["user"] == tb["user"], (
                f"user 与独立重推的结果不一致（{tb['role_name']}）——"
                f"最可能是阶段顺序错了（先剔 turn 后重放）\n"
                f"期望：{tf['user'][:160]!r}\n产物：{tb['user'][:160]!r}")
            if row_replayed:
                n_replayed += 1
                assert "当前状态：" in tb["user"], (
                    f"重放后的 user 仍缺黑板段（{tb['role_name']}）："
                    f"{tb['user'][:120]!r}")
                assert "<interaction>" not in tb["user"], "重放的 user 残留块"
                assert tb["user"] != ta["user"], (
                    f"重放没换掉手写 prompt（{tb['role_name']}）")
            else:
                n_plain += 1
                assert strip_interaction(ta["user"], trim=False) == tb["user"], \
                    "未重放的 user 除了剥块之外还被改动了"

            # —— response ——
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
    print(f"  ✓ 逐 turn 校验通过（重放 {n_replayed} + 原形 {n_plain} = "
          f"{n_replayed + n_plain} turn）：response 只有块位置变化，"
          f"user 与独立重推逐字节相同")


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
    # 不可重建行的上限，同样按文件判。实测 v2 是 1/323 = 0.31%（唯一那条是
    # proposer 修正轮缺发起方上下文），v3 触发不到重放。这个数暴涨的含义很具体：
    # 重放器对 pending 上下文的重建与 executor 脱节了（比如 executor 改了硬触发
    # 修正的条件），**要去对齐两边，而不是放宽这里**。
    ap.add_argument("--max-unreplayable-ratio", type=float, default=0.02)
    args = ap.parse_args()

    total_moved = total_dropped = total_cleaned = total_replayed = 0
    for path_in in args.inputs:
        stem, ext = os.path.splitext(path_in)
        path_out = None if args.check_only else f"{stem}{args.suffix}{ext}"
        st = convert(path_in, path_out)
        total_moved += st["moved"]
        total_dropped += st["dropped"]
        total_cleaned += st["user_cleaned"]
        total_replayed += st["user_replayed"]
        ratio = st["dropped"] / max(st["turns"], 1)
        unrep_ratio = st["rows_unreplayable"] / max(st["rows"], 1)
        print(f"{path_in} → {path_out or '(dry-run)'}")
        print(f"  行={st['rows']}（整行剔除 {st['rows_dropped']}） "
              f"turn={st['turns']} 搬移={st['moved']} 无块={st['no_block']} "
              f"剔除={st['dropped']}（{ratio:.2%}） "
              f"user 剥块={st['user_cleaned']} "
              f"system 重写={st['system_rewritten']}")
        why_u = "，".join(f"{k}×{v}"
                          for k, v in sorted(st["why_unreplayable"].items()))
        print(f"  重放：行={st['rows_replayed']} turn={st['user_replayed']} "
              f"不可重建行={st['rows_unreplayable']}（{unrep_ratio:.2%}）"
              + (f"（{why_u}）" if why_u else ""))
        for role, r in sorted(st["roles"].items()):
            why = "，".join(f"{k}×{v}" for k, v in sorted(r["why"].items()))
            print(f"    {role:11s} 搬移={r['moved']:5d} 无块={r['no_block']:5d} "
                  f"剔除={r['dropped']:3d} user 剥块={r['user_cleaned']:4d} "
                  f"user 重放={r['user_replayed']:5d}"
                  + (f"（{why}）" if why else ""))
        assert ratio <= args.max_drop_ratio, (
            f"剔除率 {ratio:.2%} 超过上限 {args.max_drop_ratio:.2%}——"
            f"先查 keep_turn 的判据是不是和数据格式脱节了，不要直接放宽阈值")
        assert unrep_ratio <= args.max_unreplayable_ratio, (
            f"不可重建行占比 {unrep_ratio:.2%} 超过上限 "
            f"{args.max_unreplayable_ratio:.2%}——去查重放器与 executor 的 pending "
            f"上下文重建是否脱节，不要直接放宽阈值")
        if path_out:
            verify(path_in, path_out)
    print(f"\n合计搬移 {total_moved} 个 turn，剔除 {total_dropped} 个，"
          f"user 剥块 {total_cleaned} 个，user 重放 {total_replayed} 个。")


if __name__ == "__main__":
    main()
