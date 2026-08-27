"""把 SFT 数据机械重排成 M1 布局（`<interaction>` 块从开头移到末尾）。

为什么需要这一步而不是重新生成数据：M1 只改**块的位置**，重新调用 LLM 生成会
同时改掉推理内容、错误分析、分数分布，届时 sel/eff 的任何变化都无法归因到 M1。
所以这里做的是纯机械搬移——块内容、实质内容、题目、答案，一个字符都不改。

同时必须重写 `system` 字段：SFT 数据里的 system 与 `PromptTemplates` 是逐字节
相同的快照（已校验 2914/2914 全等），模板一改而数据不改，SFT 教的格式就和 RL
推理时给的格式不一致，模型会在两套 prompt 之间漂移。

用法：
    python3 data/apply_m1_reorder.py                  # 就地重排 v2 与 v3，写 *_m1.jsonl
    python3 data/apply_m1_reorder.py --check-only     # 只校验，不写文件

产物 `data/sft_train_v2_m1.jsonl` / `data/sft_train_v3_m1.jsonl`；
`submit_primus_sft.sh` 里拼接的两个输入需相应改名。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.prompt_templates import PromptTemplates  # noqa: E402

# 只匹配「开头的」整块（含其后的空白），确保搬移而非复制
_HEAD_BLOCK = re.compile(r"\A\s*(<interaction>.*?</interaction>)\s*", re.S)

_SYSTEM = {
    "controller": PromptTemplates.controller_system,
    "proposer":   PromptTemplates.proposer_system,
    "critic":     PromptTemplates.critic_system,
    "verifier":   PromptTemplates.verifier_system,
}


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
    stats = {"rows": 0, "turns": 0, "moved": 0, "system_rewritten": 0,
             "no_block": 0, "roles": {}}
    out_lines = []
    for line in open(path_in, encoding="utf-8"):
        row = json.loads(line)
        stats["rows"] += 1
        for t in row["turns"]:
            stats["turns"] += 1
            role = t["role_name"]
            stats["roles"].setdefault(role, {"moved": 0, "no_block": 0})

            new_resp, moved = reorder_response(t["response"])
            if moved:
                stats["moved"] += 1
                stats["roles"][role]["moved"] += 1
            else:
                stats["no_block"] += 1
                stats["roles"][role]["no_block"] += 1
            t["response"] = new_resp

            new_sys = _SYSTEM[role]()
            if t["system"] != new_sys:
                stats["system_rewritten"] += 1
            t["system"] = new_sys
        out_lines.append(json.dumps(row, ensure_ascii=False))
    if path_out:
        with open(path_out, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
    return stats


def verify(path_in: str, path_out: str) -> None:
    """逐 turn 对照原文件：只允许块位置与 system 两处变化，其余必须逐字节相同。

    这是本脚本的核心保障。判据：把新 response 的尾部块摘掉、旧 response 的头部
    块摘掉，剩下的实质内容必须完全一致（含空白差异只允许末尾 strip）；两处的块
    文本本身也必须完全一致。
    """
    tail = re.compile(r"\n(<interaction>.*?</interaction>)\s*\Z", re.S)
    n = 0
    with open(path_in, encoding="utf-8") as fa, open(path_out, encoding="utf-8") as fb:
        for la, lb in zip(fa, fb):
            a, b = json.loads(la), json.loads(lb)
            assert a["question"] == b["question"], "question 被改动"
            assert a["answer"] == b["answer"], "answer 被改动"
            assert len(a["turns"]) == len(b["turns"]), "turn 数变了"
            for ta, tb in zip(a["turns"], b["turns"]):
                n += 1
                assert ta["role_name"] == tb["role_name"], "role_name 被改动"
                assert ta["user"] == tb["user"], "user prompt 被改动"
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
    print(f"  ✓ 逐 turn 校验通过（{n} turn）：只有块位置与 system 两处变化")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=["data/sft_train_v2.jsonl", "data/sft_train_v3.jsonl"])
    ap.add_argument("--suffix", default="_m1")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    total = 0
    for path_in in args.inputs:
        stem, ext = os.path.splitext(path_in)
        path_out = None if args.check_only else f"{stem}{args.suffix}{ext}"
        st = convert(path_in, path_out)
        total += st["moved"]
        print(f"{path_in} → {path_out or '(dry-run)'}")
        print(f"  行={st['rows']} turn={st['turns']} 搬移={st['moved']} "
              f"无块={st['no_block']} system 重写={st['system_rewritten']}")
        for role, r in sorted(st["roles"].items()):
            print(f"    {role:11s} 搬移={r['moved']:5d} 无块={r['no_block']:5d}")
        if path_out:
            verify(path_in, path_out)
    print(f"\n合计搬移 {total} 个 turn。")


if __name__ == "__main__":
    main()
