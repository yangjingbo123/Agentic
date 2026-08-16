"""把 v1 格式的 SFT 数据确定性转换为 RACA v2 格式（零 API 成本）。

转换规则：
- controller：system 换 v2 模板；<meta-plan> 的 strategy/focus →
  decision: continue|stop（stop 保留，其余一律 continue），reason 保留
- proposer/critic/verifier：system 换 v2 模板；<interaction> 的
  request_xxx → action: request + target: xxx；
  support* → action: none（v2 删除 support）；
  challenge* → action: challenge（target 缺失/非法则降级 none）；
  其余正文（推理过程/错误分析/分数）原样保留

用法：python data/convert_sft_v2.py [in.jsonl] [out.jsonl]
默认：data/sft_train.jsonl → data/sft_train_v2.jsonl
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.prompt_templates import PromptTemplates  # noqa: E402

V2_SYSTEM = {
    "controller": PromptTemplates.controller_system(),
    "proposer":   PromptTemplates.proposer_system(),
    "critic":     PromptTemplates.critic_system(),
    "verifier":   PromptTemplates.verifier_system(),
}
ROLES = ("proposer", "critic", "verifier")


def convert_controller_response(resp: str) -> str:
    m = re.search(r"strategy:\s*(\w+)", resp)
    strategy = m.group(1) if m else "explore"
    decision = "stop" if strategy == "stop" else "continue"
    rm = re.search(r"reason:\s*(.+)", resp)
    reason = rm.group(1).strip() if rm else ""
    return f"<meta-plan>\ndecision: {decision}\nreason: {reason}\n</meta-plan>"


def convert_worker_response(resp: str) -> str:
    block = re.search(r"<interaction>(.*?)</interaction>", resp, re.S)
    if not block:
        return resp
    content = block.group(1)
    am = re.search(r"action:\s*(\S+)", content)
    tm = re.search(r"target:\s*(\S+)", content)
    rm = re.search(r"reason:\s*(.+)", content)
    action_raw = (am.group(1).strip().lower() if am else "none")
    target = (tm.group(1).strip().lower() if tm else "none")
    reason = rm.group(1).strip() if rm else ""

    if action_raw.startswith("request"):
        action = "request"
        # request_critic / request:critic → target 从 action 后缀提取
        suf = re.split(r"[_:]", action_raw, 1)
        if len(suf) == 2 and suf[1] in ROLES:
            target = suf[1]
    elif action_raw.startswith("challenge"):
        action = "challenge"
    else:
        action, target = "none", "none"       # none / support* 一律 none

    if action != "none" and target not in ROLES:
        action, target = "none", "none"       # 无合法 target 的动作降级
    if action == "none":
        target = "none"

    new_block = f"<interaction>\naction: {action}\ntarget: {target}\nreason: {reason}\n</interaction>"
    return resp[:block.start()] + new_block + resp[block.end():]


def convert(in_path: str, out_path: str):
    n_ep = n_turn = 0
    stats = {"stop": 0, "request": 0, "challenge": 0, "none": 0}
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            ep = json.loads(line)
            for turn in ep.get("turns", []):
                role = turn.get("role_name", "")
                if role not in V2_SYSTEM:
                    continue
                turn["system"] = V2_SYSTEM[role]
                if role == "controller":
                    turn["response"] = convert_controller_response(turn["response"])
                    if "decision: stop" in turn["response"]:
                        stats["stop"] += 1
                else:
                    turn["response"] = convert_worker_response(turn["response"])
                    for k in ("request", "challenge", "none"):
                        if f"action: {k}" in turn["response"]:
                            stats[k] += 1
                            break
                n_turn += 1
            fout.write(json.dumps(ep, ensure_ascii=False) + "\n")
            n_ep += 1
    print(f"converted {n_ep} episodes / {n_turn} turns -> {out_path}")
    print(f"stats: {stats}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/sft_train.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/sft_train_v2.jsonl"
    convert(src, dst)
