"""SFT checkpoint 格式验收（RL 启动前的最后一道闸）。

只看 SFT loss 无法判断模型是否学会 v2 输出格式，而格式是 v2 的命脉：
- controller 不输出 decision: stop → stop_gate 下每个 episode 跑满 max_rounds
- verifier 不输出「分数:」→ 黑板永远没有分数 → stop 闸门死锁
- proposer 不输出「最终答案：」→ 答案靠抽末尾数字兜底 → parse_rate 崩
- <interaction> 块缺失 → 交互率恒 0，v2 的核心机制形同虚设

本脚本走 load_trainable_models 的真实 RL 加载路径，因此同时验证：
① checkpoint 能被正确加载（打印 loaded 参数量）② 四角色格式可解析率。

用法：
    python verify_sft_format.py --ckpt checkpoints/sft_v2 [--n 8]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.parsing import (                       # noqa: E402
    critic_found_errors, parse_decision, parse_interaction,
    parse_reasoning, parse_score,
)
from llm.prompt_templates import PromptTemplates   # noqa: E402
from llm.trainable_llm import load_trainable_models, ROLE_ADAPTER  # noqa: E402


def build_prompts(questions):
    """构造四角色的 (role, system, user)，尽量贴近 executor 真实调用形态。"""
    bb_empty = "尚无信息"
    bb_with_trace = "已有1个解法，答案：['42']"
    out = []
    for qq in questions:
        out.append(("controller", PromptTemplates.controller_system(),
                    f"问题：{qq}\n当前状态：{bb_empty}"))
        out.append(("controller", PromptTemplates.controller_system(),
                    f"问题：{qq}\n当前状态：{bb_with_trace}\n最高置信答案：42（分数0.90）"))
        out.append(("proposer", PromptTemplates.proposer_system(),
                    f"问题：{qq}\n当前状态：{bb_empty}"))
        out.append(("critic", PromptTemplates.critic_system(),
                    f"待审查解法：先算 6*7=42\n答案：42\n当前状态：{bb_with_trace}"))
        out.append(("verifier", PromptTemplates.verifier_system(),
                    f"待验证答案：42\n推理：先算 6*7=42\n当前状态：{bb_with_trace}"))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft_v2")
    ap.add_argument("--config", default="configs/llm/qwen3_8b.yaml")
    ap.add_argument("--n", type=int, default=6, help="抽多少道题（每题 5 个 prompt）")
    ap.add_argument("--max-new", type=int, default=400)
    args = ap.parse_args()

    model_path = None
    for line in open(args.config):
        if "model_path" in line:
            model_path = line.split(":", 1)[1].strip().strip('"').strip("'")
    assert model_path, f"未能从 {args.config} 解析 model_path"

    print(f"加载 base={model_path}\n     sft={args.ckpt}", flush=True)
    model, tokenizer = load_trainable_models(model_path, sft_checkpoint=args.ckpt)
    model._model.eval()
    # generate 需要 kv cache；SFT/RL 训练时关掉了
    model._model.config.use_cache = True

    questions = [json.loads(l)["question"]
                 for l in open("data/math_test.jsonl")][:args.n]
    prompts = build_prompts(questions)

    stats = {}
    samples = {}
    for role, system, user in prompts:
        model._model.set_adapter(ROLE_ADAPTER[role])
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        out = model._model.generate(
            **ids, max_new_tokens=args.max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        s = stats.setdefault(role, {"n": 0, "ok": 0, "stop": 0, "act": {},
                                    "flag": 0, "score": 0})
        s["n"] += 1
        if role == "controller":
            d = parse_decision(resp)
            # 「有 <meta-plan> 且能解析出 decision 字面量」才算格式正确
            s["ok"] += ("decision:" in resp)
            s["stop"] += d == "stop"
        else:
            a, t, _ = parse_interaction(resp)
            s["act"][a] = s["act"].get(a, 0) + 1
            if role == "proposer":
                _, ans = parse_reasoning(resp)
                s["ok"] += ("最终答案：" in resp and bool(ans))
            elif role == "critic":
                s["ok"] += ("错误分析" in resp)
                s["flag"] += critic_found_errors(resp)
            else:
                sc = parse_score(resp)
                s["ok"] += sc is not None
                s["score"] += sc is not None
        samples.setdefault(role, resp)

    print("\n" + "=" * 70)
    print("格式可解析率")
    print("=" * 70)
    fatal = []
    for role, s in stats.items():
        rate = s["ok"] / max(s["n"], 1)
        flag = "ok  " if rate >= 0.9 else ("WARN" if rate >= 0.7 else "FAIL")
        extra = ""
        if role == "controller":
            extra = f"  stop 占比={s['stop'] / max(s['n'], 1):.0%}"
        elif role == "verifier":
            extra = f"  「分数:」出现={s['score']}/{s['n']}"
        elif role == "critic":
            extra = f"  flag 率={s['flag'] / max(s['n'], 1):.0%}"
        acts = f"  action={s['act']}" if s["act"] else ""
        print(f"[{flag}] {role:11s} {s['ok']}/{s['n']} ({rate:.0%}){extra}{acts}")
        if rate < 0.7:
            fatal.append(f"{role} 格式可解析率 {rate:.0%} < 70%")

    # v2 机制前置条件
    print("\n" + "=" * 70)
    print("v2 机制前置条件")
    print("=" * 70)
    ctrl = stats.get("controller", {})
    if ctrl.get("stop", 0) == 0:
        print("[WARN] controller 从不输出 stop → stop_gate 下每个 episode 会跑满 "
              "max_rounds；观察 RL 首跑的 stop_rate，若恒 0 需检查")
    else:
        print(f"[ok  ] controller 会输出 stop（{ctrl['stop']}/{ctrl['n']}）")
    verif = stats.get("verifier", {})
    if verif.get("score", 0) == 0:
        fatal.append("verifier 从不输出「分数:」→ stop 闸门必然死锁")
        print("[FAIL] verifier 从不输出「分数:」→ 黑板永远无分数 → stop 闸门死锁")
    else:
        print(f"[ok  ] verifier 输出分数（{verif['score']}/{verif['n']}）")
    n_req = sum(s["act"].get("request", 0) + s["act"].get("challenge", 0)
                for s in stats.values() if s["act"])
    if n_req == 0:
        print("[WARN] 三角色均未发起 interaction（action 全 none）→ 交互率初始为 0；"
              "ε 强制注入(eps_force_init=0.3)会兜底，但 int_rate 可能偏低")
    else:
        print(f"[ok  ] 存在主动 interaction 发起（{n_req} 次）")

    print("\n" + "=" * 70)
    print("样例输出（每角色首条，截断 300 字）")
    print("=" * 70)
    for role, resp in samples.items():
        print(f"\n----- {role} -----\n{resp[:300]}")

    print("\n" + "=" * 70)
    if fatal:
        print("✗ 存在致命格式问题，不建议启动 RL：")
        for f in fatal:
            print(f"   - {f}")
        sys.exit(1)
    print("✓ 格式验收通过，可以启动 v2 RL")


if __name__ == "__main__":
    main()
