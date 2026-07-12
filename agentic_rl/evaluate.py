import argparse
import json
import re

import torch
import yaml

from agents.agentic_executor import AgenticExecutor, normalize_answer
from llm.trainable_llm import load_trainable_models
from peft import PeftModel


def load_finetuned_models(model_path: str, checkpoint_dir: str):
    """加载训练后的模型。
    - SFT checkpoint（平铺结构）：直接作为 sft_checkpoint 加载
    - RL checkpoint（含 proposer/critic/verifier/controller 子目录）：逐 adapter 加载
    """
    import os, re
    import safetensors.torch as st
    from llm.trainable_llm import ROLE_ADAPTER

    is_rl = os.path.isdir(os.path.join(checkpoint_dir, "proposer"))
    sft_ckpt = None if is_rl else checkpoint_dir

    model, tokenizer = load_trainable_models(model_path, sft_checkpoint=sft_ckpt)
    model._model.eval()

    if is_rl:
        for adapter_name in ROLE_ADAPTER.values():
            ckpt = os.path.join(checkpoint_dir, adapter_name, "adapter_model.safetensors")
            if not os.path.exists(ckpt):
                continue
            weights = st.load_file(ckpt)
            model._model.set_adapter(adapter_name)
            for name, param in model._model.named_parameters():
                if not param.requires_grad:
                    continue
                src = re.sub(rf"\.{adapter_name}\.", ".", name)
                if src in weights:
                    param.data.copy_(weights[src].to(param.device))
        print(f"Loaded RL checkpoint from {checkpoint_dir}")

    return model, tokenizer


def evaluate(executor: AgenticExecutor, dataset: list) -> dict:
    correct = 0
    total_turns = 0
    total_interactions = 0

    # 分角色指标
    proposer_correct = 0
    proposer_total = 0
    critic_tp = 0   # 挑错且 proposer 确实错了
    critic_fp = 0   # 挑错但 proposer 是对的
    critic_total = 0
    verifier_agree = 0  # verifier 判断与最终答案一致
    verifier_total = 0

    for item in dataset:
        ep = executor.run_episode(item["question"], item["answer"])
        correct_answer = normalize_answer(item["answer"])

        if ep["is_correct"]:
            correct += 1
        total_turns += len(set(ep["turn_ids"]))
        total_interactions += sum(
            1 for msg in ep["messages"]
            if msg["role_name"] in ("proposer", "critic", "verifier")
            and "<interaction>" in msg["response"]
            and "action: none" not in msg["response"]
        )

        # 逐 turn 分析各角色，追踪最近一次 proposer 答案
        last_proposer_ans = ""
        for msg in ep["messages"]:
            role = msg["role_name"]
            resp = msg["response"]

            if role == "proposer":
                proposer_total += 1
                m = re.search(r"最终答案：(.+)", resp)
                last_proposer_ans = normalize_answer(m.group(1).strip()) if m else ""
                if last_proposer_ans == correct_answer:
                    proposer_correct += 1

            elif role == "critic":
                critic_total += 1
                if "无错误" not in resp:
                    if last_proposer_ans == correct_answer:
                        critic_fp += 1  # 误报：proposer 对但 critic 挑错
                    else:
                        critic_tp += 1  # 正确挑错

            elif role == "verifier":
                verifier_total += 1
                m = re.search(r"分数:\s*([0-9.]+)", resp)
                score = float(m.group(1)) if m else 0.5
                verifier_agree += 1 if (score > 0.5) == ep["is_correct"] else 0

    n = len(dataset)
    return {
        "accuracy":              correct / n,
        "avg_turns":             total_turns / n,
        "interaction_rate":      total_interactions / n,
        "proposer_accuracy":     proposer_correct / proposer_total if proposer_total else 0,
        "critic_precision":      critic_tp / (critic_tp + critic_fp) if (critic_tp + critic_fp) else 0,
        "critic_flag_rate":      (critic_tp + critic_fp) / critic_total if critic_total else 0,
        "verifier_consistency":  verifier_agree / verifier_total if verifier_total else 0,
        "n":                     n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints")
    parser.add_argument("--split",      default="test", choices=["train", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = config["data"][f"{args.split}_path"]
    with open(data_path) as f:
        dataset = [json.loads(line) for line in f]
    if args.max_samples:
        dataset = dataset[:args.max_samples]

    model, tokenizer = load_finetuned_models(
        config["llm"]["model_path"], args.checkpoint
    )
    executor = AgenticExecutor(model, tokenizer, config.get("agentic", {}))

    print(f"Evaluating on {len(dataset)} samples ({args.split})...")
    results = evaluate(executor, dataset)

    print(f"Accuracy:             {results['accuracy']:.4f} ({int(results['accuracy']*results['n'])}/{results['n']})")
    print(f"Avg turns:            {results['avg_turns']:.2f}")
    print(f"Interaction rate:     {results['interaction_rate']:.2f}")
    print(f"Proposer accuracy:    {results['proposer_accuracy']:.4f}")
    print(f"Critic precision:     {results['critic_precision']:.4f}  (flag rate: {results['critic_flag_rate']:.4f})")
    print(f"Verifier consistency: {results['verifier_consistency']:.4f}")


if __name__ == "__main__":
    main()
