import argparse
import json
from collections import defaultdict

import torch
import yaml

from agents.agentic_executor import AgenticExecutor, normalize_answer
from llm.trainable_llm import load_trainable_models
from peft import PeftModel


def load_finetuned_models(model_path: str, checkpoint_dir: str):
    """加载训练后的模型（base + LoRA checkpoint）"""
    from llm.trainable_llm import ROLE_DEVICES
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    models = {}
    for role, device in ROLE_DEVICES.items():
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        role_ckpt = f"{checkpoint_dir}/{role}"
        model = PeftModel.from_pretrained(base, role_ckpt)
        model.eval()
        models[role] = model
    return models, tokenizer


def evaluate(executor: AgenticExecutor, dataset: list) -> dict:
    correct = 0
    total_turns = 0
    total_interactions = 0

    for item in dataset:
        ep = executor.run_episode(item["question"], item["answer"])
        if ep["is_correct"]:
            correct += 1
        total_turns += len(set(ep["turn_ids"]))
        total_interactions += sum(
            1 for msg in ep["messages"]
            if msg["role_name"] in ("proposer", "critic", "verifier")
            and "<interaction>" in msg["response"]
            and "action: none" not in msg["response"]
        )

    n = len(dataset)
    return {
        "accuracy":         correct / n,
        "avg_turns":        total_turns / n,
        "interaction_rate": total_interactions / n,
        "n":                n,
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

    models, tokenizer = load_finetuned_models(
        config["llm"]["model_path"], args.checkpoint
    )
    executor = AgenticExecutor(models, tokenizer, config.get("agentic", {}))

    print(f"Evaluating on {len(dataset)} samples ({args.split})...")
    results = evaluate(executor, dataset)

    print(f"Accuracy:         {results['accuracy']:.4f} ({int(results['accuracy']*results['n'])}/{results['n']})")
    print(f"Avg turns:        {results['avg_turns']:.2f}")
    print(f"Interaction rate: {results['interaction_rate']:.2f}")


if __name__ == "__main__":
    main()
