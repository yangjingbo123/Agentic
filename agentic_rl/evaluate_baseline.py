"""
Baseline 评估：用未训练的模型直接跑，不加载 checkpoint。
"""
import argparse
import json
import sys

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from llm.trainable_llm import load_trainable_models
    from llm.vllm_engine import VLLMInferenceEngine
    from agents.agentic_executor import AgenticExecutor
    from agents.parsing import parse_interaction

    model, tokenizer = load_trainable_models(config["llm"]["model_path"])

    print("Initializing vLLM engine...", flush=True)
    vllm_engine = VLLMInferenceEngine(
        config["llm"]["model_path"],
        max_tokens=config["agentic"].get("max_tokens", 512),
        gpu_memory_utilization=0.45,
    )
    print("vLLM ready.", flush=True)

    executor = AgenticExecutor(model, tokenizer, config.get("agentic", {}),
                              vllm_engine=vllm_engine, eval_mode=True)

    data_path = config["data"][f"{args.split}_path"]
    with open(data_path) as f:
        dataset = [json.loads(line) for line in f][:args.max_samples]

    print(f"Evaluating baseline on {len(dataset)} samples ({args.split})...", flush=True)

    correct = 0
    total_turns = 0
    total_interactions = 0

    for i, item in enumerate(dataset):
        ep = executor.run_episode(item["question"], item["answer"])
        if ep["is_correct"]:
            correct += 1
        total_turns += len(set(ep["turn_ids"]))
        total_interactions += sum(
            1 for msg in ep["messages"]
            if msg["role_name"] in ("proposer", "critic", "verifier")
            # 与运行时同一把尺子（baseline 的 int_rate 要能和 RL 侧对着看，
            # 两处各写一套字面量判据就没有可比性）
            and parse_interaction(msg["response"])[0] != "none"
        )
        print(f"[{i+1}/{len(dataset)}] correct={ep['is_correct']} acc={correct/(i+1):.3f}", flush=True)

    n = len(dataset)
    print(f"\n=== Baseline Results ===")
    print(f"Accuracy:         {correct/n:.4f} ({correct}/{n})")
    print(f"Avg turns:        {total_turns/n:.2f}")
    print(f"Interaction rate: {total_interactions/n:.2f}")


if __name__ == "__main__":
    main()
