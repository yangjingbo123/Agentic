import argparse
import json
import random
import yaml


def load_dataset(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="agentic")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--sft-checkpoint", default="checkpoints/sft", help="SFT checkpoint to load")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.stage == "agentic":
        from llm.trainable_llm import load_trainable_models
        from training.grpo_trainer import GRPOAgenticTrainer

        models, ref_models, tokenizer = load_trainable_models(
            config["llm"]["model_path"],
            sft_checkpoint=args.sft_checkpoint if args.sft_checkpoint else None,
        )

        vllm_engine = None
        if not args.no_vllm:
            from llm.vllm_engine import VLLMInferenceEngine
            print("Initializing vLLM engine...", flush=True)
            import torch
            n_gpus = torch.cuda.device_count()
            # 多卡时 vLLM 用 tensor parallelism，显存利用率可以更高
            vllm_mem = 0.85 if n_gpus >= 4 else 0.35
            vllm_engine = VLLMInferenceEngine(
                config["llm"]["model_path"],
                max_tokens=config["agentic"].get("max_tokens", 512),
                gpu_memory_utilization=vllm_mem,
            )
            print("vLLM ready.", flush=True)

        trainer = GRPOAgenticTrainer(models, ref_models, tokenizer, config.get("agentic", {}), vllm_engine=vllm_engine)

        agentic_cfg = config.get("agentic", {})
        dataset = load_dataset(config["data"]["train_path"])
        batch_size = agentic_cfg.get("batch_size", len(dataset))
        epochs = agentic_cfg.get("epochs", 3)

        print(f"Dataset: {len(dataset)} items, batch_size={batch_size}, epochs={epochs}", flush=True)

        step = 0
        for epoch in range(epochs):
            batch = random.sample(dataset, min(batch_size, len(dataset)))
            for item in batch:
                try:
                    stats = trainer.train_step(item["question"], item["answer"])
                except Exception:
                    import traceback; traceback.print_exc()
                    raise
                step += 1
                print(
                    f"epoch={epoch} step={step} "
                    f"reward={stats['mean_reward']:.3f} acc={stats['accuracy']:.2f} "
                    f"loss={stats['loss']:.4f} kl={stats['kl']:.4f}",
                    flush=True,
                )

        import os
        os.makedirs("checkpoints", exist_ok=True)
        for role in ["controller", "proposer", "critic", "verifier"]:
            models[role].save_pretrained(f"checkpoints/{role}", role=role)

    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()

