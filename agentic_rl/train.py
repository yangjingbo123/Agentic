import argparse
import json
import yaml


def load_dataset(path: str):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="agentic")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.stage == "agentic":
        from llm.trainable_llm import load_trainable_models
        from training.grpo_trainer import GRPOAgenticTrainer

        models, ref_models, tokenizer = load_trainable_models(config["llm"]["model_path"])
        trainer = GRPOAgenticTrainer(models, ref_models, tokenizer, config.get("agentic", {}))

        dataset = list(load_dataset(config["data"]["train_path"]))
        for epoch in range(config["agentic"].get("epochs", 3)):
            for item in dataset:
                stats = trainer.train_step(item["question"], item["answer"])
                print(
                    f"epoch={epoch} "
                    f"reward={stats['mean_reward']:.3f} "
                    f"acc={stats['accuracy']:.2f} "
                    f"loss={stats['loss']:.4f} "
                    f"kl={stats['kl']:.4f} "
                    f"kl_coef={stats['kl_coef']:.4f}"
                )

        for role, model in models.items():
            model.save_pretrained(f"checkpoints/{role}")

    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
