"""Sample training-shaped groups from an existing checkpoint, without training.

These are newly sampled checkpoint diagnostics, never historical training data.
"""
import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", help="Training/validation JSONL, never defaults to test data")
    parser.add_argument("--output", required=True, help="Persistent output directory")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--questions_per_batch", type=int, default=None)
    parser.add_argument("--samples_per_question", type=int, default=None)
    parser.add_argument("--eps_force", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Question selection and executor injection RNG")
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")

    import os
    import yaml
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    defaults_root = config_path.parent
    for name, fallback in (("agentic", "default"), ("llm", "qwen3_8b"), ("data", "math")):
        if name not in config:
            selected = next((d[name] for d in config.get("defaults", [])
                             if isinstance(d, dict) and name in d), fallback)
            config[name] = yaml.safe_load((defaults_root / name / f"{selected}.yaml").read_text())
    config["agentic"]["rebuttal_diagnostics"] = True
    count = (args.questions_per_batch if args.questions_per_batch is not None
             else int(config["agentic"].get("batch_size", 32)))
    samples = (args.samples_per_question if args.samples_per_question is not None
               else int(config["agentic"].get("n_samples", 16)))
    if count < 1 or samples < 2:
        parser.error("Need positive questions_per_batch and at least 2 samples_per_question")
    eps = (args.eps_force if args.eps_force is not None
           else float(config["agentic"].get("eps_force_min", 0.25)))
    if not 0 <= eps <= 1:
        parser.error("--eps_force must be between 0 and 1")
    path = args.data or config["data"]["train_path"]
    dataset = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if count > len(dataset):
        parser.error("questions_per_batch exceeds dataset size")
    from evaluate import load_finetuned_models
    from agents.agentic_executor import AgenticExecutor
    from llm.vllm_engine import VLLMInferenceEngine
    from training.raca_adv import compute_raca_advantages
    from training.rebuttal_diagnostics import DiagnosticWriter, credit_report
    model, tokenizer = load_finetuned_models(config["llm"]["model_path"], args.checkpoint)
    visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    slot = int(config["agentic"].get("vllm_gpu_slot", 1))
    if visible and slot >= len(visible):
        parser.error("vllm_gpu_slot exceeds CUDA_VISIBLE_DEVICES")
    cfg = config["agentic"]
    engine = VLLMInferenceEngine(config["llm"]["model_path"],
        max_tokens=cfg.get("max_tokens", 512),
        gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.45),
        max_model_len=cfg.get("vllm_max_model_len", 4096),
        vllm_use_v1=str(cfg.get("vllm_use_v1", "0")),
        enforce_eager=bool(cfg.get("vllm_enforce_eager", True)),
        vllm_gpu=visible[slot] if visible else str(slot))
    engine.sync_lora(model)
    executor = AgenticExecutor(model, tokenizer, cfg, vllm_engine=engine, eval_mode=False)
    executor._rng.seed(args.seed)
    rng = random.Random(args.seed)
    config["probe"] = vars(args)
    writer = DiagnosticWriter(args.output, config, run_name="checkpoint_probe")
    for batch in range(args.batches):
        items = rng.sample(dataset, count)
        episodes = executor.run_episodes_batch(
            [x["question"] for x in items] * samples,
            [x["answer"] for x in items] * samples, eps_force=eps)
        groups, advantages, anchors = [], [], []
        for qi in range(count):
            group = [episodes[qi + count * s] for s in range(samples)]
            rows = []
            adv = compute_raca_advantages([e["raca_turn_data"] for e in group],
                delta=cfg.get("raca_delta", 1e-4),
                balance_interaction_layers=cfg.get("balance_interaction_layers", False),
                interaction_balance_cap=cfg.get("interaction_balance_cap", 2.0),
                diagnostics=rows)
            for row in rows:
                row["question_index"] = qi
            groups.append(group)
            advantages.append(adv)
            anchors.extend(rows)
        report = credit_report(groups, advantages, anchors, model._model.config.vocab_size)
        writer.write("checkpoint_probe", batch, episodes, credit=report,
            checkpoint=args.checkpoint, temperature=1.0, eps_force=eps,
            description="New sampled diagnostics; no optimizer or parameter update")


if __name__ == "__main__":
    main()
