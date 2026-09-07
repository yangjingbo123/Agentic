"""Evaluate non-RL and fixed-pipeline system baselines on MATH/AIME."""

import argparse
import json
import os
import sys


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["sft_cot", "self_consistency", "self_refine",
                                 "fixed_four_role"])
    parser.add_argument("--checkpoint", required=True,
                        help="SFT checkpoint used by all system baselines")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--suite", required=True, choices=["math_l5", "aime"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--vllm_use_v1", default="0")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from baselines.systems.runtime import SystemEvaluator
    from llm.trainable_llm import load_trainable_models
    from llm.vllm_engine import VLLMInferenceEngine

    dataset = load_jsonl(args.data)
    if args.suite == "math_l5":
        dataset = [x for x in dataset if x.get("level") == "Level 5"]
    if args.max_samples:
        dataset = dataset[:args.max_samples]

    model, tokenizer = load_trainable_models(
        args.model_path, sft_checkpoint=args.checkpoint)
    model._model.eval()
    visible = [x.strip() for x in os.environ.get(
        "CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    gpu = visible[1] if len(visible) > 1 else (visible[0] if visible else "1")
    engine = VLLMInferenceEngine(
        args.model_path, max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, vllm_gpu=gpu,
        vllm_use_v1=args.vllm_use_v1, enforce_eager=args.enforce_eager)
    engine.sync_lora(model)
    evaluator = SystemEvaluator(
        tokenizer, engine, args.method, n_samples=args.n_samples,
        temperature=args.temperature)

    episodes = []
    for start in range(0, len(dataset), max(1, args.batch_size)):
        part = dataset[start:start + args.batch_size]
        episodes.extend(evaluator.run_batch(
            [x["question"] for x in part], [x["answer"] for x in part]))
        print("[baseline-eval:%s] %d/%d" %
              (args.method, min(start + args.batch_size, len(dataset)), len(dataset)),
              flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item, ep in zip(dataset, episodes):
            row = {"method": args.method, "suite": args.suite,
                   "question": item["question"], "gold": item["answer"],
                   "prediction": ep["final_answer"],
                   "is_correct": bool(ep["is_correct"]),
                   "n_calls": ep["n_calls"],
                   "prompt_tokens": ep["prompt_tokens"],
                   "generated_tokens": ep["generated_tokens"]}
            for key in ("year", "exam", "problem", "level"):
                if key in item:
                    row[key] = item[key]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = max(len(episodes), 1)
    correct = sum(bool(x["is_correct"]) for x in episodes)
    summary = {
        "method": args.method, "suite": args.suite, "correct": correct,
        "total": len(episodes), "accuracy": correct / n,
        "prompt_tokens_per_problem": sum(x["prompt_tokens"] for x in episodes) / n,
        "generated_tokens_per_problem": sum(x["generated_tokens"] for x in episodes) / n,
        "llm_calls_per_problem": sum(x["n_calls"] for x in episodes) / n,
    }
    if args.suite == "aime":
        summary["by_year"] = {}
        for year in sorted({x.get("year") for x in dataset if x.get("year")}):
            idx = [i for i, item in enumerate(dataset) if item.get("year") == year]
            yc = sum(bool(episodes[i]["is_correct"]) for i in idx)
            summary["by_year"][str(year)] = {"correct": yc, "total": len(idx),
                                                   "accuracy": yc / max(len(idx), 1)}
    summary_path = args.output + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("method=%s suite=%s accuracy=%.4f correct=%d/%d" %
          (args.method, args.suite, correct / n, correct, len(episodes)))
    if "by_year" in summary:
        print("AIME years=" + ",".join(
            "%s:%d/%d" % (year, row["correct"], row["total"])
            for year, row in summary["by_year"].items()))
    print("prompt_tokens/problem=%.1f generated_tokens/problem=%.1f calls/problem=%.2f" % (
        summary["prompt_tokens_per_problem"],
        summary["generated_tokens_per_problem"],
        summary["llm_calls_per_problem"]))
    print("summary=%s" % summary_path)
    engine.close()


if __name__ == "__main__":
    main()
