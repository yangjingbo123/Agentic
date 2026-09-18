"""Evaluate system-level baselines with one reusable vLLM engine.

``--suite all`` evaluates MATH-L5 and AIME sequentially in the same process.
Completed per-item JSONL files are reused, so a preempted/failed job resumes at
the missing suite instead of regenerating completed predictions.
"""

import argparse
import gc
import json
import os
import sys


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def complete_rows(path, expected, suite=None):
    """Return validated rows when a suite output is complete, else ``None``."""
    if not path or not os.path.isfile(path):
        return None
    try:
        rows = load_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if len(rows) != expected:
        return None
    required = ("question", "gold", "prediction", "is_correct")
    if any(any(key not in row for key in required) for row in rows):
        return None
    if suite and any(row.get("suite") != suite for row in rows):
        return None
    return rows


def summary_from_rows(rows, method, suite):
    n = max(len(rows), 1)
    summary = {
        "method": method,
        "suite": suite,
        "correct": sum(bool(x.get("is_correct")) for x in rows),
        "total": len(rows),
    }
    summary["accuracy"] = summary["correct"] / n
    for src, dst in (("prompt_tokens", "prompt_tokens_per_problem"),
                     ("generated_tokens", "generated_tokens_per_problem"),
                     ("n_calls", "llm_calls_per_problem")):
        values = [float(row[src]) for row in rows if src in row]
        if values:
            summary[dst] = sum(values) / len(values)
    if any("initial_is_correct" in row for row in rows):
        values = [bool(row.get("initial_is_correct")) for row in rows]
        summary["initial_accuracy"] = sum(values) / len(values)
        summary["accuracy_gain_over_initial"] = (
            summary["accuracy"] - summary["initial_accuracy"])
    if any("candidate_correctness" in row for row in rows):
        candidates = [bool(value) for row in rows
                      for value in row.get("candidate_correctness", [])]
        if candidates:
            summary["proposal_accuracy"] = sum(candidates) / len(candidates)
            summary["total_proposals"] = len(candidates)
        for src, dst in (
                ("n_distinct_answers", "distinct_answers_per_problem"),
                ("winning_votes", "winning_votes_per_problem"),
                ("vote_margin", "vote_margin_mean"),
                ("n_tie_break_proposals", "tie_break_calls_per_problem")):
            values = [float(row[src]) for row in rows if src in row]
            if values:
                summary[dst] = sum(values) / len(values)
        summary["tie_break_problem_rate"] = sum(
            1 for row in rows if row.get("n_tie_break_proposals", 0) > 0) / n
        summary["unresolved_tie_rate"] = sum(
            1 for row in rows if row.get("unresolved_tie")) / n
    if suite == "aime":
        summary["by_year"] = {}
        for year in sorted({row.get("year") for row in rows if row.get("year")}):
            part = [row for row in rows if row.get("year") == year]
            yc = sum(bool(row.get("is_correct")) for row in part)
            summary["by_year"][str(year)] = {
                "correct": yc, "total": len(part),
                "accuracy": yc / max(len(part), 1),
            }
    return summary


def write_summary(path, summary):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def select_dataset(path, suite, limit):
    dataset = load_jsonl(path)
    if suite == "math_l5":
        dataset = [item for item in dataset if item.get("level") == "Level 5"]
    if limit:
        dataset = dataset[:limit]
    return dataset


def evaluate_suite(evaluator, dataset, method, suite, output, batch_size):
    episodes = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(dataset), batch_size):
        part = dataset[start:start + batch_size]
        episodes.extend(evaluator.run_batch(
            [item["question"] for item in part],
            [item["answer"] for item in part],
        ))
        print("[baseline-eval:%s:%s] %d/%d" % (
            method, suite, min(start + batch_size, len(dataset)), len(dataset)),
            flush=True)

    rows = []
    for item, episode in zip(dataset, episodes):
        row = {
            "method": method, "suite": suite,
            "question": item["question"], "gold": item["answer"],
            "prediction": episode["final_answer"],
            "is_correct": bool(episode["is_correct"]),
            "n_calls": episode["n_calls"],
            "prompt_tokens": episode["prompt_tokens"],
            "generated_tokens": episode["generated_tokens"],
        }
        for key in (
                "initial_answer", "initial_is_correct", "candidate_answers",
                "candidate_correctness", "proposal_phases",
                "n_independent_proposals", "n_interactive_proposals",
                "n_tie_break_proposals", "n_distinct_answers",
                "winning_votes", "valid_votes", "vote_margin",
                "vote_classes", "unresolved_tie"):
            if key in episode:
                row[key] = episode[key]
        for key in ("year", "exam", "problem", "level"):
            if key in item:
                row[key] = item[key]
        rows.append(row)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, output)
    summary = summary_from_rows(rows, method, suite)
    write_summary(output + ".summary.json", summary)
    print("method=%s suite=%s accuracy=%.4f correct=%d/%d" % (
        method, suite, summary["accuracy"], summary["correct"], summary["total"]))
    print("prompt_tokens/problem=%.1f generated_tokens/problem=%.1f calls/problem=%.2f" % (
        summary.get("prompt_tokens_per_problem", 0.0),
        summary.get("generated_tokens_per_problem", 0.0),
        summary.get("llm_calls_per_problem", 0.0)))
    if "proposal_accuracy" in summary:
        print("initial_acc=%.4f proposal_acc=%.4f gain=%+.4f "
              "distinct/problem=%.2f tie_break_rate=%.3f unresolved_tie=%.3f" % (
                  summary.get("initial_accuracy", 0.0),
                  summary["proposal_accuracy"],
                  summary.get("accuracy_gain_over_initial", 0.0),
                  summary.get("distinct_answers_per_problem", 0.0),
                  summary.get("tie_break_problem_rate", 0.0),
                  summary.get("unresolved_tie_rate", 0.0)))
    if "by_year" in summary:
        print("AIME years=" + ",".join(
            "%s:%d/%d" % (year, value["correct"], value["total"])
            for year, value in summary["by_year"].items()))
    del episodes
    gc.collect()
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["sft_cot", "self_consistency", "self_refine",
                                 "fixed_four_role",
                                 "iterative_proposal_voting"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--suite", default="all",
                        choices=["math_l5", "aime", "all"])
    # Legacy single-suite interface.
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    # Combined/resumable interface.
    parser.add_argument("--math_data", default="data/math_test.jsonl")
    parser.add_argument("--aime_data", default="data/aime_2022_2026.jsonl")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--math_samples", type=int, default=1000)
    parser.add_argument("--aime_samples", type=int, default=150)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--independent_proposals", type=int, default=3)
    parser.add_argument("--interactive_proposals", type=int, default=2)
    parser.add_argument("--max_tie_break_proposals", type=int, default=2)
    parser.add_argument("--proposal_context_chars", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--vllm_use_v1", default="0")
    parser.add_argument("--enforce_eager", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    if args.suite == "all":
        if not args.output_dir:
            raise SystemExit("--output_dir is required for --suite all")
        specs = [
            ("math_l5", args.math_data, args.math_samples,
             os.path.join(args.output_dir, "math_l5.jsonl")),
            ("aime", args.aime_data, args.aime_samples,
             os.path.join(args.output_dir, "aime.jsonl")),
        ]
    else:
        if not args.data or not args.output:
            raise SystemExit("--data and --output are required for a single suite")
        specs = [(args.suite, args.data, args.max_samples, args.output)]

    os.makedirs(args.output_dir or os.path.dirname(specs[0][3]) or ".", exist_ok=True)
    summaries = {}
    pending = []
    for suite, data_path, limit, output in specs:
        dataset = select_dataset(data_path, suite, limit)
        cached = None if args.no_resume else complete_rows(output, len(dataset), suite)
        if cached is not None:
            print("[baseline-eval:%s:%s] reuse complete output: %s" % (
                args.method, suite, output), flush=True)
            summary = summary_from_rows(cached, args.method, suite)
            write_summary(output + ".summary.json", summary)
            summaries[suite] = summary
        else:
            pending.append((suite, dataset, output))

    engine = None
    if pending:
        from baselines.systems.runtime import SystemEvaluator
        from llm.trainable_llm import load_trainable_models
        from llm.vllm_engine import VLLMInferenceEngine

        model, tokenizer = load_trainable_models(
            args.model_path, sft_checkpoint=args.checkpoint)
        model._model.eval()
        visible = [item.strip() for item in os.environ.get(
            "CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        gpu = visible[1] if len(visible) > 1 else (visible[0] if visible else "1")
        try:
            engine = VLLMInferenceEngine(
                args.model_path, max_tokens=args.max_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len, vllm_gpu=gpu,
                vllm_use_v1=args.vllm_use_v1,
                enforce_eager=args.enforce_eager)
            engine.sync_lora(model)
            evaluator = SystemEvaluator(
                tokenizer, engine, args.method, n_samples=args.n_samples,
                temperature=args.temperature,
                independent_proposals=args.independent_proposals,
                interactive_proposals=args.interactive_proposals,
                max_tie_break_proposals=args.max_tie_break_proposals,
                proposal_context_chars=args.proposal_context_chars)
            for suite, dataset, output in pending:
                summaries[suite] = evaluate_suite(
                    evaluator, dataset, args.method, suite, output,
                    args.batch_size)
        finally:
            if engine is not None:
                engine.close()

    if args.suite == "all":
        combined = [summaries[name] for name in ("math_l5", "aime")]
        combined_path = os.path.join(args.output_dir, "combined_summary.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print("combined_summary=%s" % combined_path)


if __name__ == "__main__":
    main()
