import argparse
import json
import re

from agents.agentic_executor import AgenticExecutor, math_equal
from agents.parsing import parse_interaction, parse_reasoning, parse_score


def load_finetuned_models(model_path: str, checkpoint_dir: str):
    """加载训练后的模型。
    - SFT checkpoint（平铺结构）：直接作为 sft_checkpoint 加载
    - RL checkpoint（含 proposer/critic/verifier/controller 子目录）：逐 adapter 加载
    """
    import os, re
    import safetensors.torch as st
    from llm.trainable_llm import ROLE_ADAPTER, load_trainable_models

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
            loaded = 0
            for name, param in model._model.named_parameters():
                if "lora_" not in name:
                    continue
                src = re.sub(rf"\.{adapter_name}\.", ".", name)
                if src in weights:
                    param.data.copy_(weights[src].to(param.device))
                    loaded += 1
            print(f"  [eval] {adapter_name}: loaded {loaded} params", flush=True)
        print(f"Loaded RL checkpoint from {checkpoint_dir}")

    return model, tokenizer


def evaluate(executor: AgenticExecutor, dataset: list) -> dict:
    """离线评测。**三把尺子一律复用运行时解析器**（`agents.parsing`）。

    这里曾经自己手写 `最终答案：(.+)` / `分数:\\s*([0-9.]+)` / 字面量
    `"action: none" not in resp` 三条判据，与 RL 侧各自漂移。实测 2864 个
    SFT turn 上的差异：verifier 分数 0/748、交互判定 0/1982 一致，但
    **proposer 答案 3/580 不一致**——模型写半角 `最终答案: 24` 时旧尺子抓不到，
    `last_proposer_ans` 变空串。后果不止 proposer_accuracy 少算 3 次：空串使
    `math_equal("", gold)` 恒假，于是这些 turn 上 critic 的每一次挑错都被记成
    真阳性（critic_precision 虚高），一个解析缺口污染两个指标。
    """
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
    verifier_unparsed = 0  # 没给出可解析分数的 turn（不进一致率分母）

    for item in dataset:
        ep = executor.run_episode(item["question"], item["answer"])
        correct_answer = item["answer"]

        if ep["is_correct"]:
            correct += 1
        total_turns += len(set(ep["turn_ids"]))
        total_interactions += sum(
            1 for msg in ep["messages"]
            if msg["role_name"] in ("proposer", "critic", "verifier")
            and parse_interaction(msg["response"])[0] != "none"
        )

        # 逐 turn 分析各角色，追踪最近一次 proposer 答案
        last_proposer_ans = ""
        for msg in ep["messages"]:
            role = msg["role_name"]
            resp = msg["response"]

            if role == "proposer":
                proposer_total += 1
                last_proposer_ans = parse_reasoning(resp)[1]
                if math_equal(last_proposer_ans, correct_answer):
                    proposer_correct += 1

            elif role == "critic":
                critic_total += 1
                if executor._critic_found_errors(resp):
                    if math_equal(last_proposer_ans, correct_answer):
                        critic_fp += 1  # 误报：proposer 对但 critic 挑错
                    else:
                        critic_tp += 1  # 正确挑错

            elif role == "verifier":
                # 解析失败不按 0.5 兜底：0.5 落在 `score > 0.5` 的判据边界上，
                # 会把「没打分」系统性记成「打了低分」，一致率于是变成
                # 「答错率」的函数而不是 verifier 能力的度量。
                score = parse_score(resp)
                if score is None:
                    verifier_unparsed += 1
                    continue
                verifier_total += 1
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
        "verifier_unparsed":     verifier_unparsed,
        "n":                     n,
    }


def per_item_rows(dataset: list, episodes: list, suite: str) -> list:
    """生成可复查的逐题结果；不保存完整 prompt/trajectory，控制输出体积。"""
    rows = []
    for item, ep in zip(dataset, episodes):
        row = {
            "suite": suite,
            "question": item["question"],
            "gold": item["answer"],
            "prediction": ep.get("final_answer", ""),
            "is_correct": bool(ep.get("is_correct", False)),
            "n_turns": len(ep.get("raca_turn_data", {})),
            "initial_answer": ep.get("initial_answer", ""),
            "initial_is_correct": bool(ep.get("initial_is_correct", False)),
            "n_rescues": int(ep.get("n_rescues", 0)),
            "n_harms": int(ep.get("n_harms", 0)),
            "n_corrections_verified": int(ep.get("n_corrections_verified", 0)),
            "n_corrections_accepted": int(ep.get("n_corrections_accepted", 0)),
            "n_corrections_rejected": int(ep.get("n_corrections_rejected", 0)),
            "auto_stopped": bool(ep.get("auto_stopped", False)),
            "role_calls": ep.get("role_calls", {}),
            "generated_tokens": int(ep.get("generated_tokens", 0)),
        }
        for key in ("year", "exam", "problem", "level", "source"):
            if key in item:
                row[key] = item[key]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None,
                        help="直接指定 JSONL；优先于 --suite")
    parser.add_argument("--suite", default="math_test",
                        choices=["math_train", "math_test", "math_l3", "math_l4",
                                 "math_l5", "aime"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="每次送入 executor 的题目数，避免全量评测 RPC 过大")
    parser.add_argument("--output", default=None,
                        help="逐题结果 JSONL；默认 results/<suite>.jsonl")
    args = parser.parse_args()

    # 重依赖只在真跑评测时导入；--help 不要求本地已安装运行时依赖。
    import os
    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)
    # configs/config.yaml 是 Hydra defaults，需要补读实际子配置；旧的完整 YAML 仍兼容。
    if "llm" not in config or "agentic" not in config or "data" not in config:
        with open("configs/llm/qwen3_8b.yaml") as f:
            config["llm"] = yaml.safe_load(f)
        with open("configs/agentic/default.yaml") as f:
            config["agentic"] = yaml.safe_load(f)
        with open("configs/data/math.yaml") as f:
            config["data"] = yaml.safe_load(f)

    if args.data:
        data_path = args.data
    elif args.suite == "math_train":
        data_path = config["data"]["train_path"]
    elif args.suite == "aime":
        data_path = config["data"]["aime_path"]
    else:
        data_path = config["data"]["test_path"]

    with open(data_path) as f:
        dataset = [json.loads(line) for line in f]
    level_map = {"math_l3": "Level 3", "math_l4": "Level 4", "math_l5": "Level 5"}
    if args.suite in level_map:
        dataset = [item for item in dataset if item.get("level") == level_map[args.suite]]
    if args.max_samples:
        dataset = dataset[:args.max_samples]

    model, tokenizer = load_finetuned_models(
        config["llm"]["model_path"], args.checkpoint
    )
    from llm.vllm_engine import VLLMInferenceEngine
    visible = [x.strip() for x in os.environ.get(
        "CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    slot = int(config["agentic"].get("vllm_gpu_slot", 1))
    if visible and slot >= len(visible):
        raise ValueError(
            f"vllm_gpu_slot={slot}, but CUDA_VISIBLE_DEVICES has {len(visible)} entries")
    vllm_gpu = visible[slot] if visible else str(slot)
    print("Initializing vLLM engine...", flush=True)
    vllm_engine = VLLMInferenceEngine(
        config["llm"]["model_path"],
        max_tokens=config["agentic"].get("max_tokens", 512),
        gpu_memory_utilization=config["agentic"].get(
            "vllm_gpu_memory_utilization", 0.45),
        max_model_len=config["agentic"].get("vllm_max_model_len", 4096),
        vllm_use_v1=str(config["agentic"].get("vllm_use_v1", "0")),
        enforce_eager=bool(config["agentic"].get("vllm_enforce_eager", True)),
        vllm_gpu=vllm_gpu,
    )
    vllm_engine.sync_lora(model)
    print("vLLM ready.", flush=True)
    executor = AgenticExecutor(model, tokenizer, config.get("agentic", {}),
                               vllm_engine=vllm_engine, eval_mode=True)

    print(f"Evaluating {args.suite} on {len(dataset)} samples from {data_path}...", flush=True)
    episodes = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(dataset), batch_size):
        part = dataset[start:start + batch_size]
        episodes.extend(executor.run_episodes_batch(
            [item["question"] for item in part],
            [item["answer"] for item in part],
        ))
        print(f"  [eval] {min(start + batch_size, len(dataset))}/{len(dataset)}",
              flush=True)

    class ReplayExecutor:
        def __init__(self, values, source):
            self._values = iter(values)
            self._critic_found_errors = source._critic_found_errors
        def run_episode(self, _q, _a):
            return next(self._values)

    results = evaluate(ReplayExecutor(episodes, executor), dataset)
    output = args.output or os.path.join("results", f"{args.suite}.jsonl")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in per_item_rows(dataset, episodes, args.suite):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Accuracy:             {results['accuracy']:.4f} "
          f"({int(results['accuracy']*results['n'])}/{results['n']})")
    print(f"Avg turns:            {results['avg_turns']:.2f}")
    print(f"Interaction rate:     {results['interaction_rate']:.2f}")
    print(f"Proposer accuracy:    {results['proposer_accuracy']:.4f}")
    print(f"Critic precision:     {results['critic_precision']:.4f}  "
          f"(flag rate: {results['critic_flag_rate']:.4f})")
    print(f"Verifier consistency: {results['verifier_consistency']:.4f}"
          f"  (未给分 {results['verifier_unparsed']} turn，未计入分母)")
    if args.suite == "aime":
        for year in sorted({item.get("year") for item in dataset if item.get("year")}):
            pairs = [(item, ep) for item, ep in zip(dataset, episodes)
                     if item.get("year") == year]
            yc = sum(bool(ep.get("is_correct")) for _item, ep in pairs)
            print(f"AIME {year}:            {yc}/{len(pairs)} "
                  f"({yc/max(len(pairs), 1):.4f})")
    print(f"Predictions:          {output}")


if __name__ == "__main__":
    main()
