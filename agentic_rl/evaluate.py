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


def main():
    # yaml / vLLM 都只在真跑评测时才需要；留在模块顶层会让这个文件在无重依赖的
    # 环境里 import 不进来，evaluate() 的指标逻辑也就没法进纯 CPU 测试。
    import yaml

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
    # vLLM 引擎必须真起一个：`run_episodes_batch` 第一行就 `raise RuntimeError`
    # （no_vllm 路径在 v2 已移除），此前这里传 None，脚本跑第一条就崩——也就是
    # 说这个离线评测入口整个是死的。起法与 evaluate_baseline.py 保持一致。
    from llm.vllm_engine import VLLMInferenceEngine
    print("Initializing vLLM engine...", flush=True)
    vllm_engine = VLLMInferenceEngine(
        config["llm"]["model_path"],
        max_tokens=config["agentic"].get("max_tokens", 512),
        gpu_memory_utilization=0.45,
    )
    print("vLLM ready.", flush=True)
    executor = AgenticExecutor(model, tokenizer, config.get("agentic", {}),
                               vllm_engine=vllm_engine, eval_mode=True)

    print(f"Evaluating on {len(dataset)} samples ({args.split})...")
    results = evaluate(executor, dataset)

    print(f"Accuracy:             {results['accuracy']:.4f} ({int(results['accuracy']*results['n'])}/{results['n']})")
    print(f"Avg turns:            {results['avg_turns']:.2f}")
    print(f"Interaction rate:     {results['interaction_rate']:.2f}")
    print(f"Proposer accuracy:    {results['proposer_accuracy']:.4f}")
    print(f"Critic precision:     {results['critic_precision']:.4f}  (flag rate: {results['critic_flag_rate']:.4f})")
    print(f"Verifier consistency: {results['verifier_consistency']:.4f}"
          f"  (未给分 {results['verifier_unparsed']} turn，未计入分母)")


if __name__ == "__main__":
    main()
