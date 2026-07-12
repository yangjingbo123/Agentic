"""验证 _compute_loss 的新实现是否产生 finite 的 new_lps"""
import sys
sys.path.insert(0, '/cephfs/volumes/hpc_home/k24104674/aed22256-9e0b-4f4f-86c1-c56793988876/jingbo/marl/Agentic/agentic_rl')

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    from llm.trainable_llm import load_trainable_models
    from agents.agentic_executor import AgenticExecutor

    model_path = "/scratch/users/k24104674/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    model, tokenizer = load_trainable_models(model_path, sft_checkpoint="checkpoints/sft")

    class MockVLLM:
        def generate(self, role, prompt):
            if role == "controller":
                text = "<meta-plan>\nstrategy: explore\nfocus: proposer\nreason: test\n</meta-plan>"
            elif role == "proposer":
                text = "<interaction>\naction: none\ntarget: none\nreason: test\n</interaction>\n推理过程：16-3-4=9, 9*2=18\n最终答案：18"
            else:
                text = "无错误"
            ids = tokenizer.encode(text, add_special_tokens=False)
            lps = [-1.0] * len(ids)
            return text, lps, ids

    config = {"max_tokens": 256, "max_rounds": 1, "max_interactions": 1}
    executor = AgenticExecutor(model, tokenizer, config, vllm_engine=MockVLLM())

    ep = executor.run_episode(
        "Janet's ducks lay 16 eggs. She eats 3, bakes 4, sells rest at $2. How much?", "18"
    )

    device = next(p for p in model._model.parameters()).device
    vocab_size = model._model.config.vocab_size
    all_lps = torch.tensor(ep["log_probs"], dtype=torch.float32).to(device)
    turn_ids = torch.tensor(ep["turn_ids"]).to(device)

    for t_idx, msg in enumerate(ep["messages"]):
        mask = (turn_ids == t_idx)
        old_lps = all_lps[mask]
        if old_lps.shape[0] == 0:
            continue

        resp_ids_raw = msg.get("response_ids", [])
        resp_ids = [t for t in resp_ids_raw if 0 <= t < vocab_size]
        bad_ids = [t for t in resp_ids_raw if not (0 <= t < vocab_size)]
        print(f"turn {t_idx}: resp_ids_raw={len(resp_ids_raw)} valid={len(resp_ids)} bad={bad_ids[:3]}")

        if not resp_ids:
            print(f"  SKIP: no valid resp_ids")
            continue

        response_ids = torch.tensor(resp_ids, device=device).unsqueeze(0)
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": msg["system"]},
             {"role": "user",   "content": msg["user"]}],
            return_tensors="pt", add_generation_prompt=True, return_dict=True,
        )["input_ids"].to(device)

        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        resp_start = prompt_ids.shape[1]
        n = response_ids.shape[1]

        with torch.no_grad():
            logits = model(full_ids, use_cache=False).logits[0, resp_start - 1: resp_start + n - 1]

        new_lps = torch.stack([
            F.log_softmax(logits[j].float(), dim=-1)[tok]
            for j, tok in enumerate(response_ids[0])
        ])
        print(f"  new_lps: finite={torch.isfinite(new_lps).all().item()} min={new_lps.min():.3f} max={new_lps.max():.3f}")
        n_align = min(n, old_lps.shape[0])
        avg_log_ratio = (new_lps[:n_align] - old_lps[:n_align]).mean()
        ratio = torch.exp(torch.clamp(avg_log_ratio, -10, 10))
        print(f"  avg_log_ratio={avg_log_ratio:.4f} ratio={ratio:.4f}")
