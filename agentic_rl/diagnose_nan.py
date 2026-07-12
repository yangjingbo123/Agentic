"""最小化诊断：跑一次完整 episode，打印所有中间值"""
import sys
sys.path.insert(0, '/cephfs/volumes/hpc_home/k24104674/aed22256-9e0b-4f4f-86c1-c56793988876/jingbo/marl/Agentic/agentic_rl')
import torch
import torch.nn.functional as F
from llm.trainable_llm import load_trainable_models
from llm.vllm_engine import VLLMInferenceEngine
from agents.agentic_executor import AgenticExecutor

model_path = "/scratch/users/k24104674/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
model, tokenizer = load_trainable_models(model_path, sft_checkpoint="checkpoints/sft")

vllm = VLLMInferenceEngine(model_path, max_tokens=256, gpu_memory_utilization=0.55)

config = {"max_tokens": 256, "max_rounds": 1, "max_interactions": 1}
executor = AgenticExecutor(model, tokenizer, config, vllm_engine=vllm)

question = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. How much does she make from selling the rest at $2 per egg?"
answer = "18"

ep = executor.run_episode(question, answer)
print(f"is_correct={ep['is_correct']} final_answer={repr(ep['final_answer'])}")
print(f"n_messages={len(ep['messages'])} n_log_probs={len(ep['log_probs'])}")
print(f"log_probs sample: {ep['log_probs'][:5]}")
print(f"turn_ids sample: {ep['turn_ids'][:5]}")

# 模拟 _compute_loss 的前几步
device = next(p for p in model._model.parameters()).device
all_lps = torch.nan_to_num(torch.tensor(ep["log_probs"]), nan=-100.0, posinf=0.0, neginf=-100.0).to(device)
turn_ids = torch.tensor(ep["turn_ids"]).to(device)

for t_idx, msg in enumerate(ep["messages"]):
    mask = (turn_ids == t_idx)
    old_lps = all_lps[mask]
    print(f"\n--- turn {t_idx} (role={msg['role_name']}) ---")
    print(f"  old_lps: n={old_lps.shape[0]} min={old_lps.min():.3f} max={old_lps.max():.3f} has_neg100={((old_lps==-100).sum().item())}")

    if "response_ids" in msg and msg["response_ids"]:
        resp_ids = [t for t in msg["response_ids"] if 0 <= t < model._model.config.vocab_size]
        response_ids = torch.tensor(resp_ids, device=device).unsqueeze(0)
        print(f"  response_ids (from vLLM): n={len(resp_ids)}")
    else:
        print(f"  WARNING: no response_ids, using tokenizer")
        response_ids = tokenizer(msg["response"], return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": msg["system"]}, {"role": "user", "content": msg["user"]}],
        return_tensors="pt", add_generation_prompt=True, return_dict=True,
    )["input_ids"].to(device)

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    resp_start = prompt_ids.shape[1]
    n = response_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits[0, resp_start-1: resp_start+n-1]
    new_lps = torch.stack([F.log_softmax(logits[j].float(), dim=-1)[tok] for j, tok in enumerate(response_ids[0])])

    print(f"  new_lps: min={new_lps.min():.3f} max={new_lps.max():.3f} has_nan={not torch.isfinite(new_lps).all()}")

    n_align = min(n, old_lps.shape[0])
    if n_align > 0:
        avg_log_ratio = (new_lps[:n_align] - old_lps[:n_align]).mean()
        print(f"  avg_log_ratio={avg_log_ratio:.4f} finite={torch.isfinite(avg_log_ratio).item()}")
