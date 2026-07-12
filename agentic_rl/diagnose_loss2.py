"""诊断 _compute_loss 里哪一步导致 n_valid=0"""
import sys
sys.path.insert(0, ".")

import torch
from llm.trainable_llm import load_trainable_models, ROLE_ADAPTER
from llm.vllm_engine import VLLMInferenceEngine
from agents.agentic_executor import AgenticExecutor
from training.grpo_trainer import logprobs_from_logits

MODEL_PATH = "/scratch/users/k24104674/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
SFT_CKPT   = "checkpoints/sft"

if __name__ == "__main__":
    print("Loading model...", flush=True)
    model, tokenizer = load_trainable_models(MODEL_PATH, sft_checkpoint=SFT_CKPT)
    vllm = VLLMInferenceEngine(MODEL_PATH, max_tokens=128, gpu_memory_utilization=0.85)

    cfg = {"max_tokens": 128, "max_rounds": 1, "max_interactions": 0}
    executor = AgenticExecutor(model, tokenizer, cfg, vllm_engine=vllm)

    print("Running one episode...", flush=True)
    ep = executor.run_episode("What is 2+2?", "4")
    print(f"  messages={len(ep['messages'])} log_probs={len(ep['log_probs'])} "
          f"turn_ids={len(ep['turn_ids'])} is_correct={ep['is_correct']}", flush=True)

    device     = next(p for p in model._model.parameters()).device
    vocab_size = model._model.config.vocab_size
    all_old_lps  = torch.tensor(ep["log_probs"], dtype=torch.float32, device=device)
    all_old_lps  = torch.nan_to_num(all_old_lps, nan=0.0, posinf=0.0, neginf=0.0)
    all_turn_ids = torch.tensor(ep["turn_ids"], device=device)

    print(f"\nturn_ids unique: {all_turn_ids.unique().tolist()}", flush=True)

    for i, msg in enumerate(ep["messages"]):
        turn_id  = msg.get("turn_id", i)
        role     = msg.get("role_name", "?")
        resp_ids = [t for t in msg.get("response_ids", []) if 0 <= t < vocab_size]
        mask     = (all_turn_ids == turn_id)
        old_lps  = all_old_lps[mask]
        n_resp   = len(resp_ids)
        n_align  = min(n_resp, old_lps.shape[0])

        print(f"\n[turn {i}] role={role} turn_id={turn_id} "
              f"old_lps={old_lps.shape[0]} resp_ids={n_resp} n_align={n_align}", flush=True)

        if n_align == 0:
            print("  -> SKIP: n_align=0", flush=True); continue

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": msg["system"]},
             {"role": "user",   "content": msg["user"]}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        p_len = len(prompt_ids)
        if p_len == 0:
            print("  -> SKIP: p_len=0", flush=True); continue

        input_ids   = torch.tensor(prompt_ids + resp_ids, dtype=torch.long, device=device).unsqueeze(0)
        resp_labels = torch.tensor(resp_ids, dtype=torch.long, device=device)

        model._model.set_adapter(ROLE_ADAPTER.get(role, "proposer"))
        logits_new    = model._model(input_ids, use_cache=False).logits[0].float()
        logits_device = logits_new.device
        print(f"  input_device={device} logits_device={logits_device}", flush=True)
        new_lps = logprobs_from_logits(logits_new[p_len-1:p_len+n_resp-1], resp_labels.to(logits_device))
        del logits_new
        print(f"  new_lps: shape={new_lps.shape} finite={torch.isfinite(new_lps).all()} "
              f"mean={new_lps.mean():.3f} req_grad={new_lps.requires_grad}", flush=True)
        if not torch.isfinite(new_lps).all():
            print("  -> SKIP: new_lps non-finite", flush=True); continue

        model._model.disable_adapter_layers()
        with torch.no_grad():
            ref_logits = model._model(input_ids, use_cache=False).logits[0].float()
        model._model.enable_adapter_layers()
        model._model.set_adapter(ROLE_ADAPTER.get(role, "proposer"))
        ref_lps = logprobs_from_logits(ref_logits[p_len-1:p_len+n_resp-1], resp_labels.to(ref_logits.device))
        del ref_logits
        print(f"  ref_lps: shape={ref_lps.shape} finite={torch.isfinite(ref_lps).all()} "
              f"mean={ref_lps.mean():.3f}", flush=True)
        if not torch.isfinite(ref_lps).all():
            print("  -> SKIP: ref_lps non-finite", flush=True); continue

        old_aligned = old_lps[:n_align].to(new_lps.device)
        log_ratio   = (new_lps[:n_align] - old_aligned).sum()
        print(f"  log_ratio={log_ratio:.4f} finite={torch.isfinite(log_ratio)}", flush=True)
        if not torch.isfinite(log_ratio):
            print("  -> SKIP: log_ratio non-finite", flush=True); continue

        kl = (new_lps - ref_lps).mean()
        print(f"  kl={kl:.4f} -> PASS", flush=True)

    print("\nDone.", flush=True)
