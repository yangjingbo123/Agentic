"""
完整流程诊断：
  1. enable_thinking=False 去掉了 <think> token
  2. 加载训练模型 + vLLM，sync LoRA，rollout 一次
  3. 检验 new_lps vs old_lps，log_ratio 是否 < 50
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "/scratch/users/k24104674/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
SFT_CKPT   = "checkpoints/sft"

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    messages = [{"role": "system", "content": "你是一个数学助手"},
                {"role": "user",   "content": "1+1等于多少？"}]

    # ── 1. thinking mode ────────────────────────────────────────────────────
    p_think   = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_nothink = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    THINK_ID  = tok.encode("<think>", add_special_tokens=False)
    ids_t  = tok.encode(p_think,   add_special_tokens=False)
    ids_nt = tok.encode(p_nothink, add_special_tokens=False)

    print("=== [1] Thinking mode ===")
    print(f"  with    thinking: len={len(ids_t)},  last={tok.convert_ids_to_tokens([ids_t[-1]])}")
    print(f"  without thinking: len={len(ids_nt)}, last={tok.convert_ids_to_tokens([ids_nt[-1]])}")
    ok = not (ids_nt[-len(THINK_ID):] == THINK_ID)
    print(f"  {'PASS' if ok else 'FAIL'}: no <think> in no-think prompt")

    # ── 2. full rollout ──────────────────────────────────────────────────────
    print("\n=== [2] Full rollout + log_ratio check ===")
    from llm.trainable_llm import load_trainable_models, ROLE_ADAPTER
    from llm.vllm_engine import VLLMInferenceEngine

    model, tokenizer = load_trainable_models(MODEL_PATH, sft_checkpoint=SFT_CKPT)
    print("  Model loaded.")

    engine = VLLMInferenceEngine(MODEL_PATH, max_tokens=64, gpu_memory_utilization=0.85)
    print("  vLLM loaded.")

    engine.sync_lora(model)
    print("  sync_lora done.")

    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
    response, old_lps, token_ids = engine.generate("proposer", prompt)
    print(f"  Response: {response!r}")
    print(f"  Tokens: {len(token_ids)},  old_lps mean: {sum(old_lps)/max(len(old_lps),1):.3f}")

    device = model.device
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    p_len = len(prompt_ids)
    vocab = model._model.config.vocab_size
    resp_ids = [t for t in token_ids if 0 <= t < vocab]
    if not resp_ids:
        print("  FAIL: resp_ids empty"); sys.exit(1)

    input_ids   = torch.tensor(prompt_ids + resp_ids, dtype=torch.long, device=device).unsqueeze(0)
    resp_labels = torch.tensor(resp_ids, dtype=torch.long, device=device)

    model._model.set_adapter(ROLE_ADAPTER.get("proposer", "proposer"))
    with torch.no_grad():
        logits = model._model(input_ids, use_cache=False).logits[0].float()

    new_lps = F.log_softmax(logits[p_len-1: p_len+len(resp_ids)-1], dim=-1)\
                .gather(1, resp_labels.unsqueeze(1)).squeeze(1)

    n = min(len(new_lps), len(old_lps))
    old_t     = torch.tensor(old_lps[:n], device=device)
    log_ratio = (new_lps[:n] - old_t).sum().item()

    print(f"\n  new_lps mean : {new_lps.mean().item():.3f}")
    print(f"  old_lps mean : {old_t.mean().item():.3f}")
    print(f"  log_ratio    : {log_ratio:.3f}  (over {n} tokens)")
    print(f"  all finite   : {torch.isfinite(new_lps).all().item()}")

    if abs(log_ratio) > 50:
        print(f"  FAIL: |log_ratio|={abs(log_ratio):.1f} > 50")
        sys.exit(1)
    print(f"  PASS: |log_ratio|={abs(log_ratio):.1f} < 50")
