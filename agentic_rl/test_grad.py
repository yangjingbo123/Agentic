import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

model_path = "/scratch/users/k24104674/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
tokenizer = AutoTokenizer.from_pretrained(model_path)
base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(base, lora_cfg)
model.config.use_cache = False
model.gradient_checkpointing_enable()

text = "Hello world, this is a test."
ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

for i in range(3):
    opt.zero_grad()
    out = model(input_ids=ids, use_cache=False)
    loss = F.cross_entropy(out.logits[:,:-1].reshape(-1, out.logits.size(-1)), ids[:,1:].reshape(-1))
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    opt.step()
    print(f"step={i} loss={loss.item():.4f} grad_norm={grad_norm:.4f}", flush=True)
