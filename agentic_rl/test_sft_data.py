import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/cephfs/volumes/hpc_home/k24104674/aed22256-9e0b-4f4f-86c1-c56793988876/jingbo/marl/Agentic/agentic_rl')
from train_sft import SFTDataset

model_path = "/scratch/users/k24104674/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
tokenizer = AutoTokenizer.from_pretrained(model_path)
base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(base, lora_cfg)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()  # 测试这行是否解决问题

dataset = SFTDataset("data/sft_train.jsonl", tokenizer, max_length=512)
loader = DataLoader(dataset, batch_size=2, shuffle=False)
batch = next(iter(loader))
device = next(model.parameters()).device

input_ids      = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
labels         = batch["labels"].to(device)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
for i in range(3):
    opt.zero_grad()
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    response_mask = (labels[:, 1:] != -100)
    loss = F.cross_entropy(shift_logits[response_mask].view(-1, logits.size(-1)), shift_labels[response_mask].view(-1))
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    opt.step()
    print(f"step={i} loss={loss.item():.4f} grad_norm={grad_norm:.4f}", flush=True)


