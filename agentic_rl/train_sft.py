"""SFT 监督微调 - 单模型，四角色混合训练"""
import json
import os

import hydra
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(path) as f:
            for line in f:
                episode = json.loads(line)
                for turn in episode.get("turns", []):
                    if all(k in turn for k in ("system", "user", "response")):
                        self.examples.append(turn)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        turn = self.examples[idx]
        messages = [
            {"role": "system",    "content": turn["system"]},
            {"role": "user",      "content": turn["user"]},
            {"role": "assistant", "content": turn["response"]},
        ]
        full_text   = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full_ids   = list(self.tokenizer.encode(full_text,   add_special_tokens=False))
        prompt_ids = list(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        full_ids = full_ids[:self.max_length]
        labels   = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids):])[:self.max_length]
        pad = self.max_length - len(full_ids)
        return {
            "input_ids":      torch.tensor(full_ids + [self.tokenizer.pad_token_id or 0] * pad),
            "attention_mask": torch.tensor([1] * len(full_ids) + [0] * pad),
            "labels":         torch.tensor(labels + [-100] * pad),
        }


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    wandb.init(project="agentic-rl", name=f"sft-{cfg.exp_name}",
               config=OmegaConf.to_container(cfg, resolve=True))

    sft_cfg = cfg.get("sft", {})
    model_path = cfg.llm.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 用 torch_dtype 而非 dtype：dtype 是 transformers 4.56+ 的新参数名，较旧版本
    # 会当成 model_kwargs 透传给 cls(config, **kwargs) 而报 unexpected keyword。
    # torch_dtype 在新旧版本都可用，也与 llm/trainable_llm.py 保持一致。
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, lora_cfg)
    model.config.use_cache = False
    model.enable_input_require_grads()
    # use_reentrant=False：与 RL 路径（trainable_llm.py）一致。reentrant 版本在
    # LoRA 上易出现梯度不回传且不报错的静默失败。
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dataset = SFTDataset(cfg.data.get("sft_path", "data/sft_train.jsonl"), tokenizer,
                         sft_cfg.get("max_len", 512))
    print(f"SFT dataset: {len(dataset)} turns", flush=True)

    loader    = DataLoader(dataset, batch_size=sft_cfg.get("batch", 2), shuffle=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=sft_cfg.get("lr", 2e-5))
    device = next(model.parameters()).device
    epochs = sft_cfg.get("epochs", 3)

    step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            shift_logits  = logits[:, :-1].contiguous()
            shift_labels  = input_ids[:, 1:].contiguous()
            response_mask = (labels[:, 1:] != -100)
            if response_mask.sum() == 0:
                step += 1
                continue
            loss = F.cross_entropy(shift_logits[response_mask].view(-1, logits.size(-1)),
                                   shift_labels[response_mask].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            total_loss += loss.item()
            step += 1
            if step % 20 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)
                wandb.log({"sft_loss": loss.item(), "epoch": epoch}, step=step)

        print(f"epoch={epoch} avg_loss={total_loss/max(len(loader),1):.4f}", flush=True)

    save_dir = sft_cfg.get("save_dir", "checkpoints/sft")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"Saved to {save_dir}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
