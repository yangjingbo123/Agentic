"""
Stage 1: SFT 监督微调
用生成的高质量轨迹数据微调模型，教会格式和基本协作行为。
"""
import argparse
import json
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(path) as f:
            for line in f:
                episode = json.loads(line)
                for turn in episode.get("turns", []):
                    if not all(k in turn for k in ("role_name", "system", "user", "response")):
                        continue
                    self.examples.append(turn)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        turn = self.examples[idx]
        messages = [
            {"role": "system", "content": turn["system"]},
            {"role": "user",   "content": turn["user"]},
            {"role": "assistant", "content": turn["response"]},
        ]
        # 完整对话 token ids
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_ids   = self.tokenizer.encode(full_text,   add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if not isinstance(full_ids, list):
            full_ids = list(full_ids)
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)
        prompt_len = len(prompt_ids)

        full_ids = full_ids[:self.max_length]
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:self.max_length]

        # padding
        pad = self.max_length - len(full_ids)
        input_ids = full_ids + [self.tokenizer.pad_token_id or 0] * pad
        labels    = labels   + [-100] * pad
        attention_mask = [1] * len(full_ids) + [0] * pad

        return {
            "input_ids":      torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels":         torch.tensor(labels),
            "role_name":      turn["role_name"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--sft_data", default="data/sft_train.jsonl")
    parser.add_argument("--epochs",   type=int,   default=3)
    parser.add_argument("--lr",       type=float, default=2e-5)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--max_len",  type=int,   default=1024)
    parser.add_argument("--save_dir", default="checkpoints/sft")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from llm.trainable_llm import load_trainable_models, ROLES
    models, _, tokenizer = load_trainable_models(config["llm"]["model_path"])
    shared = models["controller"]  # 所有角色共享同一个 SharedModel

    dataset = SFTDataset(args.sft_data, tokenizer, args.max_len)
    print(f"SFT dataset: {len(dataset)} turns", flush=True)

    # 每个角色独立 optimizer
    optimizers = {
        role: torch.optim.AdamW(list(shared.parameters(role)), lr=args.lr)
        for role in ROLES
    }

    device = shared.device

    for epoch in range(args.epochs):
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
        total_loss = 0.0
        steps = 0

        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            role_names     = batch["role_name"]

            # 按 role 分组处理（确保 adapter 切换正确）
            from collections import defaultdict
            role_indices = defaultdict(list)
            for i, role in enumerate(role_names):
                role_indices[role].append(i)

            batch_loss = torch.tensor(0.0, device=device)
            n_valid = 0

            for role, indices in role_indices.items():
                shared.set_role(role)
                optimizers[role].zero_grad()

                idx = torch.tensor(indices, device=device)
                out = shared(
                    input_ids=input_ids[idx],
                    attention_mask=attention_mask[idx],
                )
                logits = out.logits  # [B, seq, vocab]
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = labels[idx, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(shared.parameters(role)), 1.0)
                optimizers[role].step()

                batch_loss = batch_loss + loss.detach()
                n_valid += 1

            total_loss += (batch_loss / max(n_valid, 1)).item()
            steps += 1

            if steps % 20 == 0:
                print(f"epoch={epoch} step={steps} loss={total_loss/steps:.4f}", flush=True)

        print(f"epoch={epoch} avg_loss={total_loss/steps:.4f}", flush=True)

    # 保存每个角色的 LoRA checkpoint
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    for role in ROLES:
        shared.save_pretrained(f"{args.save_dir}/{role}", role=role)
        print(f"Saved {role} to {args.save_dir}/{role}")


if __name__ == "__main__":
    main()
