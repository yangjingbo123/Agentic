"""
SFT warm-up for 4 role-specific LoRA adapters.

Run via torchrun (one process per role, 2 GPUs each):
    torchrun --nproc_per_node 2 train_sft_4role.py --role proposer
Or via scripts/train_sft_4role.sh which launches all 4 in parallel.
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import wandb


ROLES = ["controller", "proposer", "critic", "verifier"]


class RoleDataset(Dataset):
    """Load turns for a single role from sft_train.jsonl."""

    def __init__(self, path: str, role: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(path) as f:
            for line in f:
                ep = json.loads(line)
                for turn in ep.get("turns", []):
                    if turn.get("role_name") == role:
                        self.examples.append(turn)

        print(f"[{role}] loaded {len(self.examples)} turns from {path}", flush=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        turn = self.examples[idx]
        messages = [
            {"role": "system",    "content": turn["system"]},
            {"role": "user",      "content": turn["user"]},
            {"role": "assistant", "content": turn["response"]},
        ]
        full_text   = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)

        full_ids   = self.tokenizer.encode(full_text,   add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        full_ids = full_ids[:self.max_length]
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels   = [-100] * n_prompt + full_ids[n_prompt:]
        labels   = labels[:self.max_length]

        pad_len = self.max_length - len(full_ids)
        pad_id  = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        return {
            "input_ids":      torch.tensor(full_ids  + [pad_id] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(full_ids) + [0] * pad_len, dtype=torch.long),
            "labels":         torch.tensor(labels    + [-100]   * pad_len, dtype=torch.long),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--role",        required=True, choices=ROLES)
    p.add_argument("--model_path",  default="/data/yangjingbo/models/Qwen3-8B")
    p.add_argument("--data_path",   default="data/sft_train.jsonl")
    p.add_argument("--save_dir",    default="checkpoints/sft")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=4)   # per GPU
    p.add_argument("--max_length",  type=int,   default=1024)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--lora_rank",   type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--warmup_ratio",type=float, default=0.1)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--wandb_project", default="agentic_rl_sft")
    p.add_argument("--no_wandb",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Distributed setup ────────────────────────────────────────────────
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    is_main     = (local_rank == 0)

    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ── W&B (main process only) ──────────────────────────────────────────
    if is_main and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"sft_{args.role}",
            config=vars(args),
        )

    # ── Tokenizer & model ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    base_model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg, adapter_name=args.role)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if is_main:
        model.print_trainable_parameters()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Data ──────────────────────────────────────────────────────────────
    dataset = RoleDataset(args.data_path, args.role, tokenizer, args.max_length)
    if len(dataset) == 0:
        print(f"[{args.role}] WARNING: no training data found — check data_path.", flush=True)
        sys.exit(0)

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
    )

    # ── Optimizer & scheduler ────────────────────────────────────────────
    raw_model  = model.module if world_size > 1 else model
    optimizer  = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps   = args.epochs * len(loader)
    warmup_steps  = int(args.warmup_ratio * total_steps)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

            if torch.isnan(loss) or not torch.isfinite(loss):
                print(f"[{args.role}] step={global_step} SKIP non-finite loss", flush=True)
                global_step += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if is_main and global_step % 20 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"[{args.role}] epoch={epoch} step={global_step} "
                      f"loss={loss.item():.4f} lr={lr_now:.2e}", flush=True)
                if not args.no_wandb:
                    wandb.log({f"{args.role}/loss": loss.item(),
                               f"{args.role}/lr":   lr_now}, step=global_step)

        if is_main:
            avg = epoch_loss / max(len(loader), 1)
            print(f"[{args.role}] epoch={epoch} avg_loss={avg:.4f}", flush=True)

    # ── Save adapter ─────────────────────────────────────────────────────
    if is_main:
        save_path = os.path.join(args.save_dir, args.role)
        os.makedirs(save_path, exist_ok=True)
        raw_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[{args.role}] adapter saved to {save_path}", flush=True)
        if not args.no_wandb:
            wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
