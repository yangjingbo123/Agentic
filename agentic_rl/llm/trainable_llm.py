import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from copy import deepcopy

ROLES = ["controller", "proposer", "critic", "verifier"]

# 4卡设计：每个角色独立模型，各占一张卡
ROLE_DEVICES = {
    "controller": "cuda:0",
    "proposer":   "cuda:1",
    "critic":     "cuda:2",
    "verifier":   "cuda:3",
}


def _num_gpus():
    return torch.cuda.device_count()


def _get_device(role: str) -> str:
    """单卡时全部用 cuda:0，多卡时按角色分配。"""
    if _num_gpus() >= 4:
        return ROLE_DEVICES[role]
    return "cuda:0"


class RoleModel:
    """单个角色的独立模型（多卡时每个角色在不同 GPU）。"""

    def __init__(self, peft_model):
        self._model = peft_model
        self.device = next(peft_model.parameters()).device

    def set_role(self, role: str):
        pass  # 独立模型无需切换 adapter

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def parameters(self, role: str = None):
        return (p for p in self._model.parameters() if p.requires_grad)

    def named_parameters(self):
        return self._model.named_parameters()

    def save_pretrained(self, path, role: str = None):
        self._model.save_pretrained(path)


def _load_one(model_path: str, device: str, lora_cfg: LoraConfig,
              sft_ckpt_dir: str = None) -> RoleModel:
    base = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device
    )
    peft_model = get_peft_model(base, lora_cfg)

    if sft_ckpt_dir and os.path.exists(sft_ckpt_dir):
        import safetensors.torch as st
        ckpt = os.path.join(sft_ckpt_dir, "adapter_model.safetensors")
        if os.path.exists(ckpt):
            weights = st.load_file(ckpt, device=device)
            # safetensors key: "...lora_A.weight" → peft key: "...lora_A.default.weight"
            for name, param in peft_model.named_parameters():
                key = name.replace(".default.", ".")
                if key in weights and param.requires_grad:
                    param.data.copy_(weights[key].to(param.device))

    for p in peft_model.parameters():
        if not p.requires_grad:
            continue
    peft_model.gradient_checkpointing_enable()
    return RoleModel(peft_model)


def load_trainable_models(model_path: str, lora_rank: int = 16, sft_checkpoint: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    lora_cfg = LoraConfig(
        r=lora_rank, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    models = {}
    ref_models = {}

    for role in ROLES:
        device = _get_device(role)
        sft_dir = f"{sft_checkpoint}/{role}/{role}" if sft_checkpoint else None
        models[role] = _load_one(model_path, device, lora_cfg, sft_dir)

        ref = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device
        )
        for p in ref.parameters():
            p.requires_grad_(False)
        ref_models[role] = ref

    if sft_checkpoint:
        print(f"Loaded SFT checkpoint from {sft_checkpoint}", flush=True)

    return models, ref_models, tokenizer
