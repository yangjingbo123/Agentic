import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from copy import deepcopy

ROLES = ["controller", "proposer", "critic", "verifier"]
DEVICE = "cuda:0"


class SharedModel:
    """单个 base 模型挂多套 LoRA adapter，按角色切换。显存只占一份 base (~16GB)。"""

    def __init__(self, peft_model):
        self._model = peft_model
        self.device = next(peft_model.parameters()).device

    def set_role(self, role: str):
        self._model.set_adapter(role)

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def parameters(self, role: str = None):
        if role:
            return (p for n, p in self._model.named_parameters()
                    if f'.{role}.' in n and p.requires_grad)
        return self._model.parameters()

    def named_parameters(self):
        return self._model.named_parameters()

    def save_pretrained(self, path, role: str = None):
        save_role = role or ROLES[0]
        self._model.set_adapter(save_role)
        self._model.save_pretrained(path, selected_adapters=[save_role])


def load_trainable_models(model_path: str, lora_rank: int = 16, sft_checkpoint: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    base = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=DEVICE
    )

    ref_base = deepcopy(base)
    for p in ref_base.parameters():
        p.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=lora_rank, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(base, lora_cfg, adapter_name=ROLES[0])
    for role in ROLES[1:]:
        peft_model.add_adapter(role, lora_cfg)

    if sft_checkpoint:
        import safetensors.torch as st
        import os
        for role in ROLES:
            ckpt = f"{sft_checkpoint}/{role}/{role}/adapter_model.safetensors"
            if os.path.exists(ckpt):
                weights = st.load_file(ckpt, device=str(DEVICE))
                # safetensors key: "...lora_A.weight" → peft key: "...lora_A.{role}.weight"
                role_weights = {
                    k.replace("lora_A.weight", f"lora_A.{role}.weight")
                     .replace("lora_B.weight", f"lora_B.{role}.weight"): v
                    for k, v in weights.items()
                }
                missing = []
                for name, param in peft_model.named_parameters():
                    if f'.{role}.' in name and name in role_weights:
                        param.data.copy_(role_weights[name])
                    elif f'.{role}.' in name and param.requires_grad:
                        missing.append(name)
                if missing:
                    print(f"Warning: {len(missing)} params not found for {role}")
        print(f"Loaded SFT checkpoint from {sft_checkpoint}", flush=True)

    for n, p in peft_model.named_parameters():
        if any(f'.{role}.' in n for role in ROLES):
            p.requires_grad_(True)

    peft_model.gradient_checkpointing_enable()
    shared = SharedModel(peft_model)

    models = {role: shared for role in ROLES}
    ref_models = {role: ref_base for role in ROLES}

    return models, ref_models, tokenizer
