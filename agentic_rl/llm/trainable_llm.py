import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

ROLE_DEVICES = {
    "controller": "cuda:0",
    "proposer":   "cuda:1",
    "critic":     "cuda:2",
    "verifier":   "cuda:3",
}


def _load_one(model_path: str, device: str, lora_rank: int):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    lora_cfg = LoraConfig(
        r=lora_rank, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def load_trainable_models(model_path: str, lora_rank: int = 16) -> tuple[dict, dict, any]:
    """
    返回 (models, ref_models, tokenizer)
    - models:     可训练，每个角色独立LoRA
    - ref_models: 冻结副本，用于KL penalty，与models共享同一张卡
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    models = {}
    ref_models = {}
    for role, device in ROLE_DEVICES.items():
        models[role] = _load_one(model_path, device, lora_rank)

        # reference model：同设备，全量冻结，无LoRA
        ref = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        for p in ref.parameters():
            p.requires_grad_(False)
        ref_models[role] = ref

    return models, ref_models, tokenizer
