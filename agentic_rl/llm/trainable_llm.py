import os
import torch
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel


ROLE_ADAPTER = {
    "proposer":   "proposer",
    "controller": "controller",
    "critic":     "critic",
    "verifier":   "verifier",
}


class RoleModel:
    """单模型，四个独立 LoRA adapter（proposer/controller/critic/verifier），base 权重兼作参考模型。"""

    def __init__(self, peft_model):
        self._model = peft_model
        self.device = next(peft_model.parameters()).device

    @contextmanager
    def as_ref(self):
        """禁用所有 adapter，使用 base 权重作参考模型。"""
        self._model.disable_adapter_layers()
        try:
            yield self._model
        finally:
            self._model.enable_adapter_layers()

    @contextmanager
    def as_role(self, role: str):
        """切换到指定角色对应的 adapter。不恢复之前的 adapter，避免清空计算图。"""
        adapter_name = ROLE_ADAPTER.get(role, "proposer")
        self._model.set_adapter(adapter_name)
        try:
            yield self._model
        finally:
            pass  # 保持当前 adapter，避免 set_adapter 导致梯度失效

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def parameters(self):
        return (p for p in self._model.parameters() if p.requires_grad)

    def lora_parameters(self):
        """返回全部四个 adapter 的 LoRA 参数（不依赖 requires_grad）。

        PEFT 的 set_adapter() 会把非激活 adapter 的 requires_grad 置为 False。
        如果优化器只收集 requires_grad=True 的参数，则只有「最后被
        set_adapter 的那个 adapter」会进入优化器，其余 adapter 永远不会被
        优化器更新，也不会被 zero_grad() 清零——梯度持续累加。
        因此优化器和 clip_grad_norm_ 都应使用此方法。
        """
        return [p for n, p in self._model.named_parameters() if "lora_" in n]

    def named_parameters(self):
        return self._model.named_parameters()

    def save_pretrained(self, path):
        self._model.save_pretrained(path)


def load_trainable_models(model_path: str, lora_rank: int = 16, sft_checkpoint: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 训练模型固定在 cuda:0，vLLM 使用剩余 GPU，避免显存冲突
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )

    lora_cfg = LoraConfig(
        r=lora_rank, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg, adapter_name="proposer")
    for name in ("controller", "critic", "verifier"):
        model.add_adapter(name, lora_cfg)

    # get_peft_model 的 autocast_adapter_dtype 只作用于首个 adapter（proposer），
    # 其余三个 add_adapter 创建的 adapter 可能留在 F32。统一 cast 为 base 的 dtype。
    base_dtype = next(base.parameters()).dtype
    for name, param in model.named_parameters():
        if "lora_" in name and param.dtype != base_dtype:
            param.data = param.data.to(base_dtype)

    # 恢复 SFT checkpoint 到所有 adapter（所有角色从相同的 SFT 起点开始训练）
    # 支持两种目录结构：
    #   1) {sft_checkpoint}/{role}/{role}/adapter_model.safetensors （当前实际结构）
    #   2) {sft_checkpoint}/adapter_model.safetensors                （扁平结构，兼容）
    if sft_checkpoint:
        if not os.path.isdir(sft_checkpoint):
            raise FileNotFoundError(
                f"sft_checkpoint 目录不存在: {sft_checkpoint}"
            )
        import safetensors.torch as st
        for adapter_name in ("proposer", "controller", "critic", "verifier"):
            # 优先查找嵌套结构 {sft}/{role}/{role}/
            nested = os.path.join(sft_checkpoint, adapter_name, adapter_name,
                                  "adapter_model.safetensors")
            flat   = os.path.join(sft_checkpoint, adapter_name,
                                  "adapter_model.safetensors")
            top    = os.path.join(sft_checkpoint, "adapter_model.safetensors")
            ckpt   = nested if os.path.isfile(nested) else flat
            if not os.path.isfile(ckpt):
                ckpt = top
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(
                    f"SFT checkpoint 未找到 adapter_model.safetensors "
                    f"for adapter '{adapter_name}'，已尝试:\n"
                    f"  {nested}\n  {flat}\n  {top}"
                )
            weights = st.load_file(ckpt)
            model.set_adapter(adapter_name)
            loaded = 0
            for name, param in model.named_parameters():
                if f"lora_" not in name:
                    continue
                # model param name: "...lora_A.{adapter_name}.weight"
                # ckpt key format:  "...lora_A.weight"
                src_key = name.replace(f".{adapter_name}.", ".")
                if src_key in weights:
                    param.data.copy_(weights[src_key].to(param.device))
                    loaded += 1
            print(f"  [sft] {adapter_name}: loaded {loaded} params from {ckpt}", flush=True)
        print(f"Loaded SFT checkpoint into all adapters from {sft_checkpoint}", flush=True)

    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return RoleModel(model), tokenizer
