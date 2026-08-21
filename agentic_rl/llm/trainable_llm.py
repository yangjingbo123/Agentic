import importlib.util
import os
import torch
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel

from llm.adapter_names import ADAPTER_NAMES, REF_ADAPTER, REF_PREFIX, ROLE_ADAPTER


class RoleModel:
    """单模型，四个独立 LoRA adapter（proposer/controller/critic/verifier），base 权重兼作参考模型。"""

    def __init__(self, peft_model):
        self._model = peft_model
        self.device = next(peft_model.parameters()).device

    @contextmanager
    def as_ref(self, role: str = "proposer"):
        """切换到冻结的 SFT 快照 adapter（klref{i}）作参考模型。

        v2.0 曾 disable_adapter_layers() 用裸 base 作参考：KL 惩罚把策略持续
        拉向从未学过 <interaction>/decision: stop 格式的 base，系统性抹除 SFT
        行为（v2.0 全量日志：kl 0.75→0.18 单调下降，与 int/stop 坍塌同步）。
        KL 锚定 SFT 起点才是 trust region 的本意。

        注意 ref 名不能叫 `ref_{role}`：PEFT 的 get_peft_model_state_dict 用
        子串匹配筛选，导致导出 controller 时把 ref_controller 权重误纳入
        → sync_lora 给 vLLM 的 safetensors 里带上 `.ref_controller.`后缀
        → vLLM 报 unsupported LoRA weight 而崩。新命名见 adapter_names.py。
        无 ref adapter 时回退旧行为（兼容未建 ref 的旧 checkpoint）。
        """
        ref_name = REF_ADAPTER.get(ROLE_ADAPTER.get(role, "proposer"),
                                   REF_ADAPTER["proposer"])
        if ref_name in self._model.peft_config:
            self._model.set_adapter(ref_name)
            try:
                yield self._model
            finally:
                # set_adapter 会把 ref 参数 requires_grad 置 True，回滚冻结状态
                for n, p in self._model.named_parameters():
                    if f".{ref_name}." in n:
                        p.requires_grad_(False)
        else:
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
        klref* adapter 是冻结的 KL 参考快照，永远不进优化器。
        """
        return [p for n, p in self._model.named_parameters()
                if "lora_" in n and f".{REF_PREFIX}" not in n]

    def named_parameters(self):
        return self._model.named_parameters()

    def save_pretrained(self, path):
        # 只保存训练 adapter；klref* 是 SFT 快照，load 时从 SFT ckpt 重建即可。
        # 关键：ref 名与角色名互不为子串（adapter_names.py 保证），否则 PEFT
        # 子串匹配会将 ref 权重混入导出→ vLLM 加载崩。
        trainable = [n for n in self._model.peft_config
                     if not n.startswith(REF_PREFIX)]
        self._model.save_pretrained(path, selected_adapters=trainable)


def load_trainable_models(model_path: str, lora_rank: int = 16, sft_checkpoint: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 训练模型固定在 cuda:0，vLLM 使用剩余 GPU，避免显存冲突。
    # flash-attn 非必需：镜像缺包时回退 sdpa（正确性一致，速度略降），
    # 避免在 Primus 等定制镜像上因 pip 编译 flash-attn 阻塞作业。
    attn_impl = ("flash_attention_2"
                 if importlib.util.find_spec("flash_attn") else "sdpa")
    print(f"[load] attn_implementation = {attn_impl}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation=attn_impl,
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

    # ── KL 参考 = 冻结的 SFT 快照（ref_* adapter） ─────────────────────────
    # 每个角色克隆一份训练起点权重并冻结，as_ref(role) 切换到 klref{i}。
    # 无 SFT ckpt 时 LoRA B 为零初始化，ref 等价于 base，行为与旧实现一致。
    # 注意必须在 resume 覆盖 role adapter 之前克隆（train.py 的 resume 只回填
    # role adapter），保证 ref 始终是 SFT 起点而非 RL 中间态。
    params_by_name = dict(model.named_parameters())
    for role_name in ADAPTER_NAMES:
        ref_name = REF_ADAPTER[role_name]
        model.add_adapter(ref_name, lora_cfg)
        for name, param in model.named_parameters():
            if "lora_" not in name or f".{ref_name}." not in name:
                continue
            src = params_by_name[name.replace(f".{ref_name}.", f".{role_name}.")]
            param.data = src.data.clone()   # clone 同时对齐 dtype（bf16）
            param.requires_grad_(False)
    model.set_adapter("proposer")   # add_adapter 会激活新 adapter，切回训练态
    print(f"Created frozen ref adapters {list(REF_ADAPTER.values())} "
          f"(KL anchor = SFT snapshot)", flush=True)

    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return RoleModel(model), tokenizer
