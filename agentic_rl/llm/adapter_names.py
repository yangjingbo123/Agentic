"""LoRA adapter 命名（零依赖，可 CPU 单测）。

**为什么 ref adapter 不能叫 `ref_{role}`**

PEFT 的 `get_peft_model_state_dict` 用**子串匹配**筛选某个 adapter 的权重：

    {k: v for k, v in sd.items() if ("lora_" in k and adapter_name in k) or ...}

若 ref 名含角色名（`ref_controller ⊃ controller`），导出 `controller` 时
ref 权重会被误纳入；而后续 key 清理用 `k.replace(f".{adapter_name}", "")`，
`.ref_controller.` 中的 `controller` 前面是 `_` 而非 `.`，替换不生效，于是
`lora_A.ref_controller.weight` 原样写进 safetensors，vLLM 加载时抛
`ValueError: ... is unsupported LoRA weight`（v2.1 首启即崩）。

因此 ref 名必须与所有角色名**互不为子串**，见 `_validate()`。
"""

ROLE_ADAPTER = {
    "proposer":   "proposer",
    "controller": "controller",
    "critic":     "critic",
    "verifier":   "verifier",
}

# 训练 adapter 名（与 vLLM 侧 lora 名一致）
ADAPTER_NAMES = ("proposer", "controller", "critic", "verifier")

# KL 参考快照 adapter：故意不含任何角色名子串（见模块 docstring）
REF_ADAPTER = {role: f"klref{i}" for i, role in enumerate(ADAPTER_NAMES)}

# 判定「是否为 ref 参数」的统一前缀，供 lora_parameters / save_pretrained 过滤
REF_PREFIX = "klref"


def _validate():
    """启动即校验命名不冲突，避免再次踩 PEFT 子串匹配的坑。"""
    refs = list(REF_ADAPTER.values())
    assert len(set(refs)) == len(refs), f"ref 名重复: {refs}"
    for role in ADAPTER_NAMES:
        for ref in refs:
            assert role not in ref, f"ref 名 {ref!r} 含角色名 {role!r} 子串"
            assert ref not in role, f"角色名 {role!r} 含 ref 名 {ref!r} 子串"
    for a in refs:
        for b in refs:
            if a is not b:
                assert a not in b, f"ref 名互为子串: {a!r} ⊂ {b!r}"
    for ref in refs:
        assert ref.startswith(REF_PREFIX), f"{ref!r} 不以 {REF_PREFIX!r} 开头"


_validate()
