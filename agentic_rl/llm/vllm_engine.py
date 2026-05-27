import os
import tempfile
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ROLES = ["controller", "proposer", "critic", "verifier"]


class VLLMInferenceEngine:
    """vLLM 推理引擎，支持多角色 LoRA 热更新。"""

    def __init__(self, model_path: str, max_tokens: int = 512,
                 gpu_memory_utilization: float = 0.45):
        self.max_tokens = max_tokens
        self.lora_dir = tempfile.mkdtemp(prefix="agentic_lora_")
        # 为每个角色创建占位目录（首次加载前需要存在）
        for role in ROLES:
            os.makedirs(os.path.join(self.lora_dir, role), exist_ok=True)

        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=16,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=1.0,
        )
        self._lora_loaded = {role: False for role in ROLES}

    def generate(self, role: str, prompt: str) -> tuple[str, list[float]]:
        """生成文本，返回 (response, log_probs)。"""
        lora_req = LoRARequest(
            lora_name=role,
            lora_int_id=ROLES.index(role) + 1,
            lora_path=os.path.join(self.lora_dir, role),
        ) if self._lora_loaded[role] else None

        params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=1.0,
            logprobs=1,
        )
        outputs = self.llm.generate(
            [prompt], sampling_params=params,
            lora_request=lora_req, use_tqdm=False,
        )
        out = outputs[0].outputs[0]
        response = out.text
        log_probs = [
            list(lp.values())[0].logprob
            for lp in (out.logprobs or [])
        ]
        return response, log_probs

    def sync_lora(self, shared_model, role: str):
        """把 HF LoRA 权重保存到磁盘，供 vLLM 下次加载。"""
        role_dir = os.path.join(self.lora_dir, role)
        shared_model.save_pretrained(role_dir, role=role)
        self._lora_loaded[role] = True

    def sync_all_loras(self, shared_model):
        for role in ROLES:
            self.sync_lora(shared_model, role)
