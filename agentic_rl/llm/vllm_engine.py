import os
import shutil
import tempfile
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ROLES = ["controller", "proposer", "critic", "verifier"]


class VLLMInferenceEngine:
    """vLLM 推理引擎，支持多卡 tensor parallelism 和多角色 LoRA 热更新。"""

    def __init__(self, model_path: str, max_tokens: int = 512,
                 gpu_memory_utilization: float = 0.45):
        self.max_tokens = max_tokens
        self.lora_dir = tempfile.mkdtemp(prefix="agentic_lora_")
        for role in ROLES:
            os.makedirs(os.path.join(self.lora_dir, role), exist_ok=True)

        n_gpus = torch.cuda.device_count()
        # 多卡时用 tensor parallelism，单卡时 tp=1
        tp_size = min(n_gpus, 4) if n_gpus >= 2 else 1

        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=16,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            tensor_parallel_size=tp_size,
        )
        self._lora_loaded = {role: False for role in ROLES}
        print(f"vLLM initialized with tensor_parallel_size={tp_size}", flush=True)

    def generate(self, role: str, prompt: str) -> tuple[str, list[float]]:
        lora_req = LoRARequest(
            lora_name=role,
            lora_int_id=ROLES.index(role) + 1,
            lora_path=os.path.join(self.lora_dir, role),
        ) if self._lora_loaded[role] else None

        params = SamplingParams(max_tokens=self.max_tokens, temperature=1.0, logprobs=1)
        outputs = self.llm.generate(
            [prompt], sampling_params=params,
            lora_request=lora_req, use_tqdm=False,
        )
        out = outputs[0].outputs[0]
        log_probs = [list(lp.values())[0].logprob for lp in (out.logprobs or [])]
        return out.text, log_probs

    def sync_lora(self, model, role: str):
        """把 HF LoRA 权重保存到磁盘，供 vLLM 下次加载。"""
        role_dir = os.path.join(self.lora_dir, role)
        tmp_dir = role_dir + "_tmp"
        model.save_pretrained(tmp_dir)
        # save_pretrained 可能在子目录，找到 adapter_config.json 所在目录
        for root, dirs, files in os.walk(tmp_dir):
            if "adapter_config.json" in files:
                if os.path.exists(role_dir):
                    shutil.rmtree(role_dir)
                shutil.move(root, role_dir)
                break
        shutil.rmtree(tmp_dir, ignore_errors=True)
        self._lora_loaded[role] = True

    def sync_all_loras(self, models: dict):
        for role, model in models.items():
            self.sync_lora(model, role)

