"""LLM调用接口 - 支持vLLM本地部署"""
from typing import Dict, Any, Optional
import hashlib
import json
import os


class LLMInterface:
    """LLM调用接口"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model", "Qwen/Qwen3-7B-Instruct")
        self.backend = config.get("backend", "vllm")
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_dir = config.get("cache_dir", "./llm_cache")
        self.cache = {}

        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_cache()

        self.llm = None
        if self.backend == "vllm":
            self._init_vllm()

    def _init_vllm(self):
        """初始化vLLM"""
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.3),
                quantization=self.config.get("quantization", "awq"),
                max_model_len=self.config.get("max_model_len", 2048),
                dtype="half"
            )
            self.SamplingParams = SamplingParams
        except ImportError:
            print("Warning: vLLM not installed, using mock mode")
            self.llm = None

    def _get_cache_key(self, prompt: str, params: Dict) -> str:
        """生成缓存key"""
        content = f"{prompt}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cache(self):
        """加载缓存"""
        cache_file = os.path.join(self.cache_dir, "cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)

    def _save_cache(self):
        """保存缓存"""
        if self.cache_enabled:
            cache_file = os.path.join(self.cache_dir, "cache.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 512) -> str:
        """生成文本"""
        params = {"temperature": temperature, "max_tokens": max_tokens}
        cache_key = self._get_cache_key(prompt, params)

        # 检查缓存
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]

        # 调用LLM
        if self.llm is None:
            output = f"[Mock output for: {prompt[:50]}...]"
        else:
            sampling_params = self.SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95
            )
            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0].outputs[0].text

        # 保存缓存
        if self.cache_enabled:
            self.cache[cache_key] = output
            self._save_cache()

        return output

    def count_tokens(self, text: str) -> int:
        """估算token数量"""
        return len(text) // 4  # 简单估算
