import argparse
import json
import os
import socket
import subprocess
import sys
import traceback

_ADAPTER_NAMES = ("proposer", "controller", "critic", "verifier")
_ROLE_TO_ADAPTER = {
    "proposer": "proposer",
    "controller": "controller",
    "critic": "critic",
    "verifier": "verifier",
}


def _log(message: str):
    print(f"[vllm-worker] {message}", file=sys.stderr, flush=True)


def _nvidia_smi_lines():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return ["nvidia-smi not found"]
    output = result.stdout.strip() or result.stderr.strip() or "<no output>"
    return output.splitlines()


class VLLMWorker:
    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        # V0/V1 引擎选择：
        #   "0" = 强制 V0（默认，与训练机 vllm 0.9.2 行为一致）
        #   "1" = 强制 V1（vLLM ≥0.10 已删 V0，高版本镜像必须用这个）
        #   "auto" = 不设环境变量，交由 vLLM 自己选
        # 历史上硬编码 "0" 是因为当时 V1 不稳定；它不是本质需求。
        # 真正的兼容风险只在 logprobs 返回结构（见 _extract_output）。
        if args.vllm_use_v1 != "auto":
            os.environ["VLLM_USE_V1"] = args.vllm_use_v1
        else:
            os.environ.pop("VLLM_USE_V1", None)

        # Import CUDA/vLLM only after CUDA_VISIBLE_DEVICES is fixed for this process.
        import torch
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.torch = torch
        self.SamplingParams = SamplingParams
        self.LoRARequest = LoRARequest
        self.max_tokens = args.max_tokens
        self.lora_version = 0
        self.adapters = {}

        try:
            import vllm as _vllm
            self.vllm_version = getattr(_vllm, "__version__", "unknown")
        except Exception:
            self.vllm_version = "unknown"

        _log(
            f"starting pid={os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"torch_device_count={torch.cuda.device_count()} "
            f"vllm={self.vllm_version} VLLM_USE_V1={os.environ.get('VLLM_USE_V1', '<unset>')}"
        )
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _log(f"logical cuda:0 name={torch.cuda.get_device_name(0)}")
        for line in _nvidia_smi_lines():
            _log(f"nvidia-smi {line}")

        self.llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=16,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            # enforce_eager 禁用 CUDA graph。V0 时代损失有限，但 V1 把 piecewise
            # CUDA graph 当核心优化，小 batch + 中等生成长度（本项目典型工况）
            # 下 kernel launch 开销占比高，关掉 graph 可能慢数倍。默认保持 True
            # 与历史行为一致；用 --no-enforce-eager 开启 graph（额外占几 GB 显存，
            # 且首次捕获有建图耗时）。vLLM 官方支持 enable_lora 与 CUDA graph 共存。
            enforce_eager=args.enforce_eager,
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend="mp",
        )
        _log(f"vLLM initialized (enforce_eager={args.enforce_eager})")

    def ping(self):
        device_name = None
        if self.torch.cuda.is_available() and self.torch.cuda.device_count() > 0:
            device_name = self.torch.cuda.get_device_name(0)
        return {
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_device_count": self.torch.cuda.device_count(),
            "device_name": device_name,
            "lora_version": self.lora_version,
            "vllm_version": self.vllm_version,
            "vllm_use_v1": os.environ.get("VLLM_USE_V1", "<unset>"),
        }

    def sync_lora(self, manifest):
        self.lora_version = manifest["version"]
        self.adapters = manifest["adapters"]
        _log(f"synced LoRA version={self.lora_version} adapters={sorted(self.adapters)}")
        return {"lora_version": self.lora_version, "adapters": sorted(self.adapters)}

    def _make_lora_req(self, role):
        adapter_name = _ROLE_TO_ADAPTER.get(role, "proposer")
        adapter = self.adapters.get(adapter_name)
        if adapter:
            return self.LoRARequest(
                lora_name=adapter["lora_name"],
                lora_int_id=adapter["lora_int_id"],
                lora_path=adapter["path"],
            )
        return None

    def _extract_output(self, out):
        """抽取 text / logprobs / token_ids。

        logprobs 结构在 V0 与 V1 下均为 list[dict[token_id -> Logprob]]，但 V1
        对未命中 token 的填充策略可能不同；这里的三分支兼容（命中 / 取
        已知最小值减 1 / 缺失记 0.0）对两个引擎都成立。old_lps 是 GRPO
        importance ratio 的分母，对齐错位会直接体现为首步 kl 不为 0。
        """
        log_probs = []
        for token_id, lp_dict in zip(out.token_ids, (out.logprobs or [])):
            if lp_dict is None:
                log_probs.append(0.0)
            elif token_id in lp_dict:
                log_probs.append(lp_dict[token_id].logprob)
            elif lp_dict:
                log_probs.append(min(lp.logprob for lp in lp_dict.values()) - 1.0)
            else:
                log_probs.append(0.0)
        return {"text": out.text, "log_probs": log_probs, "token_ids": list(out.token_ids)}

    def generate(self, role: str, prompt: str, temperature: float = 1.0):
        # Only the controller is a short routing turn. Proposer/Critic/Verifier
        # must continue after </interaction> so their answer/score fields exist.
        stop = ["</meta-plan>"] if role == "controller" else None
        params = self.SamplingParams(
            max_tokens=self.max_tokens, temperature=temperature, logprobs=20,
            stop=stop, include_stop_str_in_output=bool(stop),
        )
        outputs = self.llm.generate(
            [prompt], sampling_params=params,
            lora_request=self._make_lora_req(role), use_tqdm=False,
        )
        return self._extract_output(outputs[0].outputs[0])

    def generate_batch(self, requests: list):
        """requests: list of {"role": str, "prompt": str, "temperature": float}.
        temperature defaults to 1.0 when omitted (for training rollouts).
        Pass temperature=0.0 for greedy eval decoding.
        Returns list in same order."""
        inputs, lora_reqs, params_list = [], [], []
        for req in requests:
            role = req["role"]
            temp = req.get("temperature", 1.0)
            stop = ["</meta-plan>"] if role == "controller" else None
            inputs.append(req["prompt"])
            lora_reqs.append(self._make_lora_req(role))
            params_list.append(self.SamplingParams(
                max_tokens=self.max_tokens, temperature=temp, logprobs=20,
                stop=stop, include_stop_str_in_output=bool(stop),
            ))
        # vLLM generate accepts per-request SamplingParams and LoRARequest
        outputs = self.llm.generate(
            inputs, sampling_params=params_list,
            lora_request=lora_reqs, use_tqdm=False,
        )
        return [self._extract_output(o.outputs[0]) for o in outputs]


def _send(writer, payload):
    writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
    writer.flush()


def serve(sock, worker):
    sock.settimeout(None)
    reader = sock.makefile("r", encoding="utf-8", newline="\n")
    writer = sock.makefile("w", encoding="utf-8", newline="\n")
    _send(writer, {"type": "ready", "ok": True, "result": worker.ping()})
    for line in reader:
        if not line.strip():
            continue
        req = json.loads(line)
        req_id = req.get("id")
        op = req.get("op")
        try:
            if op == "ping":
                result = worker.ping()
            elif op == "sync_lora":
                result = worker.sync_lora(req["manifest"])
            elif op == "generate":
                result = worker.generate(req["role"], req["prompt"],
                                        req.get("temperature", 1.0))
            elif op == "generate_batch":
                result = worker.generate_batch(req["requests"])
            elif op == "shutdown":
                _send(writer, {"id": req_id, "ok": True, "result": {"shutdown": True}})
                return
            else:
                raise ValueError(f"unknown op: {op}")
            _send(writer, {"id": req_id, "ok": True, "result": result})
        except Exception as exc:
            _send(
                writer,
                {
                    "id": req_id,
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--gpu-memory-utilization", required=True, type=float)
    parser.add_argument("--max-model-len", required=True, type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-use-v1", default="0",
                        choices=["0", "1", "auto"],
                        help="0=强制V0（默认） 1=强制V1（vLLM≥0.10 必选） auto=交给 vLLM")
    parser.add_argument("--no-enforce-eager", dest="enforce_eager",
                        action="store_false", default=True,
                        help="开启 CUDA graph（V1 下测到显著提速，代价是额外显存）")
    return parser.parse_args()


def main():
    args = parse_args()
    sock = socket.create_connection((args.host, args.port), timeout=30)
    try:
        try:
            worker = VLLMWorker(args)
        except Exception as exc:
            writer = sock.makefile("w", encoding="utf-8", newline="\n")
            _send(
                writer,
                {
                    "type": "ready",
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
            raise
        serve(sock, worker)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
