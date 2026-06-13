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
        os.environ["VLLM_USE_V1"] = "0"

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

        _log(
            f"starting pid={os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"torch_device_count={torch.cuda.device_count()}"
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
            enforce_eager=True,
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend="mp",
        )
        _log("vLLM initialized")

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
        log_probs = []
        for token_id, lp_dict in zip(out.token_ids, (out.logprobs or [])):
            if token_id in lp_dict:
                log_probs.append(lp_dict[token_id].logprob)
            elif lp_dict:
                log_probs.append(min(lp.logprob for lp in lp_dict.values()) - 1.0)
            else:
                log_probs.append(0.0)
        return {"text": out.text, "log_probs": log_probs, "token_ids": list(out.token_ids)}

    def generate(self, role: str, prompt: str):
        # Only the controller is a short routing turn. Proposer/Critic/Verifier
        # must continue after </interaction> so their answer/score fields exist.
        stop = ["</meta-plan>"] if role == "controller" else None
        params = self.SamplingParams(
            max_tokens=self.max_tokens, temperature=1.0, logprobs=20,
            stop=stop, include_stop_str_in_output=bool(stop),
        )
        outputs = self.llm.generate(
            [prompt], sampling_params=params,
            lora_request=self._make_lora_req(role), use_tqdm=False,
        )
        return self._extract_output(outputs[0].outputs[0])

    def generate_batch(self, requests: list):
        """requests: list of {"role": str, "prompt": str}. Returns list in same order."""
        # Group by (role, stop) to share SamplingParams, but vLLM handles mixed lora fine.
        inputs, lora_reqs, params_list = [], [], []
        for req in requests:
            role = req["role"]
            stop = ["</meta-plan>"] if role == "controller" else None
            inputs.append(req["prompt"])
            lora_reqs.append(self._make_lora_req(role))
            params_list.append(self.SamplingParams(
                max_tokens=self.max_tokens, temperature=1.0, logprobs=20,
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
                result = worker.generate(req["role"], req["prompt"])
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
