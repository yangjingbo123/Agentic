import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

_ADAPTER_NAMES = ("proposer", "controller", "critic", "verifier")


class VLLMInferenceEngine:
    """Parent-side client for a vLLM worker pinned to a separate CUDA process."""

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 512,
        gpu_memory_utilization: float = 0.45,
        max_model_len: int = 8192,
        vllm_gpu: str = None,
        tensor_parallel_size: int = 1,
        startup_timeout_s: float = 300,
        rpc_timeout_s: float = 600,
        vllm_use_v1: str = "0",
    ):
        if not vllm_gpu:
            raise ValueError("vllm_gpu must be set for the subprocess vLLM worker")

        self.model_path = model_path
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.vllm_gpu = str(vllm_gpu)
        self.startup_timeout_s = startup_timeout_s
        self.rpc_timeout_s = rpc_timeout_s
        self.vllm_use_v1 = str(vllm_use_v1)
        self._lock = threading.Lock()
        self._next_request_id = 0
        self._closed = False

        self.lora_dir = tempfile.mkdtemp(prefix="agentic_lora_")
        self._lora_paths = {name: os.path.join(self.lora_dir, name) for name in _ADAPTER_NAMES}
        for path in self._lora_paths.values():
            os.makedirs(path, exist_ok=True)
            for fname in os.listdir(model_path):
                if "token" in fname.lower():
                    shutil.copy2(os.path.join(model_path, fname), path)

        self._lora_version = 0
        self._lora_loaded = {name: False for name in _ADAPTER_NAMES}
        self._last_synced_root = self.lora_dir

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self._server.settimeout(startup_timeout_s)
        host, port = self._server.getsockname()

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.vllm_gpu
        # V0/V1 交由 worker 自己根据 --vllm-use-v1 设置（auto 时不设）；
        # 这里先清掉继承值，避免外层 shell 的 VLLM_USE_V1 默默生效。
        env.pop("VLLM_USE_V1", None)
        # V1 引擎的 ZMQ IPC socket 必须建在本地文件系统：TMPDIR 常被指向
        # OSS/NAS 挂载（防 /tmp 写满），而网络挂载不支持 socket，bind() 会报
        # ZMQError: Input/output error。未显式指定时回退 /tmp，不跟随 TMPDIR。
        env.setdefault("VLLM_RPC_BASE_PATH", "/tmp")
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "llm.vllm_worker",
            "--host",
            host,
            "--port",
            str(port),
            "--cuda-visible-devices",
            self.vllm_gpu,
            "--model-path",
            model_path,
            "--max-tokens",
            str(max_tokens),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-model-len",
            str(max_model_len),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--vllm-use-v1",
            self.vllm_use_v1,
        ]
        self._proc = subprocess.Popen(cmd, cwd=project_root, env=env, text=True)
        self._worker_cmd = cmd
        self._sock = None
        self._reader = None
        self._writer = None
        try:
            self._sock, _ = self._server.accept()
            self._sock.settimeout(rpc_timeout_s)
            self._reader = self._sock.makefile("r", encoding="utf-8", newline="\n")
            self._writer = self._sock.makefile("w", encoding="utf-8", newline="\n")
            ready = self._read_message(timeout_s=startup_timeout_s)
            if not ready.get("ok"):
                error = ready.get("error", {})
                raise RuntimeError(
                    f"vLLM worker failed during startup: {error.get('type')}: {error.get('message')}\n"
                    f"{error.get('traceback', '')}"
                )
            self.worker_info = ready.get("result", {})
            print(f"vLLM worker ready: {self.worker_info}", flush=True)
        except Exception:
            self.close(kill=True)
            raise
        finally:
            self._server.close()

        atexit.register(self.close)

    def _read_message(self, timeout_s=None):
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("timed out waiting for vLLM worker response")
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM worker exited with code {self._proc.returncode}; cmd={' '.join(self._worker_cmd)}"
                )
            line = self._reader.readline()
            if line:
                return json.loads(line)
            time.sleep(0.05)

    def _request(self, op: str, **payload):
        if self._closed:
            raise RuntimeError("vLLM worker is closed")
        with self._lock:
            self._next_request_id += 1
            req_id = self._next_request_id
            request = {"id": req_id, "op": op, **payload}
            try:
                self._writer.write(json.dumps(request, ensure_ascii=False) + "\n")
                self._writer.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(f"failed to send {op} to vLLM worker") from exc

            response = self._read_message(timeout_s=self.rpc_timeout_s)
            if response.get("id") != req_id:
                raise RuntimeError(f"vLLM worker response id mismatch: expected {req_id}, got {response}")
            if not response.get("ok"):
                error = response.get("error", {})
                raise RuntimeError(
                    f"vLLM worker {op} failed: {error.get('type')}: {error.get('message')}\n"
                    f"{error.get('traceback', '')}"
                )
            return response.get("result")

    def ping(self):
        return self._request("ping")

    def generate(self, role: str, prompt: str, temperature: float = 1.0) -> tuple[str, list[float], list[int]]:
        result = self._request("generate", role=role, prompt=prompt, temperature=temperature)
        return result["text"], result["log_probs"], result["token_ids"]

    def generate_batch(self, requests: list[dict]) -> list[tuple[str, list[float], list[int]]]:
        """requests: list of {"role": str, "prompt": str, "temperature": float}.
        temperature defaults to 1.0 when omitted (training rollout).
        Pass temperature=0.0 for greedy eval decoding.
        Returns list in same order."""
        results = self._request("generate_batch", requests=requests)
        return [(r["text"], r["log_probs"], r["token_ids"]) for r in results]

    def sync_lora(self, model):
        """Save adapters in the parent process, then hand the manifest to the worker."""
        with self._lock:
            self._lora_version += 1
            version = self._lora_version
            new_dir = tempfile.mkdtemp(prefix="agentic_lora_sync_")
            adapters = {}
            try:
                for i, adapter_name in enumerate(_ADAPTER_NAMES):
                    model._model.set_adapter(adapter_name)
                    model._model.save_pretrained(new_dir, selected_adapters=[adapter_name])
                    new_path = os.path.join(new_dir, adapter_name)
                    weight_file = os.path.join(new_path, "adapter_model.safetensors")
                    alt_weight_file = os.path.join(new_path, "adapter_model.bin")
                    if not os.path.isfile(weight_file) and not os.path.isfile(alt_weight_file):
                        raise RuntimeError(f"sync_lora: no weights saved for {adapter_name} at {new_path}")
                    adapters[adapter_name] = {
                        "path": new_path,
                        "lora_name": f"{adapter_name}_v{version}",
                        "lora_int_id": len(_ADAPTER_NAMES) * version + (i + 1),
                    }

                # Avoid re-entering _request's lock while preserving single-flight semantics.
                self._next_request_id += 1
                req_id = self._next_request_id
                request = {
                    "id": req_id,
                    "op": "sync_lora",
                    "manifest": {"version": version, "adapters": adapters},
                }
                self._writer.write(json.dumps(request, ensure_ascii=False) + "\n")
                self._writer.flush()
                response = self._read_message(timeout_s=self.rpc_timeout_s)
                if response.get("id") != req_id:
                    raise RuntimeError(f"vLLM worker response id mismatch: expected {req_id}, got {response}")
                if not response.get("ok"):
                    error = response.get("error", {})
                    raise RuntimeError(
                        f"vLLM worker sync_lora failed: {error.get('type')}: {error.get('message')}\n"
                        f"{error.get('traceback', '')}"
                    )

                old_root = self._last_synced_root
                self._lora_paths = {name: adapters[name]["path"] for name in _ADAPTER_NAMES}
                self._lora_loaded = {name: True for name in _ADAPTER_NAMES}
                self._last_synced_root = new_dir
                shutil.rmtree(old_root, ignore_errors=True)
                return response.get("result")
            except Exception:
                shutil.rmtree(new_dir, ignore_errors=True)
                raise

    def close(self, kill: bool = False):
        if self._closed:
            return
        self._closed = True
        try:
            if not kill and self._writer is not None and self._proc.poll() is None:
                try:
                    self._next_request_id += 1
                    req_id = self._next_request_id
                    self._writer.write(json.dumps({"id": req_id, "op": "shutdown"}) + "\n")
                    self._writer.flush()
                    self._read_message(timeout_s=5)
                except Exception:
                    pass
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=10)
        finally:
            try:
                if self._reader is not None:
                    self._reader.close()
            except Exception:
                pass
            try:
                if self._writer is not None:
                    self._writer.close()
            except Exception:
                pass
            try:
                if self._sock is not None:
                    self._sock.close()
            except Exception:
                pass
            try:
                if hasattr(self, "_server"):
                    self._server.close()
            except Exception:
                pass
            shutil.rmtree(self._last_synced_root, ignore_errors=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MultiVLLMEngine:
    """Two independent vLLM workers on separate GPUs; splits generate_batch across both."""

    def __init__(self, engines: list):
        self.engines = engines
        self._lora_loaded = engines[0]._lora_loaded

    def generate(self, role: str, prompt: str, temperature: float = 1.0):
        return self.engines[0].generate(role, prompt, temperature=temperature)

    def generate_batch(self, requests: list) -> list:
        if not requests:
            return []
        k = len(self.engines)
        # interleave: engine i handles requests[i::k]
        chunks = [requests[i::k] for i in range(k)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=k) as pool:
            futures = [pool.submit(eng.generate_batch, chunk)
                       for eng, chunk in zip(self.engines, chunks) if chunk]
            results = [f.result() for f in futures]
        # reassemble in original order
        merged = [None] * len(requests)
        for ei, chunk_results in enumerate(results):
            for j, r in enumerate(chunk_results):
                merged[ei + j * k] = r
        return merged

    def sync_lora(self, model):
        """Serialize sync_lora calls to avoid concurrent set_adapter/save_pretrained
        on the same PEFT model — those operations are not thread-safe.
        The save is fast (LoRA weights only); the per-worker RPC calls inside
        each eng.sync_lora are already serialized by eng._lock.
        """
        for eng in self.engines:
            eng.sync_lora(model)
        self._lora_loaded = self.engines[0]._lora_loaded

    def ping(self):
        return self.engines[0].ping()

    def close(self):
        for eng in self.engines:
            eng.close()
