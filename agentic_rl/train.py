import json
import os
import random
import subprocess

import numpy as np
import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


def load_dataset(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]


def resolve_vllm_cuda_visible_devices(cfg: DictConfig) -> str:
    explicit = cfg.agentic.get("vllm_cuda_visible_devices", None)
    if explicit not in (None, "", "null"):
        return str(explicit)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    slot = int(cfg.agentic.get("vllm_gpu_slot", 1))
    if not tokens:
        return str(slot)
    if slot >= len(tokens):
        raise ValueError(
            f"agentic.vllm_gpu_slot={slot} but CUDA_VISIBLE_DEVICES={visible!r} only has {len(tokens)} entries"
        )
    return tokens[slot]


def print_gpu_snapshot(label: str):
    print(f"\n[diag][gpu] ===== {label} =====", flush=True)
    print(
        f"[diag][gpu] pid={os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    print(
        f"[diag][gpu] torch.cuda.is_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()}",
        flush=True,
    )
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(idx) / 1024**3
            reserved = torch.cuda.memory_reserved(idx) / 1024**3
            print(
                f"[diag][gpu] logical cuda:{idx} name={torch.cuda.get_device_name(idx)} "
                f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB",
                flush=True,
            )

    commands = [
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        except FileNotFoundError:
            print("[diag][gpu] nvidia-smi not found", flush=True)
            break
        print(f"[diag][gpu] $ {' '.join(cmd)}", flush=True)
        output = result.stdout.strip() or result.stderr.strip() or "<no output>"
        for line in output.splitlines():
            print(f"[diag][gpu] {line}", flush=True)
    print(f"[diag][gpu] ===== end {label} =====\n", flush=True)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    wandb.init(
        project="agentic-rl",
        name=cfg.exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    from llm.trainable_llm import load_trainable_models
    from training.grpo_trainer import GRPOAgenticTrainer

    model, tokenizer = load_trainable_models(
        cfg.llm.model_path,
        sft_checkpoint=cfg.sft_checkpoint if cfg.sft_checkpoint else None,
    )
    print_gpu_snapshot("after train model load")

    vllm_engine = None
    if not cfg.no_vllm:
        from llm.vllm_engine import VLLMInferenceEngine, MultiVLLMEngine
        vllm_cuda_visible_devices = resolve_vllm_cuda_visible_devices(cfg)
        num_workers = cfg.agentic.get("vllm_num_workers", 1)
        gpu_util = cfg.agentic.get("vllm_gpu_memory_utilization", 0.65)
        common = dict(
            max_tokens=cfg.agentic.max_tokens,
            gpu_memory_utilization=gpu_util,
            max_model_len=cfg.agentic.get("vllm_max_model_len", 4096),
            startup_timeout_s=cfg.agentic.get("vllm_start_timeout_s", 300),
            rpc_timeout_s=cfg.agentic.get("vllm_rpc_timeout_s", 600),
        )
        if num_workers > 1:
            _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            _tokens = [t.strip() for t in _visible.split(",") if t.strip()]
            _slot = int(cfg.agentic.get("vllm_gpu_slot", 1))
            gpus = _tokens[_slot:_slot + num_workers] if _tokens else [str(_slot + i) for i in range(num_workers)]
            print(f"Initializing {num_workers} vLLM workers on GPUs {gpus}...", flush=True)
            engines = [VLLMInferenceEngine(cfg.llm.model_path, vllm_gpu=g, **common) for g in gpus]
            vllm_engine = MultiVLLMEngine(engines)
        else:
            print(f"Initializing vLLM worker with CUDA_VISIBLE_DEVICES={vllm_cuda_visible_devices}...", flush=True)
            vllm_engine = VLLMInferenceEngine(cfg.llm.model_path, vllm_gpu=vllm_cuda_visible_devices, **common)
        print(f"vLLM worker ready: {vllm_engine.ping()}", flush=True)
        print_gpu_snapshot("after vLLM worker init")
        # sync SFT weights into vLLM before first rollout
        _test_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
        model._model.set_adapter("controller")
        with torch.no_grad():
            _logits_before = model._model(_test_ids, use_cache=False).logits[0].float()
        print(f"[diag] logits_finite BEFORE sync_lora: {torch.isfinite(_logits_before).all().item()}", flush=True)
        del _logits_before, _test_ids

        vllm_engine.sync_lora(model)
        print("vLLM synced with SFT checkpoint.", flush=True)

        _test_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
        model._model.set_adapter("controller")
        with torch.no_grad():
            _logits_after = model._model(_test_ids, use_cache=False).logits[0].float()
        print(f"[diag] logits_finite AFTER sync_lora: {torch.isfinite(_logits_after).all().item()}", flush=True)
        del _logits_after, _test_ids

    trainer = GRPOAgenticTrainer(model, tokenizer, OmegaConf.to_container(cfg.agentic), vllm_engine=vllm_engine)

    dataset = load_dataset(cfg.data.train_path)
    eval_dataset = load_dataset(cfg.data.test_path) if cfg.data.get("test_path") else []
    eval_freq = cfg.agentic.get("eval_freq", 20)
    eval_samples = cfg.agentic.get("eval_samples", 100)
    from agents.agentic_executor import AgenticExecutor
    eval_executor = AgenticExecutor(
        model, tokenizer, OmegaConf.to_container(cfg.agentic),
        vllm_engine=vllm_engine, eval_mode=True,
    )

    # fixed eval subset — Level 5 only (hardest tier), first N items, same across all experiments
    # Level 5 problems are where multi-agent collaboration has the most impact;
    # easier levels inflate baseline accuracy and reduce discrimination.
    _eval_items = [it for it in eval_dataset if it.get("level") == "Level 5"][:eval_samples] if eval_dataset else []
    if _eval_items:
        print(f"Eval subset: {len(_eval_items)} Level-5 items (from {len(eval_dataset)} total)", flush=True)

    def run_eval(step):
        if not _eval_items or vllm_engine is None:
            return
        episodes = eval_executor.run_episodes_batch(
            [it["question"] for it in _eval_items],
            [it["answer"]   for it in _eval_items],
        )
        acc = sum(ep["is_correct"] for ep in episodes) / len(episodes)
        # reward_mean: mean total RACA reward per episode (sum of all turn rewards)
        reward_mean = float(np.mean([
            sum(v["reward"] for v in ep.get("raca_turn_data", {}).values())
            for ep in episodes
        ]))
        # avg_turns: mean number of turns per episode
        avg_turns = float(np.mean([
            len(ep.get("raca_turn_data", {})) for ep in episodes
        ]))
        # RACA v2 行为指标（greedy、无 ε 注入，反映学到的策略本身）
        _rounds = [m for ep in episodes for m in ep.get("raca_round_meta", [])]
        eval_int_rate  = float(np.mean([m["u"] for m in _rounds])) if _rounds else 0.0
        eval_stop_rate = float(np.mean([1.0 if ep.get("stopped") else 0.0 for ep in episodes]))
        print(f"  [eval] step={step} eval_acc={acc:.3f} reward={reward_mean:.3f} "
              f"avg_turns={avg_turns:.1f} int_rate={eval_int_rate:.2f} "
              f"stop_rate={eval_stop_rate:.2f} (n={len(_eval_items)})", flush=True)
        wandb.log({"eval_accuracy": acc, "eval_reward": reward_mean,
                   "eval_avg_turns": avg_turns, "eval_int_rate": eval_int_rate,
                   "eval_stop_rate": eval_stop_rate}, step=step)
    batch_size = cfg.agentic.batch_size
    max_steps = cfg.agentic.get("max_steps", 500)
    print(f"Dataset: {len(dataset)} items, batch_size={batch_size}, max_steps={max_steps}", flush=True)

    ckpt_dir = f"checkpoints/rl-{cfg.exp_name}"
    save_freq = cfg.agentic.get("save_freq", 50)

    # resume from checkpoint if exists
    step = 0
    resume_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(resume_path):
        with open(resume_path) as f:
            state = json.load(f)
        step = state["step"]
        import safetensors.torch as st
        from llm.trainable_llm import ROLE_ADAPTER
        for adapter_name in ROLE_ADAPTER.values():
            w = st.load_file(os.path.join(ckpt_dir, f"{adapter_name}", "adapter_model.safetensors"))
            model._model.set_adapter(adapter_name)
            loaded = 0
            for name, param in model._model.named_parameters():
                if "lora_" not in name:
                    continue
                src = name.replace(f".{adapter_name}.", ".")
                if src in w:
                    param.data.copy_(w[src].to(param.device))
                    loaded += 1
            print(f"  [resume] {adapter_name}: loaded {loaded} params", flush=True)
        print(f"Resumed from step={step}", flush=True)

    # infinite dataloader — shuffle and cycle, no epoch concept
    data_pool = dataset[:]
    random.shuffle(data_pool)
    pool_idx = 0

    # Ensure vLLM uses LoRA weights from the start, not the base model
    if vllm_engine is not None and not any(vllm_engine._lora_loaded.values()):
        vllm_engine.sync_lora(model)
        print("Initial vLLM LoRA sync done.", flush=True)

    if cfg.agentic.get("val_before_train", False):
        run_eval(0)

    # A batch whose groups all failed the variance filter applies no gradient, so
    # it must not be charged to the step budget. That removes the unconditional
    # `step += 1` that used to guarantee termination, hence the livelock guard.
    skipped_batches       = 0
    consecutive_skips     = 0
    max_consecutive_skips = int(cfg.agentic.get("max_consecutive_skips", 50))

    def note_skip(reason: str):
        """Record a batch that produced no gradient; abort if nothing is learnable."""
        nonlocal skipped_batches, consecutive_skips
        skipped_batches   += 1
        consecutive_skips += 1
        print(f"  [skip] {reason} — step stays {step}, "
              f"skipped_total={skipped_batches}", flush=True)
        if consecutive_skips >= max_consecutive_skips:
            raise RuntimeError(
                f"{consecutive_skips} consecutive batches produced no gradient "
                f"(last reason: {reason}). Training cannot progress — every rollout "
                f"in every group scored identically. Check the reward function and "
                f"rollout diversity (temperature, n_samples)."
            )

    while step < max_steps:
        # take next batch, reshuffle when exhausted
        if pool_idx + batch_size > len(data_pool):
            random.shuffle(data_pool)
            pool_idx = 0
        batch = data_pool[pool_idx: pool_idx + batch_size]
        pool_idx += batch_size

        # RACA v2: ε 强制注入概率线性衰减（冷启动保护，§2.4）
        _eps0 = float(cfg.agentic.get("eps_force_init", 0.3))
        _epsm = float(cfg.agentic.get("eps_force_min", 0.05))
        eps_force = max(_epsm, _eps0 - (_eps0 - _epsm) * step / max(max_steps, 1))

        # Run all batch_size * n_samples rollouts in one batched vLLM call
        questions = [item["question"] for item in batch]
        answers   = [item["answer"]   for item in batch]
        n_s = trainer.n_samples
        all_q = questions * n_s
        all_a = answers * n_s
        if vllm_engine is not None and hasattr(vllm_engine, "engines") and len(vllm_engine.engines) > 1:
            from concurrent.futures import ThreadPoolExecutor as _TPE
            _engines = vllm_engine.engines
            _k = len(_engines)
            _agentic_cfg = OmegaConf.to_container(cfg.agentic)
            _chunks_q = [all_q[i::_k] for i in range(_k)]
            _chunks_a = [all_a[i::_k] for i in range(_k)]
            def _run_chunk(eng, qs, ans):
                ex = AgenticExecutor(model, tokenizer, _agentic_cfg, vllm_engine=eng)
                return ex.run_episodes_batch(qs, ans, eps_force=eps_force)
            with _TPE(max_workers=_k) as _pool:
                _futures = [_pool.submit(_run_chunk, eng, qs, ans)
                            for eng, qs, ans in zip(_engines, _chunks_q, _chunks_a)]
                _chunks_out = [f.result() for f in _futures]
            all_eps = [None] * len(all_q)
            for _ei, _res in enumerate(_chunks_out):
                for _j, _ep in enumerate(_res):
                    all_eps[_ei + _j * _k] = _ep
        else:
            all_eps = trainer.executor.run_episodes_batch(all_q, all_a, eps_force=eps_force)
        # group by question — each group is the list of N rollouts for that question
        batch_rollouts = []
        for qi in range(len(batch)):
            eps = [all_eps[qi + len(batch) * s] for s in range(n_s)]
            valid = [ep for ep in eps if ep is not None and ep.get("raca_turn_data")]
            if len(valid) >= 2:   # need at least 2 for anchor-group comparison
                batch_rollouts.append(valid)
        if not batch_rollouts:
            note_skip("no group had >=2 valid episodes")
            continue

        try:
            stats = trainer.update(batch_rollouts)
        except Exception:
            import traceback; traceback.print_exc()
            raise

        _g_kept, _g_total = stats.get("groups_kept", 0), stats.get("groups_total", 0)
        if stats.get("skipped", False):
            # Reusing the current step index for wandb would also clash with the
            # point already logged there, so skip the log entirely and fold the
            # running total into the next real step.
            note_skip(f"no usable groups (0/{_g_total})")
            continue

        consecutive_skips = 0
        step += 1
        # int_effectiveness/selectivity 可能缺失（无自发求助样本时），缺失时打 "--"
        # 而非 0，避免与真实零值混淆（v2.0 日志正因缺这几项无法事后验证 q）。
        _eff = stats.get("int_effectiveness")
        _sel = stats.get("int_selectivity")
        print(
            f"step={step} reward={stats['mean_reward']:.3f} acc={stats['accuracy']:.2f} "
            f"loss={stats['loss']:.4f} kl={stats['kl']:.4f} "
            f"ent={stats.get('entropy', 0.0):.3f} "
            f"clip={stats.get('clip_frac', 0.0):.3f} "
            f"len={stats.get('resp_len', 0.0):.0f} "
            f"groups={_g_kept}/{_g_total} "
            f"int_rate={stats.get('int_rate', 0.0):.2f} "
            f"eff={'--' if _eff is None else f'{_eff:.2f}'} "
            f"sel={'--' if _sel is None else f'{_sel:+.2f}'} "
            f"parse={stats.get('parse_rate', 1.0):.2f} "
            f"gate={stats.get('gate_blocked', 0)} "
            f"fnl={stats.get('funnel_flag', 0)}/{stats.get('funnel_corr', 0)}"
            f"/{stats.get('funnel_flip', 0)} "
            f"stop_rate={stats.get('stop_rate', 0.0):.2f} eps={eps_force:.2f}",
            flush=True,
        )
        log_data = {
            "reward":          stats["mean_reward"],
            "accuracy":        stats["accuracy"],
            "loss":            stats["loss"],
            "kl":              stats["kl"],
            "skipped_batches": skipped_batches,
            "eps_force":       eps_force,
        }
        # 策略健康（熵坍塌/漂移/权重脱节）+ 信号质量（梯度还能用多久）
        # + 行为（reward hacking 侦测）+ RACA v2 证据指标（§8），有则上报
        for _mk in ("entropy", "clip_frac", "ratio_mean", "ratio_max", "resp_len",
                    "all_pass_frac", "all_fail_frac", "group_reward_std",
                    "parse_rate",
                    "int_rate", "int_effectiveness", "int_selectivity",
                    "forced_rate", "stop_rate", "stop_acc", "exhaust_acc",
                    "gate_blocked",
                    "funnel_flag", "funnel_corr", "funnel_flip"):
            if _mk in stats:
                log_data[_mk] = stats[_mk]
        if _g_total:
            # Fraction of question-groups that produced a usable advantage. A
            # sustained drop means rollouts are collapsing to identical rewards
            # and the batch is mostly dead weight.
            log_data["group_keep_rate"] = _g_kept / _g_total
        wandb.log(log_data, step=step)

        if eval_freq > 0 and step % eval_freq == 0:
            run_eval(step)

        if step % save_freq == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            with open(resume_path, "w") as f:
                json.dump({"step": step}, f)
            print(f"  [ckpt] saved step={step} to {ckpt_dir}", flush=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    with open(resume_path, "w") as f:
        json.dump({"step": step}, f)
    print(f"Saved to {ckpt_dir}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
