import json
import os
import random
import subprocess
import time

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
            # "0"=强制V0（默认，同训练机 vllm 0.9.2）"1"=强制V1（高版本镜像
            # 必选，vLLM ≥0.10 已删 V0）"auto"=交给 vLLM。首次切 V1 请用 SMOKE
            # 作业验收：首步 kl 应 ≈0，不为 0 则 logprobs 对齐有差异。
            vllm_use_v1=str(cfg.agentic.get("vllm_use_v1", "0")),
            # enforce_eager=False 开启 CUDA graph。V1 下 kernel launch 开销占比高，
            # 实测每步耗时可差数倍；代价是每 worker 额外几 GB 显存与首次建图耗时。
            enforce_eager=bool(cfg.agentic.get("vllm_enforce_eager", True)),
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

    # ── 尺子精度（这一段是所有结论的前提，n 与 K 不要随手调小）─────────────
    # 单点 se = sqrt(p(1-p)/n)：n=300、p≈0.85 时是 2.06 点，跨 run 比较的差值
    # 2·se ≈ 5.8 点。而交互纠错在这个操作点的理论收益上限只有 1.7 点——等于
    # 用比效应大三倍的尺子去量，任何机制改动的结果都无法验证。
    # n=1000（Level-5 池共 1324 题，够取）+ 末 K 点平均把 2·se 压到 1.43 点。
    _EVAL_TAIL_K = 5
    _eval_hist = []

    def run_eval(step):
        if not _eval_items or vllm_engine is None:
            return
        episodes = eval_executor.run_episodes_batch(
            [it["question"] for it in _eval_items],
            [it["answer"]   for it in _eval_items],
        )
        acc = sum(ep["is_correct"] for ep in episodes) / len(episodes)
        # 与历史 n=300 数字对齐的切片。run_episodes_batch 按输入顺序返回，故
        # episodes[:300] 恒等于扩容前那个固定子集。扩容后 acc 的绝对值不再与
        # 历史可比（新增的 700 题从没测过，难度分布不保证一致），这一路只为
        # 对齐旧数字，不用于决策——决策看 acc_tail。
        _n300 = min(300, len(episodes))
        acc_300 = sum(ep["is_correct"] for ep in episodes[:_n300]) / _n300
        # 末 K 点平均。单点最大值带 max-of-k 选择偏差：v2 基线 6 个 eval 点里
        # 挑 max 比自己均值高 2.42 点，与模拟预期的 2.57 点吻合——报 max 等于
        # 把选择噪声当成能力。看趋势和下结论都用这一路。
        _eval_hist.append(acc)
        _tail = _eval_hist[-_EVAL_TAIL_K:]
        acc_tail = float(np.mean(_tail))
        # §19.3 判据①：多种计票各自的 acc。两个 Δ 的方向都恒定、与当前生产开关
        # 无关——v3.1 跑 uniform 时仍能持续监测加权投票能不能重开，同理第十轮
        # 起 `correction_in_vote` 无论开关如何都能读到"修正票进池的净收益"。
        #   d_vote = weighted − uniform （钉在「修正票排除」臂）
        #   d_corr = 进池 − 不进池      （钉在 uniform 臂）
        def _eacc(key):
            return sum(1 for ep in episodes
                       if ep.get(key, ep["is_correct"])) / len(episodes)
        acc_uni = _eacc("is_correct_uni_excl")
        acc_wt  = _eacc("is_correct_wt_excl")
        acc_uni_incl = _eacc("is_correct_uni_incl")
        d_vote = acc_wt - acc_uni
        d_corr = acc_uni_incl - acc_uni
        # reward_mean: mean total RACA reward per episode (sum of all turn rewards)
        reward_mean = float(np.mean([
            sum(v["reward"] for v in ep.get("raca_turn_data", {}).values())
            for ep in episodes
        ]))
        # avg_turns: mean number of turns per episode
        avg_turns = float(np.mean([
            len(ep.get("raca_turn_data", {})) for ep in episodes
        ]))
        # 票池埋点：deg（n_distinct≤1 的比例）是判决量。eval 走 greedy、且同一
        # episode 的后轮能看见前轮答案，deg 高就说明池子塌成了重复票、acc 完全
        # 由 greedy 单票贡献，聚合收益（offline 实测 +10 点）一点没吃到；deg 低而
        # margin 高则相反，聚合已经在出力。两种情形对「加大 k 是不是杠杆」的答案相反。
        pool_votes = float(np.mean([ep.get("n_votes", 0) for ep in episodes]))
        pool_dist = float(np.mean([ep.get("n_distinct", 0) for ep in episodes]))
        pool_deg = float(np.mean([1.0 if ep.get("n_distinct", 0) <= 1 else 0.0
                                  for ep in episodes]))
        pool_marg = float(np.mean([ep.get("vote_margin", 0.0) for ep in episodes]))
        # RACA v2 行为指标（greedy 解码；ε 注入已由 eval_mode 关掉）
        _rounds = [m for ep in episodes for m in ep.get("raca_round_meta", [])]
        eval_int_rate  = float(np.mean([m["u"] for m in _rounds])) if _rounds else 0.0
        eval_stop_rate = float(np.mean([1.0 if ep.get("stopped") else 0.0 for ep in episodes]))
        # forced/gate 两列必须和 int_rate 一起看。eval_mode 只关了 ε 注入，
        # **闸门注入没关也不该关**（controller 想停但黑板没分数 → 强注 verifier
        # 解锁，线上一样触发）。而 int_rate 量的是 `u`（自发），闸门注入记在
        # `forced` 上——只印 int_rate 的话，「策略没求助」和「策略没求助、机制
        # 替它求助了」在日志上都是 0.00，可这两种情况对「ε 要不要继续加」的答案
        # 正好相反。fgate 高而 int 低 = 交互全是机制撑的，别把它读成学会了求助。
        eval_forced_rate = float(np.mean([m["forced"] for m in _rounds])) if _rounds else 0.0
        eval_gate_rate   = float(np.mean([m["gate_blocked"] for m in _rounds])) if _rounds else 0.0
        # eval 侧此前完全没报过修正漏斗（只有训练侧有）。两边采样条件不同
        # （greedy 单样本 vs temperature 多份），所以训练侧的 flip/unflip 不能外推。
        ev_flag = int(sum(m.get("n_flagged", 0) for m in _rounds))
        ev_corr = int(sum(m.get("n_corrections", 0) for m in _rounds))
        ev_flip = int(sum(1 for m in _rounds if m.get("flip")))
        ev_unflip = int(sum(1 for m in _rounds if m.get("unflip")))
        print(f"  [eval] step={step} eval_acc={acc:.3f} "
              f"acc_tail{len(_tail)}={acc_tail:.3f} acc300={acc_300:.3f} "
              f"acc_uniform={acc_uni:.3f} acc_weighted={acc_wt:.3f} "
              f"acc_corr_in={acc_uni_incl:.3f} "
              f"d_vote={d_vote:+.3f} d_corr={d_corr:+.3f} "
              f"fnl={ev_flag}/{ev_corr}/{ev_flip}/{ev_unflip} "
              f"reward={reward_mean:.3f} "
              f"avg_turns={avg_turns:.1f} int_rate={eval_int_rate:.2f} "
              f"forced={eval_forced_rate:.2f} gate={eval_gate_rate:.2f} "
              f"stop_rate={eval_stop_rate:.2f} "
              f"pool={pool_votes:.1f}/dist={pool_dist:.2f}/deg={pool_deg:.2f}"
              f"/marg={pool_marg:.2f} (n={len(_eval_items)})", flush=True)
        wandb.log({"eval_accuracy": acc, "eval_accuracy_tail": acc_tail,
                   "eval_accuracy_n300": acc_300,
                   "eval_accuracy_uniform": acc_uni,
                   "eval_accuracy_weighted": acc_wt,
                   "eval_accuracy_corr_in": acc_uni_incl,
                   "eval_vote_gain": d_vote, "eval_corr_gain": d_corr,
                   "eval_funnel_flag": ev_flag, "eval_funnel_corr": ev_corr,
                   "eval_funnel_flip": ev_flip, "eval_funnel_unflip": ev_unflip,
                   "eval_reward": reward_mean,
                   "eval_avg_turns": avg_turns, "eval_int_rate": eval_int_rate,
                   "eval_forced_rate": eval_forced_rate,
                   "eval_gate_rate": eval_gate_rate,
                   "eval_stop_rate": eval_stop_rate,
                   "eval_pool_votes": pool_votes, "eval_pool_distinct": pool_dist,
                   "eval_pool_degenerate": pool_deg,
                   "eval_vote_margin": pool_marg}, step=step)
    batch_size = cfg.agentic.batch_size
    max_steps = cfg.agentic.get("max_steps", 500)
    print(f"Dataset: {len(dataset)} items, batch_size={batch_size}, max_steps={max_steps}", flush=True)

    # checkpoint 目录：本地默认相对路径；Primus 等平台用 ckpt_dir=... 指向持久化
    # 挂载（抢占重排后 resume 依赖同一路径下的 trainer_state.json）。
    ckpt_dir = cfg.get("ckpt_dir") or f"checkpoints/rl-{cfg.exp_name}"
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
        _t_roll0 = time.time()
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
        _dt_rollout = time.time() - _t_roll0
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

        _t_train0 = time.time()
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
        # int_effectiveness/selectivity/q_forced 可能缺失（无对应样本时），
        # 缺失时打 "--" 而非 0，避免与真实零值混淆。
        _eff = stats.get("int_effectiveness")
        _sel = stats.get("int_selectivity")
        _qf  = stats.get("q_forced")
        _tc  = stats.get("int_critic_share")
        # v3.1：gate→unlocked 展示“拦下多少 → 其中多少被解锁”；两者应接近（本轮
        # 交互链已自带分数的不需解锁）。clip_prompt 非 0 即告警：仍有文本无界点。
        _n_clip = getattr(trainer.executor, "n_prompt_clipped", 0)
        # selfT 非 0 说明 proposer 在写 `target: proposer`（自指）。已被归一为
        # none 且不再计入 int_rate/r_int，但计数本身是 prompt 是否讲清楚的信号。
        _n_self = getattr(trainer.executor, "n_self_target", 0)
        # 跳深分布（第十轮，`max_hops` 2→3 的验收指标）。第 3 跳**没有机制把守**，
        # 它要求修正后的 proposer 自己写出 `request verifier`；depth3 若接近 0，说明
        # 加预算没被用上，该照 critic 标错那条的样子做成机械触发。`3v` 单列"第 3 跳
        # 到达 verifier"的次数，因为只看总数分不出它到的是 verifier 还是又一次 critic。
        _hd = getattr(trainer.executor, "n_hop_depth", None) or {}
        _hop_s = "/".join(f"{d}:{_hd.get(d, 0)}" for d in (1, 2, 3) if _hd.get(d))
        if _hop_s and _hd.get("3:verifier"):
            _hop_s += f"(3v={_hd['3:verifier']})"
        print(
            f"step={step} t={_dt_rollout:.0f}+{time.time() - _t_train0:.0f}s "
            f"reward={stats['mean_reward']:.3f} acc={stats['accuracy']:.2f} "
            f"loss={stats['loss']:.4f} kl={stats['kl']:.4f} "
            f"ent={stats.get('entropy', 0.0):.3f} "
            f"clip={stats.get('clip_frac', 0.0):.3f} "
            f"len={stats.get('resp_len', 0.0):.0f} "
            f"groups={_g_kept}/{_g_total} "
            f"int_rate={stats.get('int_rate', 0.0):.2f} "
            f"eff={'--' if _eff is None else f'{_eff:.2f}'} "
            f"qF={'--' if _qf is None else f'{_qf:.2f}'} "
            f"tgtC={'--' if _tc is None else f'{_tc:.2f}'} "
            f"sel={'--' if _sel is None else f'{_sel:+.2f}'} "
            f"parse={stats.get('parse_rate', 1.0):.2f}"
            # 拆开报：nl = 无标签（走数字兜底 → 垃圾票），ea = 空答案（空串进票池）。
            # `parse` 是两者的合取，单看它分不出掉的是哪一种，而两种的修法不同。
            f"(nl{stats.get('no_label_rate', 0.0):.2f}"
            f"/ea{stats.get('empty_answer_rate', 0.0):.2f}) "
            f"gate={stats.get('gate_blocked', 0)}"
            f"→{getattr(trainer.executor, 'n_gate_unlocked', 0)} "
            f"fnl={stats.get('funnel_flag', 0)}/{stats.get('funnel_corr', 0)}"
            f"/{stats.get('funnel_flip', 0)}/{stats.get('funnel_unflip', 0)} "
            # dC = 训练侧 d_corr（修正票进池 − 不进池）。它是"修正票该不该进投票池"
            # 的直接判据，且**不需要真打开 correction_in_vote** 就有读数。
            f"dC={stats.get('d_corr', 0.0):+.3f} "
            f"dist={stats.get('pool_distinct', 0.0):.2f}"
            f"/deg={stats.get('pool_degenerate', 0.0):.2f}"
            f"/marg={stats.get('vote_margin', 0.0):.2f} "
            f"stop_rate={stats.get('stop_rate', 0.0):.2f} eps={eps_force:.2f}"
            + (f" clip_prompt={_n_clip}" if _n_clip else "")
            + (f" selfT={_n_self}" if _n_self else "")
            + (f" hop={_hop_s}" if _hop_s else ""),
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
                    "parse_rate", "no_label_rate", "empty_answer_rate",
                    "int_rate", "int_effectiveness", "int_selectivity",
                    "q_forced", "int_critic_share",
                    "forced_rate", "stop_rate", "stop_acc", "exhaust_acc",
                    "gate_blocked",
                    "funnel_flag", "funnel_corr", "funnel_flip", "funnel_unflip",
                    "acc_uni_excl", "acc_wt_excl", "acc_uni_incl", "acc_wt_incl",
                    "d_vote", "d_corr",
                    "pool_votes", "pool_distinct", "pool_degenerate",
                    "vote_margin"):
            if _mk in stats:
                log_data[_mk] = stats[_mk]
        log_data["gate_unlocked"] = getattr(trainer.executor, "n_gate_unlocked", 0)
        log_data["prompt_clipped"] = _n_clip
        log_data["self_target"] = _n_self
        # 跳深分布逐深度上报（键为 int，wandb 需要字符串名）。
        for _d in (1, 2, 3):
            log_data[f"hop_depth_{_d}"] = int(_hd.get(_d, 0))
        log_data["hop3_verifier"] = int(_hd.get("3:verifier", 0))
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
