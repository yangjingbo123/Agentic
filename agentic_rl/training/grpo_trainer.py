import numpy as np
import torch
import torch.nn.functional as F
import bitsandbytes as bnb

from agents.agentic_executor import AgenticExecutor
from llm.trainable_llm import ROLE_ADAPTER
from training.raca_adv import compute_raca_advantages


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: (seq, vocab), labels: (seq,) → log_probs: (seq,)"""
    return F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)


class GRPOAgenticTrainer:
    def __init__(self, model, tokenizer, config, vllm_engine=None):
        self.model         = model
        self.tokenizer     = tokenizer
        self.vllm_engine   = vllm_engine
        self.executor      = AgenticExecutor(model, tokenizer, config, vllm_engine=vllm_engine)
        # 必须收集全部四个 adapter 的 LoRA 参数，不能依赖 requires_grad。
        # PEFT set_adapter() 只把激活 adapter 的 requires_grad 设为 True，
        # 如果只收集 requires_grad=True 的参数，非激活 adapter 既不会被
        # optimizer.step() 更新，也不会被 zero_grad() 清零，梯度持续累加。
        self.optimizer     = bnb.optim.AdamW8bit(
            model.lora_parameters(),
            lr=config.get("lr", 1e-5),
        )
        self.clip_epsilon  = config.get("clip_epsilon", 0.2)
        self.n_samples     = config.get("n_samples", 8)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.ppo_epochs    = config.get("ppo_epochs", 1)
        self.raca_delta    = config.get("raca_delta", 1e-4)   # variance floor
        self.kl_coef       = config.get("kl_coef", 0.04)      # KL penalty coefficient

    # ── RACA v2 证据指标（§8）：从 rollout 行为统计，不受优势过滤影响 ──────

    @staticmethod
    def _interaction_metrics(batch_rollouts: list) -> dict:
        eps = [ep for group in batch_rollouts for ep in group]
        rounds = [m for ep in eps for m in ep.get("raca_round_meta", [])]
        out = {}
        if rounds:
            us = [1.0 if m["u"] else 0.0 for m in rounds]
            ps = [1.0 if m["p_primary"] else 0.0 for m in rounds]
            out["int_rate"]     = float(np.mean(us))
            out["forced_rate"]  = float(np.mean([m["forced"] for m in rounds]))
            out["gate_blocked"] = int(sum(m["gate_blocked"] for m in rounds))
            # 交互有效率：P(轮末修对 | 自发求助且 primary 错)
            eff = [m["p_end"] for m in rounds if m["u"] and not m["p_primary"]]
            if eff:
                out["int_effectiveness"] = float(np.mean(eff))
            # 选择性：corr(u, p_primary)，预期随训练负相关增强（错的时候才求助）
            if np.std(us) > 0 and np.std(ps) > 0:
                out["int_selectivity"] = float(np.corrcoef(us, ps)[0, 1])
        # stop 校准：P(correct | stop) vs P(correct | 耗尽轮次)
        stopped   = [ep["is_correct"] for ep in eps if ep.get("stopped")]
        exhausted = [ep["is_correct"] for ep in eps if not ep.get("stopped")]
        if stopped:
            out["stop_acc"] = float(np.mean(stopped))
        if exhausted:
            out["exhaust_acc"] = float(np.mean(exhausted))
        if eps:
            out["stop_rate"] = float(np.mean([1.0 if ep.get("stopped") else 0.0 for ep in eps]))
        return out

    # ── Training loop ────────────────────────────────────────────────────────

    def _count_valid_turns(self, episode: dict, per_turn_adv: dict) -> int:
        """Count turns that have both response tokens and a valid advantage."""
        vocab_size = self.model._model.config.vocab_size
        return sum(
            1 for msg in episode["messages"]
            if (msg.get("turn_id") in per_turn_adv
                and [t for t in msg.get("response_ids", []) if 0 <= t < vocab_size])
        )

    def update(self, batch_rollouts: list) -> dict:
        """RACA update.

        batch_rollouts: list of episode-groups, each group being the list of N
        rollouts for one question.  Shape: list[list[episode_dict]].
        """
        # ── compute RACA advantages per group ────────────────────────────────
        all_episodes:    list = []
        all_per_turn_adv: list = []
        n_groups      = 0   # question-groups eligible for advantage computation
        n_groups_kept = 0   # ...that produced at least one usable advantage

        # v2 证据指标：对全部 rollout 统计（含被优势过滤掉的 episode）
        int_metrics = self._interaction_metrics(batch_rollouts)

        for episode_group in batch_rollouts:
            if len(episode_group) < 2:
                continue
            n_groups += 1
            per_ep_adv = compute_raca_advantages(
                [ep.get("raca_turn_data", {}) for ep in episode_group],
                delta=self.raca_delta,
            )
            kept = 0
            for ep, adv in zip(episode_group, per_ep_adv):
                if adv:   # skip episodes where no turn got an advantage
                    all_episodes.append(ep)
                    all_per_turn_adv.append(adv)
                    kept += 1
            if kept:
                n_groups_kept += 1

        if not all_episodes:
            if self.vllm_engine is not None:
                self.vllm_engine.sync_lora(self.model)
            return {"loss": 0.0, "mean_reward": 0.0, "accuracy": 0.0,
                    "kl": 0.0, "skipped": True,
                    "groups_total": n_groups, "groups_kept": 0, **int_metrics}

        # Logging
        n_correct  = sum(ep["is_correct"] for ep in all_episodes)
        mean_acc   = n_correct / len(all_episodes)
        # Mean controller reward as a proxy for "reward" in logs
        mean_r = float(np.mean([
            sum(v["reward"] for v in ep.get("raca_turn_data", {}).values()
                if v["role"] == "controller")
            for ep in all_episodes
        ]))
        print(f"  [rollout] correct={n_correct}/{len(all_episodes)} "
              f"mean_ctrl_reward={mean_r:.3f} "
              f"groups={n_groups_kept}/{n_groups}", flush=True)

        # Pre-count total valid turns for normalization
        total_valid = max(
            sum(self._count_valid_turns(ep, adv)
                for ep, adv in zip(all_episodes, all_per_turn_adv)), 1
        )

        total_loss    = 0.0
        total_n_valid = 0
        total_kl      = 0.0

        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            did_backward  = False
            epoch_loss    = 0.0
            epoch_n_valid = 0
            epoch_kl      = 0.0

            for ep, adv in zip(all_episodes, all_per_turn_adv):
                ep_loss, n_valid, ep_kl = self._compute_loss(ep, adv, total_valid)
                if n_valid > 0:
                    did_backward = True
                epoch_loss    += ep_loss
                epoch_n_valid += n_valid
                epoch_kl      += ep_kl

            if did_backward:
                # 同理，裁剪全部 LoRA 参数的梯度，不依赖 requires_grad。
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.lora_parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                print(f"  [update] grad_norm={grad_norm:.4f} loss={epoch_loss:.4f} "
                      f"kl={epoch_kl / max(epoch_n_valid, 1):.6f}",
                      flush=True)
            else:
                print("  [update] SKIPPED: no valid gradients", flush=True)

            total_loss    += epoch_loss
            total_n_valid += epoch_n_valid
            total_kl      += epoch_kl

        if self.vllm_engine is not None:
            self.vllm_engine.sync_lora(self.model)

        return {
            "loss":        total_loss / max(total_n_valid, 1),
            "mean_reward": mean_r,
            "accuracy":    mean_acc,
            "kl":          total_kl / max(total_n_valid, 1),
            "groups_total": n_groups,
            "groups_kept":  n_groups_kept,
            **int_metrics,
        }

    # ── Per-turn PPO loss ────────────────────────────────────────────────────

    _LOG_RATIO_THRESHOLD = 50.0

    def _compute_loss(
        self,
        episode: dict,
        per_turn_adv: dict,   # turn_id → float advantage
        normalization: int,
    ) -> tuple:
        """Per-turn GRPO clip loss with KL penalty.

        Reference model = base model without LoRA (via as_ref()).
        KL ≈ mean(new_logprob - ref_logprob), penalises policy drift.
        Backward after each turn to keep peak memory minimal.
        per_turn_adv: maps turn_id → RACA advantage scalar.
        Returns (mean_loss, n_valid_turns, mean_kl).
        """
        device = next(p for p in self.model._model.parameters()).device

        # Guard: skip episode if LoRA weights have gone NaN/Inf.
        # Check all LoRA params, not just requires_grad ones — set_adapter()
        # during loss computation means only the active adapter has
        # requires_grad=True, but NaN could be in any adapter.
        for p in self.model.lora_parameters():
            if not torch.isfinite(p).all():
                print("  [loss] NaN/Inf in model weights, skipping episode", flush=True)
                return 0.0, 0, 0.0

        vocab_size   = self.model._model.config.vocab_size
        messages     = episode["messages"]
        all_old_lps  = torch.tensor(episode["log_probs"], dtype=torch.float32, device=device)
        all_old_lps  = torch.nan_to_num(all_old_lps, nan=0.0, posinf=0.0, neginf=0.0)
        all_turn_ids = torch.tensor(episode["turn_ids"], device=device)

        total_loss = 0.0
        total_kl   = 0.0
        n_valid    = 0

        for msg in messages:
            turn_id = msg.get("turn_id", 0)

            # Only process turns that have a RACA advantage assigned.
            advantage = per_turn_adv.get(turn_id)
            if advantage is None:
                continue

            role     = msg.get("role_name", "proposer")
            resp_ids = [t for t in msg.get("response_ids", []) if 0 <= t < vocab_size]
            if not resp_ids:
                continue

            mask    = (all_turn_ids == turn_id)
            old_lps = all_old_lps[mask]
            n_resp  = len(resp_ids)
            n_align = min(n_resp, old_lps.shape[0])
            if n_align == 0:
                continue

            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": msg["system"]},
                 {"role": "user",   "content": msg["user"]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            p_len = len(prompt_ids)
            if p_len == 0:
                continue

            input_ids   = torch.tensor(
                prompt_ids + resp_ids, dtype=torch.long, device=device
            ).unsqueeze(0)
            resp_labels = torch.tensor(resp_ids, dtype=torch.long, device=device)

            # Reference forward (base model, no LoRA, no grad)
            with self.model.as_ref():
                with torch.no_grad():
                    ref_logits = self.model._model(input_ids, use_cache=False).logits[0]
                    _ref_rl = ref_logits[p_len - 1: p_len + n_resp - 1].float().contiguous()
                    del ref_logits
            ref_lps = logprobs_from_logits(_ref_rl, resp_labels.to(_ref_rl.device))
            del _ref_rl

            # LoRA forward (active adapter, with grad)
            self.model._model.set_adapter(ROLE_ADAPTER.get(role, "proposer"))
            logits_new   = self.model._model(input_ids, use_cache=False).logits[0]
            _resp_logits = logits_new[p_len - 1: p_len + n_resp - 1].float().contiguous()
            del logits_new
            new_lps = logprobs_from_logits(_resp_logits, resp_labels.to(_resp_logits.device))
            del _resp_logits

            if not torch.isfinite(new_lps).all():
                continue

            old_lps_aligned = old_lps[:n_align].to(new_lps.device)
            log_ratio_tok   = new_lps[:n_align] - old_lps_aligned
            log_ratio       = log_ratio_tok.mean()

            if not torch.isfinite(log_ratio):
                continue
            if log_ratio.abs().item() > self._LOG_RATIO_THRESHOLD:
                print(f"  [loss] turn={turn_id} SKIP |log_ratio|={log_ratio.abs().item():.1f}",
                      flush=True)
                continue

            ratio_tok = torch.exp(log_ratio_tok)
            pg_loss = torch.maximum(
                -advantage * ratio_tok,
                -advantage * torch.clamp(ratio_tok, 1 - self.clip_epsilon,
                                                     1 + self.clip_epsilon),
            ).mean()

            # KL penalty (k3 估计器): exp(ref−new) − (ref−new) − 1 ≥ 0。
            # v1 用 k1 = mean(new − ref)：ref 是常量，梯度 = ∇mean(new_lps)，
            # 在 on-policy 采样下期望为零——只加噪声、无约束力。k3 的梯度
            # 方向正确地把策略拉向 base model（GRPO 标准做法）。
            # clamp 防 exp 溢出：delta>20 时 kl 已巨大，梯度方向不变。
            delta_lp = (ref_lps[:n_align].to(new_lps.device) - new_lps[:n_align]).clamp(max=20.0)
            kl = (torch.exp(delta_lp) - delta_lp - 1.0).mean()
            turn_loss = pg_loss + self.kl_coef * kl

            (turn_loss / normalization).backward()  # immediate backward
            total_loss += pg_loss.item()
            total_kl   += kl.item()
            n_valid    += 1

        if n_valid == 0:
            return 0.0, 0, 0.0
        return total_loss / n_valid, n_valid, total_kl / n_valid
