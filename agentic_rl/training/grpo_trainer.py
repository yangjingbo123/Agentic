import numpy as np
import torch
import torch.nn.functional as F
import bitsandbytes as bnb

from agents.agentic_executor import AgenticExecutor
from llm.trainable_llm import ROLE_ADAPTER


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: (seq, vocab), labels: (seq,) → log_probs: (seq,)"""
    return F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)


class AdaptiveKLController:
    def __init__(self, init_kl_coef=0.1, target_kl=6.0, horizon=10000):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        self.value *= 1 + proportional_error * n_steps / self.horizon


class GRPOAgenticTrainer:
    def __init__(self, model, tokenizer, config, vllm_engine=None):
        self.model = model
        self.tokenizer = tokenizer
        self.vllm_engine = vllm_engine
        self.executor = AgenticExecutor(model, tokenizer, config, vllm_engine=vllm_engine)
        self.optimizer = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.get("lr", 1e-5),
        )
        self.clip_epsilon  = config.get("clip_epsilon", 0.02)
        self.n_samples     = config.get("n_samples", 8)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.ppo_epochs    = config.get("ppo_epochs", 1)
        self.kl_ctrl = AdaptiveKLController(
            init_kl_coef=config.get("kl_coef", 0.1),
            target_kl=config.get("target_kl", 6.0),
        )

    def collect_rollouts(self, question: str, correct_answer: str):
        """Run n_samples rollouts for one question in a single batched vLLM call."""
        episodes = self.executor.run_episodes_batch(
            [question] * self.n_samples, [correct_answer] * self.n_samples
        )
        ep_rewards = [sum(ep["rewards"]) for ep in episodes]
        valid_mask = [np.isfinite(r) for r in ep_rewards]
        episodes   = [ep for ep, v in zip(episodes, valid_mask) if v]
        ep_rewards = [r  for r,  v in zip(ep_rewards, valid_mask) if v]

        if not episodes:
            print("  [rollout] all episodes have nan rewards, skipping", flush=True)
            return None
        return episodes, ep_rewards

    def update(self, batch_rollouts: list) -> dict:
        """
        batch_rollouts: list of (episodes, ep_rewards) from collect_rollouts.
        GRPO: compute advantages within each group (same question), not across the batch.
        """
        all_episodes = []
        advantages   = []
        all_rewards  = []  # for logging only

        for episodes, ep_rewards in batch_rollouts:
            group_mean = np.mean(ep_rewards)
            group_std  = np.std(ep_rewards)
            # skip group if zero variance (all same reward) — no learning signal
            if group_std < 1e-6:
                group_adv = [0.0] * len(ep_rewards)
            else:
                group_adv = [(r - group_mean) / (group_std + 1e-8) for r in ep_rewards]
            all_episodes.extend(episodes)
            advantages.extend(group_adv)
            all_rewards.extend(ep_rewards)

        mean_r = np.mean(all_rewards)
        std_r  = np.std(all_rewards)
        n_correct = sum(ep["is_correct"] for ep in all_episodes)
        print(f"  [rollout] correct={n_correct}/{len(all_episodes)} reward_mean={mean_r:.3f} "
              f"reward_std={std_r:.4f}", flush=True)

        if not any(adv != 0.0 for adv in advantages):
            if self.vllm_engine is not None:
                self.vllm_engine.sync_lora(self.model)
            return {
                "loss": 0.0, "mean_reward": mean_r,
                "accuracy": float(np.mean([ep["is_correct"] for ep in all_episodes])),
                "kl": 0.0, "skipped": True,
            }

        total_loss    = 0.0
        total_kl      = 0.0
        total_n_valid = 0

        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            did_backward = False
            epoch_loss    = 0.0
            epoch_kl      = 0.0
            epoch_n_valid = 0

            for ep, adv in zip(all_episodes, advantages):
                ep_loss, kl, n_valid = self._compute_loss(ep, adv, len(all_episodes))
                if n_valid > 0:
                    did_backward = True
                epoch_loss    += ep_loss
                epoch_kl      += kl
                epoch_n_valid += n_valid

            if did_backward:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                mean_epoch_kl = epoch_kl / max(epoch_n_valid, 1)
                print(f"  [update] grad_norm={grad_norm:.4f} loss={epoch_loss:.4f} kl={mean_epoch_kl:.4f}", flush=True)
            else:
                print(f"  [update] SKIPPED: no valid gradients", flush=True)

            total_loss    += epoch_loss
            total_kl      += epoch_kl
            total_n_valid += epoch_n_valid

        if self.vllm_engine is not None:
            self.vllm_engine.sync_lora(self.model)

        mean_kl = total_kl / max(total_n_valid, 1)
        self.kl_ctrl.update(mean_kl, n_steps=1)

        return {
            "loss":        total_loss / (len(all_episodes) * self.ppo_epochs),
            "mean_reward": mean_r,
            "accuracy":    float(np.mean([ep["is_correct"] for ep in all_episodes])),
            "kl":          mean_kl,
        }

    _LOG_RATIO_THRESHOLD = 50.0

    def _compute_loss(self, episode: dict, episode_advantage: float, normalization: int) -> tuple:
        """Per-turn PPO loss: backward immediately after each turn to keep peak memory minimal."""
        device     = next(p for p in self.model._model.parameters()).device
        # guard: skip episode if LoRA weights have gone NaN/Inf
        for p in self.model._model.parameters():
            if p.requires_grad and not torch.isfinite(p).all():
                print("  [loss] NaN/Inf in model weights, skipping episode", flush=True)
                return 0.0, 0.0, 0
        vocab_size = self.model._model.config.vocab_size
        messages   = episode["messages"]
        all_old_lps  = torch.tensor(episode["log_probs"], dtype=torch.float32, device=device)
        all_old_lps  = torch.nan_to_num(all_old_lps, nan=0.0, posinf=0.0, neginf=0.0)
        all_turn_ids = torch.tensor(episode["turn_ids"], device=device)

        # --- Pass 1: pre-compute all ref log-probs in a single no_grad block ---
        ref_lps_cache = {}  # turn_id -> ref_lps tensor
        self.model._model.disable_adapter_layers()
        with torch.inference_mode():
            for msg in messages:
                turn_id  = msg.get("turn_id", 0)
                resp_ids = [t for t in msg.get("response_ids", []) if 0 <= t < vocab_size]
                if not resp_ids:
                    continue
                prompt_text = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": msg["system"]},
                     {"role": "user",   "content": msg["user"]}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                if not prompt_ids:
                    continue
                p_len = len(prompt_ids)
                input_ids = torch.tensor(prompt_ids + resp_ids, dtype=torch.long, device=device).unsqueeze(0)
                resp_labels = torch.tensor(resp_ids, dtype=torch.long, device=device)
                _ref = self.model._model(input_ids, use_cache=False).logits[0]
                _ref_resp = _ref[p_len - 1: p_len + len(resp_ids) - 1].float().contiguous()
                del _ref, input_ids
                ref_lps_cache[turn_id] = logprobs_from_logits(_ref_resp, resp_labels)
                del _ref_resp
        self.model._model.enable_adapter_layers()

        # --- Pass 2: new log-probs with gradients, per-turn backward ---
        total_loss = 0.0
        total_kl = 0.0
        n_valid  = 0
        _logged  = False

        for msg in messages:
            turn_id  = msg.get("turn_id", 0)
            role     = msg.get("role_name", "proposer")
            resp_ids = [t for t in msg.get("response_ids", []) if 0 <= t < vocab_size]
            if not resp_ids:
                if not _logged: print(f"  [diag] turn={turn_id} SKIP resp_ids empty", flush=True); _logged=True
                continue

            ref_lps = ref_lps_cache.get(turn_id)
            if ref_lps is None or not torch.isfinite(ref_lps).all():
                if not _logged: print(f"  [diag] turn={turn_id} SKIP ref_lps invalid", flush=True); _logged=True
                continue

            mask    = (all_turn_ids == turn_id)
            old_lps = all_old_lps[mask]
            n_resp  = len(resp_ids)
            n_align = min(n_resp, old_lps.shape[0])
            if n_align == 0:
                if not _logged: print(f"  [diag] turn={turn_id} SKIP n_align=0 old_lps={old_lps.shape[0]} resp_ids={n_resp}", flush=True); _logged=True
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
                if not _logged: print(f"  [diag] turn={turn_id} SKIP p_len=0", flush=True); _logged=True
                continue

            input_ids   = torch.tensor(prompt_ids + resp_ids, dtype=torch.long, device=device).unsqueeze(0)
            resp_labels = torch.tensor(resp_ids, dtype=torch.long, device=device)

            self.model._model.set_adapter(ROLE_ADAPTER.get(role, "proposer"))
            if not _logged:
                print(f"  [diag] p_len={p_len} n_resp={n_resp} prompt_max={max(prompt_ids)} resp_max={max(resp_ids)}", flush=True)
            logits_new = self.model._model(input_ids, use_cache=False).logits[0]
            logits_device = logits_new.device
            if not _logged:
                print(f"  [diag] turn={turn_id} role={role} p_len={p_len} n_resp={n_resp} "
                      f"logits.shape={logits_new.shape} "
                      f"logits_finite={torch.isfinite(logits_new).all().item()} "
                      f"input_ids_max={input_ids.max().item()} vocab={vocab_size}", flush=True)
            _resp_logits = logits_new[p_len - 1: p_len + n_resp - 1].float().contiguous()
            del logits_new
            new_lps = logprobs_from_logits(_resp_logits, resp_labels.to(logits_device))
            del _resp_logits

            if not torch.isfinite(new_lps).all():
                if not _logged: print(f"  [diag] turn={turn_id} SKIP new_lps non-finite", flush=True); _logged=True
                continue

            ref_lps = ref_lps.to(new_lps.device)
            old_lps_aligned = old_lps[:n_align].to(new_lps.device)
            log_ratio = (new_lps[:n_align] - old_lps_aligned).mean()
            if not _logged: print(f"  [diag] turn={turn_id} role={role} n_align={n_align} log_ratio={log_ratio:.3f} new_lps_mean={new_lps.mean():.3f} old_lps_mean={old_lps_aligned.mean():.3f} req_grad={new_lps.requires_grad}", flush=True); _logged=True
            if not torch.isfinite(log_ratio):
                continue
            if log_ratio.abs().item() > self._LOG_RATIO_THRESHOLD:
                print(f"  [diag] turn={turn_id} SKIP |log_ratio|={log_ratio.abs().item():.1f} > {self._LOG_RATIO_THRESHOLD}", flush=True)
                continue
            ratio = torch.exp(log_ratio)

            pg_loss = torch.maximum(
                -episode_advantage * ratio,
                -episode_advantage * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon),
            )

            kl = (new_lps[:n_align] - ref_lps[:n_align]).mean()
            if not torch.isfinite(kl):
                continue

            total_kl += abs(kl.item())
            turn_loss = pg_loss + self.kl_ctrl.value * kl
            (turn_loss / normalization).backward()  # immediate backward — only 1 turn in graph
            total_loss += turn_loss.item()
            n_valid += 1

        if n_valid == 0:
            print(f"  [loss] n_valid=0 n_messages={len(messages)} "
                  f"turn_ids_range=[{all_turn_ids.min().item() if len(all_turn_ids) else 'N/A'},"
                  f"{all_turn_ids.max().item() if len(all_turn_ids) else 'N/A'}]", flush=True)
            return 0.0, 0.0, 0
        return total_loss / n_valid, total_kl, n_valid
