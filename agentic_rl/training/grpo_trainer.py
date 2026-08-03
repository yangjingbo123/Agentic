import numpy as np
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from collections import defaultdict

from agents.agentic_executor import AgenticExecutor
from llm.trainable_llm import ROLE_ADAPTER


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: (seq, vocab), labels: (seq,) → log_probs: (seq,)"""
    return F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)


class GRPOAgenticTrainer:
    def __init__(self, model, tokenizer, config, vllm_engine=None):
        self.model         = model
        self.tokenizer     = tokenizer
        self.vllm_engine   = vllm_engine
        self.executor      = AgenticExecutor(model, tokenizer, config, vllm_engine=vllm_engine)
        self.optimizer     = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.get("lr", 1e-5),
        )
        self.clip_epsilon  = config.get("clip_epsilon", 0.2)
        self.n_samples     = config.get("n_samples", 8)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.ppo_epochs    = config.get("ppo_epochs", 1)
        self.raca_delta    = config.get("raca_delta", 1e-4)   # variance floor

    # ── RACA Phase 3: two-level advantage computation ────────────────────────

    def _compute_raca_advantages(self, episodes: list) -> list:
        """Compute per-turn RACA advantages for N episodes of the same question.

        Returns a list of N dicts: {turn_id → float advantage}.
        Controller turns use Layer 1 (episode-level).
        Proposer/Critic/Verifier turns use Layer 2 (step-level anchor).
        """
        N     = len(episodes)
        delta = self.raca_delta

        # ── Layer 1: controller episode-level advantage ──────────────────────
        # r_ctrl per episode = the single scalar from the last controller turn.
        ctrl_episode_rewards = []
        for ep in episodes:
            td = ep.get("raca_turn_data", {})
            r = sum(v["reward"] for v in td.values() if v["role"] == "controller")
            ctrl_episode_rewards.append(r)

        mu_ctrl  = float(np.mean(ctrl_episode_rewards))
        sig_ctrl = float(np.std(ctrl_episode_rewards))
        ctrl_adv = [
            (r - mu_ctrl) / max(sig_ctrl, delta)
            for r in ctrl_episode_rewards
        ]

        # ── Layer 2: step-level anchor groups for prop/crit/verif ───────────
        # Anchor: (role, controller_strategy) — groups turns by the cognitive
        # context the controller assigned for that round, rather than by round
        # number. Turns sharing the same (role, strategy) faced a comparable
        # task context (e.g., all proposers invoked under "refine" after errors
        # were flagged), enabling a cleaner controlled comparison.
        anchor_groups: dict = defaultdict(list)
        for ep_idx, ep in enumerate(episodes):
            for tid, v in ep.get("raca_turn_data", {}).items():
                if v["role"] == "controller":
                    continue
                key = (v["role"], v.get("strategy", "explore"))
                anchor_groups[key].append((ep_idx, tid, v["reward"]))

        # Normalise within each anchor group → step advantages
        step_adv: dict = {}   # (ep_idx, tid) → float
        for (role, rnd), group in anchor_groups.items():
            if len(group) < 2:
                continue
            rewards = [r for _, _, r in group]
            mu  = float(np.mean(rewards))
            sig = float(np.std(rewards))
            denom = max(sig, delta)
            for ep_idx, tid, r in group:
                step_adv[(ep_idx, tid)] = (r - mu) / denom

        # ── Build per-episode advantage dicts ────────────────────────────────
        per_ep_adv = [{} for _ in range(N)]

        for ep_idx, ep in enumerate(episodes):
            for tid, v in ep.get("raca_turn_data", {}).items():
                if v["role"] == "controller":
                    # All controller turns in this episode share the episode advantage.
                    per_ep_adv[ep_idx][tid] = ctrl_adv[ep_idx]
                else:
                    # Use step advantage if the anchor group had enough members.
                    sa = step_adv.get((ep_idx, tid))
                    if sa is not None:
                        per_ep_adv[ep_idx][tid] = sa

        return per_ep_adv

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

        for episode_group in batch_rollouts:
            if len(episode_group) < 2:
                continue
            per_ep_adv = self._compute_raca_advantages(episode_group)
            for ep, adv in zip(episode_group, per_ep_adv):
                if adv:   # skip episodes where no turn got an advantage
                    all_episodes.append(ep)
                    all_per_turn_adv.append(adv)

        if not all_episodes:
            if self.vllm_engine is not None:
                self.vllm_engine.sync_lora(self.model)
            return {"loss": 0.0, "mean_reward": 0.0, "accuracy": 0.0,
                    "kl": 0.0, "skipped": True}

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
              f"mean_ctrl_reward={mean_r:.3f}", flush=True)

        # Pre-count total valid turns for normalization
        total_valid = max(
            sum(self._count_valid_turns(ep, adv)
                for ep, adv in zip(all_episodes, all_per_turn_adv)), 1
        )

        total_loss    = 0.0
        total_n_valid = 0

        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            did_backward  = False
            epoch_loss    = 0.0
            epoch_n_valid = 0

            for ep, adv in zip(all_episodes, all_per_turn_adv):
                ep_loss, n_valid = self._compute_loss(ep, adv, total_valid)
                if n_valid > 0:
                    did_backward = True
                epoch_loss    += ep_loss
                epoch_n_valid += n_valid

            if did_backward:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                print(f"  [update] grad_norm={grad_norm:.4f} loss={epoch_loss:.4f}",
                      flush=True)
            else:
                print("  [update] SKIPPED: no valid gradients", flush=True)

            total_loss    += epoch_loss
            total_n_valid += epoch_n_valid

        if self.vllm_engine is not None:
            self.vllm_engine.sync_lora(self.model)

        return {
            "loss":        total_loss / max(total_n_valid, 1),
            "mean_reward": mean_r,
            "accuracy":    mean_acc,
            "kl":          0.0,
        }

    # ── Per-turn PPO loss ────────────────────────────────────────────────────

    _LOG_RATIO_THRESHOLD = 50.0

    def _compute_loss(
        self,
        episode: dict,
        per_turn_adv: dict,   # turn_id → float advantage
        normalization: int,
    ) -> tuple:
        """Per-turn GRPO clip loss (no KL penalty).
        Backward after each turn to keep peak memory minimal.
        per_turn_adv: maps turn_id → RACA advantage scalar.
        """
        device = next(p for p in self.model._model.parameters()).device

        # Guard: skip episode if LoRA weights have gone NaN/Inf.
        for p in self.model._model.parameters():
            if p.requires_grad and not torch.isfinite(p).all():
                print("  [loss] NaN/Inf in model weights, skipping episode", flush=True)
                return 0.0, 0

        vocab_size   = self.model._model.config.vocab_size
        messages     = episode["messages"]
        all_old_lps  = torch.tensor(episode["log_probs"], dtype=torch.float32, device=device)
        all_old_lps  = torch.nan_to_num(all_old_lps, nan=0.0, posinf=0.0, neginf=0.0)
        all_turn_ids = torch.tensor(episode["turn_ids"], device=device)

        total_loss = 0.0
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

            (pg_loss / normalization).backward()  # immediate backward — only 1 turn in graph
            total_loss += pg_loss.item()
            n_valid    += 1

        if n_valid == 0:
            return 0.0, 0
        return total_loss / n_valid, n_valid
