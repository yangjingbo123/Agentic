"""Shared PPO trainer for credit-assignment baselines.

The production ``training/grpo_trainer.py`` is imported but never edited.  This
subclass reuses its optimizer and token-level PPO/KL implementation while
replacing only the advantage construction step.
"""

import json
import os

import numpy as np
import torch

from baselines.registry import build_credit_assigner
from training.grpo_trainer import GRPOAgenticTrainer


class BaselineGRPOTrainer(GRPOAgenticTrainer):
    """Four-role trainer with a pluggable non-RACA credit assigner."""

    def __init__(self, model, tokenizer, config, vllm_engine=None):
        super().__init__(model, tokenizer, config, vllm_engine=vllm_engine)
        name = os.environ.get("BASELINE_NAME", "outcome_grpo")
        raw = os.environ.get("BASELINE_CONFIG_JSON", "{}")
        try:
            extra = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("BASELINE_CONFIG_JSON must be valid JSON") from exc
        extra.setdefault("delta", config.get("raca_delta", 1e-4))
        self.baseline_name = name
        self.credit_assigner = build_credit_assigner(name, extra)
        # Keep baseline PPO regularization standard even when the RACA v34
        # config enables interaction-specific KL/entropy and role balancing.
        self.interaction_kl_coef = self.kl_coef
        self.interaction_entropy_coef = 0.0
        self.interaction_entropy_correct_only = True
        self.role_loss_normalization = False
        self.metrics_path = os.environ.get("BASELINE_METRICS_JSONL")
        self.update_index = 0
        print("[baseline] credit assigner = %s config=%s" %
              (self.baseline_name, extra), flush=True)


    def _record_stats(self, stats, batch_rollouts):
        self.update_index += 1
        episodes = [ep for group in batch_rollouts for ep in group]
        messages = [msg for ep in episodes for msg in ep.get("messages", [])]
        stats["baseline_update"] = self.update_index
        stats["training_llm_calls"] = len(messages)
        stats["training_generated_tokens"] = sum(
            len(msg.get("response_ids", [])) for msg in messages)
        if self.metrics_path:
            os.makedirs(os.path.dirname(self.metrics_path) or ".", exist_ok=True)
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats, ensure_ascii=False, default=str) + "\n")
        return stats

    def update(self, batch_rollouts):
        all_episodes = []
        all_per_turn_adv = []
        n_groups = 0
        n_groups_kept = 0
        diagnostics = self._interaction_metrics(batch_rollouts)
        credit_sums = {}
        credit_counts = {}

        for episode_group in batch_rollouts:
            if len(episode_group) < 2:
                continue
            n_groups += 1
            per_ep_adv, group_diag = self.credit_assigner.compute(episode_group)
            for key, value in group_diag.items():
                if isinstance(value, (int, float)):
                    credit_sums[key] = credit_sums.get(key, 0.0) + float(value)
                    credit_counts[key] = credit_counts.get(key, 0) + 1
            kept = 0
            for ep, adv in zip(episode_group, per_ep_adv):
                if adv:
                    all_episodes.append(ep)
                    all_per_turn_adv.append(adv)
                    kept += 1
            if kept:
                n_groups_kept += 1

        for key, total in credit_sums.items():
            diagnostics[key] = total / max(credit_counts.get(key, 1), 1)
        diagnostics["baseline/name"] = self.baseline_name
        numeric_diag = [
            "%s=%.3f" % (key, value)
            for key, value in sorted(diagnostics.items())
            if key.startswith("credit/") and isinstance(value, (int, float))
        ]
        if numeric_diag:
            print("  [credit:%s] %s" %
                  (self.baseline_name, " ".join(numeric_diag)), flush=True)

        if not all_episodes:
            if self.vllm_engine is not None:
                self.vllm_engine.sync_lora(self.model)
            return self._record_stats(
                {"loss": 0.0, "mean_reward": 0.0, "accuracy": 0.0,
                 "kl": 0.0, "skipped": True, "groups_total": n_groups,
                 "groups_kept": 0, **diagnostics}, batch_rollouts)

        n_correct = sum(bool(ep.get("is_correct")) for ep in all_episodes)
        mean_acc = n_correct / len(all_episodes)
        mean_r = mean_acc
        print("  [baseline-rollout:%s] correct=%d/%d groups=%d/%d" %
              (self.baseline_name, n_correct, len(all_episodes),
               n_groups_kept, n_groups), flush=True)

        total_valid = sum(self._count_valid_turns(ep, adv)
                          for ep, adv in zip(all_episodes, all_per_turn_adv))
        if total_valid == 0:
            if self.vllm_engine is not None:
                self.vllm_engine.sync_lora(self.model)
            return self._record_stats(
                {"loss": 0.0, "mean_reward": mean_r, "accuracy": mean_acc,
                 "kl": 0.0, "skipped": True, "groups_total": n_groups,
                 "groups_kept": n_groups_kept, **diagnostics}, batch_rollouts)

        total_loss = 0.0
        total_n_valid = 0
        agg = {"kl_sum": 0.0, "ent_sum": 0.0, "clip_sum": 0.0,
               "ratio_sum": 0.0, "ratio_max": 0.0, "n_tok": 0,
               "n_pg_tok": 0, "resp_len_sum": 0, "n_turn": 0,
               "solution_pg_sum": 0.0, "interaction_pg_sum": 0.0,
               "solution_adv_abs_sum": 0.0,
               "interaction_adv_abs_sum": 0.0,
               "n_solution_channel": 0, "n_interaction_channel": 0,
               "n_solution_tok": 0, "n_interaction_tok": 0}
        summed_keys = tuple(agg.keys())

        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            did_backward = False
            epoch_loss = 0.0
            epoch_n_valid = 0
            for ep, adv in zip(all_episodes, all_per_turn_adv):
                ep_loss, n_valid, diag = self._compute_loss(ep, adv, total_valid)
                did_backward = did_backward or n_valid > 0
                epoch_loss += ep_loss
                epoch_n_valid += n_valid
                for key in summed_keys:
                    if key == "ratio_max":
                        agg[key] = max(agg[key], diag.get(key, 0.0))
                    else:
                        agg[key] += diag.get(key, 0.0)
            if did_backward:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.lora_parameters(), self.max_grad_norm)
                self.optimizer.step()
                print("  [baseline-update:%s] grad_norm=%.4f loss=%.4f" %
                      (self.baseline_name, grad_norm, epoch_loss), flush=True)
            total_loss += epoch_loss
            total_n_valid += epoch_n_valid

        if self.vllm_engine is not None:
            self.vllm_engine.sync_lora(self.model)
        if total_n_valid == 0:
            return self._record_stats(
                {"loss": 0.0, "mean_reward": mean_r, "accuracy": mean_acc,
                 "kl": 0.0, "skipped": True, "groups_total": n_groups,
                 "groups_kept": n_groups_kept, **diagnostics}, batch_rollouts)

        nt = max(agg["n_tok"], 1)
        npg = max(agg["n_pg_tok"], 1)
        return self._record_stats({
            "loss": total_loss / max(total_n_valid, 1),
            "mean_reward": mean_r,
            "accuracy": mean_acc,
            "kl": agg["kl_sum"] / nt,
            "groups_total": n_groups,
            "groups_kept": n_groups_kept,
            "entropy": agg["ent_sum"] / nt,
            "clip_frac": agg["clip_sum"] / npg,
            "ratio_mean": agg["ratio_sum"] / npg,
            "ratio_max": agg["ratio_max"],
            "resp_len": agg["resp_len_sum"] / max(agg["n_turn"], 1),
            **diagnostics,
        }, batch_rollouts)
