import numpy as np
import torch
import torch.nn.functional as F

from agents.agentic_executor import AgenticExecutor


class AdaptiveKLController:
    """来自ReMA/InstructGPT：自动调整KL系数，防止训练collapse"""
    def __init__(self, init_kl_coef=0.1, target_kl=6.0, horizon=10000):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        self.value *= 1 + proportional_error * n_steps / self.horizon


class GRPOAgenticTrainer:
    def __init__(self, models: dict, ref_models: dict, tokenizer, config):
        self.models = models
        self.ref_models = ref_models   # 冻结的reference models，结构与models相同
        self.tokenizer = tokenizer
        self.executor = AgenticExecutor(models, tokenizer, config)
        self.optimizers = {
            role: torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.get("lr", 1e-5),
            )
            for role, model in models.items()
        }
        self.clip_epsilon = config.get("clip_epsilon", 0.02)
        self.n_samples = config.get("n_samples", 4)
        self.kl_ctrl = AdaptiveKLController(
            init_kl_coef=config.get("kl_coef", 0.1),
            target_kl=config.get("target_kl", 6.0),
        )
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self._step = 0

    def train_step(self, question: str, correct_answer: str) -> dict:
        episodes = [self.executor.run_episode(question, correct_answer)
                    for _ in range(self.n_samples)]

        rewards = [sum(ep["rewards"]) for ep in episodes]
        mean_r = np.mean(rewards)
        advantages = [(r - mean_r) / (np.std(rewards) + 1e-8) for r in rewards]

        role_losses = defaultdict(list)
        total_kl = 0.0

        for ep, adv in zip(episodes, advantages):
            per_role, kl = self._compute_per_role_loss(ep, adv)
            total_kl += kl
            for role, loss in per_role.items():
                role_losses[role].append(loss)

        mean_kl = total_kl / self.n_samples
        self.kl_ctrl.update(mean_kl, n_steps=1)
        self._step += 1

        for role, losses in role_losses.items():
            if not losses:
                continue
            total = sum(losses) / len(losses)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.models[role].parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self.optimizers[role].step()
            self.optimizers[role].zero_grad()

        return {
            "loss": np.mean([sum(v).item() / len(v) for v in role_losses.values() if v]),
            "mean_reward": mean_r,
            "accuracy": sum(ep["is_correct"] for ep in episodes) / self.n_samples,
            "kl": mean_kl,
            "kl_coef": self.kl_ctrl.value,
        }

    def _compute_per_role_loss(self, episode: dict, advantage: float) -> tuple[dict, float]:
        turn_ids   = torch.tensor(episode["turn_ids"])
        old_lps    = torch.tensor(episode["log_probs"])
        rewards    = torch.tensor(episode["rewards"])
        messages   = episode["messages"]

        role_new_lps = self._recompute_log_probs_by_role(messages, use_ref=False)
        role_ref_lps = self._recompute_log_probs_by_role(messages, use_ref=True)

        role_losses = {}
        total_kl = 0.0

        for role, (turn_indices, new_lps_list) in role_new_lps.items():
            ref_lps_list = role_ref_lps[role][1]
            loss = torch.zeros(1).squeeze()
            n_valid = 0

            for t_idx, new_lps, ref_lps in zip(turn_indices, new_lps_list, ref_lps_list):
                mask = turn_ids == t_idx
                if not mask.any():
                    continue

                old = old_lps[mask]
                # KL penalty加入reward：r_kl = r - kl_coef * KL
                kl = (new_lps - ref_lps).mean().item()
                total_kl += abs(kl)
                turn_reward = rewards[t_idx].item() - self.kl_ctrl.value * kl

                # Turn-level ratio（参考ReMA core_algos.py的turn clip模式）
                avg_log_ratio = (new_lps - old).mean()
                ratio = torch.exp(avg_log_ratio)
                turn_adv = advantage * turn_reward

                # Dual-clip PPO（ReMA做法：对负advantage额外clip，防止过度惩罚）
                pg_loss1 = -turn_adv * ratio
                pg_loss2 = -turn_adv * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                pg_loss = torch.maximum(pg_loss1, pg_loss2)
                if turn_adv < 0:
                    pg_loss = torch.min(pg_loss, torch.tensor(-turn_adv * 3.0))  # dual-clip c=3

                loss = loss + pg_loss
                n_valid += 1

            if n_valid > 0:
                role_losses[role] = loss / n_valid

        return role_losses, total_kl

    def _recompute_log_probs_by_role(self, messages: list, use_ref: bool) -> dict:
        model_dict = self.ref_models if use_ref else self.models
        role_data: dict[str, tuple[list, list]] = {}

        for turn_id, msg in enumerate(messages):
            role = msg["role_name"]
            model = model_dict[role]

            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": msg["system"]},
                 {"role": "user",   "content": msg["user"]}],
                return_tensors="pt", add_generation_prompt=True
            ).to(model.device)
            response_ids = self.tokenizer(
                msg["response"], return_tensors="pt", add_special_tokens=False
            ).input_ids.to(model.device)

            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            ctx = torch.no_grad() if use_ref else torch.enable_grad()
            with ctx:
                logits = model(full_ids).logits

            resp_start = prompt_ids.shape[1]
            lps = torch.stack([
                F.log_softmax(logits[0, resp_start + j - 1], dim=-1)[tok]
                for j, tok in enumerate(response_ids[0])
            ])
            if role not in role_data:
                role_data[role] = ([], [])
            role_data[role][0].append(turn_id)
            role_data[role][1].append(lps)

        return role_data
