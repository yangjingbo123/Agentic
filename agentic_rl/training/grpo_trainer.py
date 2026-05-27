from collections import defaultdict
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
    def __init__(self, models: dict, ref_models: dict, tokenizer, config, vllm_engine=None):
        self.models = models
        self.ref_models = ref_models
        self.tokenizer = tokenizer
        self.vllm_engine = vllm_engine
        self.executor = AgenticExecutor(models, tokenizer, config, vllm_engine=vllm_engine)
        self.optimizers = {
            role: torch.optim.AdamW(
                list(model.parameters(role)) if hasattr(model, 'set_role')
                else [p for p in model.parameters() if p.requires_grad],
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
        episodes = []
        for i in range(self.n_samples):
            print(f"  Running episode {i+1}/{self.n_samples}...", flush=True)
            episodes.append(self.executor.run_episode(question, correct_answer))

        rewards = [sum(ep["rewards"]) for ep in episodes]
        mean_r = np.mean(rewards)
        advantages = [(r - mean_r) / (np.std(rewards) + 1e-8) for r in rewards]

        total_kl = 0.0
        role_loss_vals = defaultdict(list)

        for opt in self.optimizers.values():
            opt.zero_grad()

        for ep, adv in zip(episodes, advantages):
            per_role, kl = self._compute_per_role_loss(ep, adv)
            total_kl += kl
            for role, loss in per_role.items():
                scaled = loss / self.n_samples
                scaled.backward()
                role_loss_vals[role].append(loss.item())

        for role, opt in self.optimizers.items():
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.models[role].parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            opt.step()

        # 同步 LoRA 权重到 vLLM（每步更新后）
        if self.vllm_engine is not None:
            shared = next(iter(self.models.values()))
            self.vllm_engine.sync_all_loras(shared)

        mean_kl = total_kl / self.n_samples
        self.kl_ctrl.update(mean_kl, n_steps=1)
        self._step += 1

        return {
            "loss": np.mean([np.mean(v) for v in role_loss_vals.values() if v]),
            "mean_reward": mean_r,
            "accuracy": sum(ep["is_correct"] for ep in episodes) / self.n_samples,
            "kl": mean_kl,
            "kl_coef": self.kl_ctrl.value,
        }

    def _compute_per_role_loss(self, episode: dict, advantage: float) -> tuple[dict, float]:
        device     = next(iter(self.models.values())).device
        turn_ids   = torch.tensor(episode["turn_ids"]).to(device)
        old_lps    = torch.tensor(episode["log_probs"]).to(device)
        rewards    = torch.tensor(episode["rewards"]).to(device)
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
                n = min(len(old), len(new_lps), len(ref_lps))
                old, new_lps, ref_lps = old[:n], new_lps[:n], ref_lps[:n]
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

        # 按 role 分组，确保同一 role 的所有 turn 连续处理，不中途切换 adapter
        from collections import defaultdict
        role_msgs = defaultdict(list)  # role -> [(turn_id, msg)]
        for turn_id, msg in enumerate(messages):
            role_msgs[msg["role_name"]].append((turn_id, msg))

        for role, turns in role_msgs.items():
            model = model_dict[role]
            if not use_ref and hasattr(model, 'set_role'):
                model.set_role(role)

            turn_ids_list, lps_list = [], []
            for turn_id, msg in turns:
                prompt_ids = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": msg["system"]},
                     {"role": "user",   "content": msg["user"]}],
                    return_tensors="pt", add_generation_prompt=True,
                    return_dict=True,
                )["input_ids"].to(model.device)
                response_ids = self.tokenizer(
                    msg["response"], return_tensors="pt", add_special_tokens=False
                ).input_ids.to(model.device)

                full_ids = torch.cat([prompt_ids, response_ids], dim=1)
                resp_start = prompt_ids.shape[1]
                ctx = torch.no_grad() if use_ref else torch.enable_grad()
                with ctx:
                    logits = model(full_ids).logits[0, resp_start - 1: resp_start + len(response_ids[0]) - 1]
                lps = torch.stack([
                    F.log_softmax(logits[j], dim=-1)[tok]
                    for j, tok in enumerate(response_ids[0])
                ])
                del logits
                turn_ids_list.append(turn_id)
                lps_list.append(lps)

            role_data[role] = (turn_ids_list, lps_list)

        return role_data
