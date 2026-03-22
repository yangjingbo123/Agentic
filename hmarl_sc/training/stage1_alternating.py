"""Stage 1: 交替冻结训练"""
import torch
from training.ppo_trainer import PPOTrainer


def train_stage1(high_policy, high_value, low_policy, low_value, env, config):
    """Stage 1: 交替冻结训练
    Phase A: 冻结低层,训练高层
    Phase B: 冻结高层,训练低层
    """
    num_rounds = config.get("num_rounds", 3)
    phase_a_steps = config.get("phase_a_steps", 1000)
    phase_b_steps = config.get("phase_b_steps", 1000)
    decay_factor = config.get("decay_factor", 0.8)
    min_steps = config.get("min_steps", 300)

    high_trainer = PPOTrainer(high_policy, high_value, config.get("ppo", {}))
    low_trainer = PPOTrainer(low_policy, low_value, config.get("ppo", {}))

    print("Stage 1: 交替冻结训练")

    for round_idx in range(num_rounds):
        # 计算当前轮次的步数
        current_a_steps = max(int(phase_a_steps * (decay_factor ** round_idx)), min_steps)
        current_b_steps = max(int(phase_b_steps * (decay_factor ** round_idx)), min_steps)

        print(f"\n轮次 {round_idx+1}/{num_rounds}")
        print(f"Phase A步数: {current_a_steps}, Phase B步数: {current_b_steps}")

        # Phase A: 训练高层
        print("Phase A: 训练高层...")
        for param in low_policy.parameters():
            param.requires_grad = False

        # TODO: 收集高层轨迹并训练
        # high_trajectories = collect_trajectories(env, high_policy, low_policy, current_a_steps)
        # high_trainer.update(high_trajectories)

        # Phase B: 训练低层
        print("Phase B: 训练低层...")
        for param in low_policy.parameters():
            param.requires_grad = True
        for param in high_policy.parameters():
            param.requires_grad = False

        # TODO: 收集低层轨迹并训练
        # low_trajectories = collect_trajectories(env, high_policy, low_policy, current_b_steps)
        # low_trainer.update(low_trajectories)

        # 恢复高层梯度
        for param in high_policy.parameters():
            param.requires_grad = True

    print("\nStage 1完成")
    return high_policy, low_policy
