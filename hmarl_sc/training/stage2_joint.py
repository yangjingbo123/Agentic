"""Stage 2: 联合微调"""
import torch
from training.ppo_trainer import PPOTrainer


def train_stage2(high_policy, high_value, low_policy, low_value, env, config):
    """Stage 2: 联合微调高层和低层"""
    num_steps = config.get("num_steps", 5000)
    lr_decay = config.get("lr_decay", 0.33)
    check_interval = config.get("check_interval", 100)

    # 降低学习率
    for param_group in high_policy.optimizer.param_groups:
        param_group['lr'] *= lr_decay
    for param_group in low_policy.optimizer.param_groups:
        param_group['lr'] *= lr_decay

    print(f"Stage 2: 联合微调 ({num_steps}步)")

    high_trainer = PPOTrainer(high_policy, high_value, config.get("ppo", {}))
    low_trainer = PPOTrainer(low_policy, low_value, config.get("ppo", {}))

    for step in range(num_steps):
        # TODO: 收集联合轨迹并训练
        # trajectories = collect_trajectories(env, high_policy, low_policy, 1)
        # high_trainer.update(high_trajectories)
        # low_trainer.update(low_trajectories)

        if (step + 1) % check_interval == 0:
            print(f"步数 {step+1}/{num_steps}")

    print("Stage 2完成")
    return high_policy, low_policy
