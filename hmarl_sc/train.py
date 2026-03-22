"""主训练脚本"""
import yaml
import torch
from models.high_policy import HighLevelPolicy
from models.low_policy import LowLevelPolicy
from models.value_nets import HighLevelValue, LowLevelValue
from training.stage0_bc import train_bc
from training.stage1_alternating import train_stage1
from training.stage2_joint import train_stage2
from envs.reasoning_env import ReasoningEnv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", default="0", choices=["0", "1", "2"])
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"开始训练 Stage {args.stage}")

    # 初始化模型
    high_policy = HighLevelPolicy(
        obs_dim=config["high_level"]["obs_dim"],
        hidden_dim=config["high_level"]["hidden_dim"]
    )
    low_policy = LowLevelPolicy(
        obs_dim=config["low_level"]["obs_dim"],
        hidden_dim=config["low_level"]["hidden_dim"]
    )
    high_value = HighLevelValue(
        obs_dim=config["high_level"]["obs_dim"],
        hidden_dim=config["high_level"]["hidden_dim"]
    )
    low_value = LowLevelValue(
        joint_obs_dim=config["low_level"]["obs_dim"] * 3,
        hidden_dim=config["low_level"]["hidden_dim"]
    )

    # 初始化环境
    env = ReasoningEnv(config)

    if args.stage == "0":
        print("Stage 0: BC初始化")
        # TODO: 收集BC数据
        trajectories = []
        high_policy, low_policy = train_bc(high_policy, low_policy, trajectories, config["stage0"])

        # 保存模型
        torch.save({
            "high_policy": high_policy.state_dict(),
            "low_policy": low_policy.state_dict(),
        }, "checkpoints/stage0_final.pt")
        print("Stage 0完成,模型已保存")

    elif args.stage == "1":
        print("Stage 1: 交替冻结训练")
        # 加载Stage 0模型
        checkpoint = torch.load("checkpoints/stage0_final.pt")
        high_policy.load_state_dict(checkpoint["high_policy"])
        low_policy.load_state_dict(checkpoint["low_policy"])

        high_policy, low_policy = train_stage1(high_policy, high_value, low_policy, low_value, env, config["stage1"])

        torch.save({
            "high_policy": high_policy.state_dict(),
            "low_policy": low_policy.state_dict(),
        }, "checkpoints/stage1_final.pt")
        print("Stage 1完成,模型已保存")

    elif args.stage == "2":
        print("Stage 2: 联合微调")
        # 加载Stage 1模型
        checkpoint = torch.load("checkpoints/stage1_final.pt")
        high_policy.load_state_dict(checkpoint["high_policy"])
        low_policy.load_state_dict(checkpoint["low_policy"])

        high_policy, low_policy = train_stage2(high_policy, high_value, low_policy, low_value, env, config["stage2"])

        torch.save({
            "high_policy": high_policy.state_dict(),
            "low_policy": low_policy.state_dict(),
        }, "checkpoints/stage2_final.pt")
        print("Stage 2完成,模型已保存")


if __name__ == "__main__":
    main()
