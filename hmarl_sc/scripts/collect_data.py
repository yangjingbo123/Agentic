"""数据收集脚本 - 使用规则策略收集BC训练数据"""
import yaml
from envs.reasoning_env import ReasoningEnv
from training.rule_policies import SimpleRule
from utils.data_loader import GSM8KLoader
import pickle


def collect_bc_data(env, rule_policy, questions, num_episodes):
    """使用规则策略收集数据"""
    trajectories = []

    print(f"开始收集BC数据: {num_episodes}个episode")

    for ep_idx, question_data in enumerate(questions[:num_episodes]):
        if ep_idx % 100 == 0:
            print(f"进度: {ep_idx}/{num_episodes}")

        question = question_data["question"]
        episode_traj = []

        # 重置环境
        obs = env.reset(question)
        done = False
        round_num = 0

        while not done:
            # 高层决策
            high_action = rule_policy.get_high_level_action(obs, round_num)
            episode_traj.append(("high", obs["controller"], high_action))

            # 执行高层动作
            obs, reward, done, info = env.step({"controller": high_action})

            if done:
                break

            # 低层执行
            step = 0
            low_done = False
            while not low_done:
                low_actions = rule_policy.get_low_level_actions(
                    high_action[0], high_action[1], env.blackboard, step
                )

                for agent_id, action in low_actions.items():
                    episode_traj.append(("low", obs[agent_id], action))

                obs, reward, low_done, info = env.step(low_actions)
                step += 1

            round_num += 1

        trajectories.append(episode_traj)

    return trajectories


def main():
    # 加载配置
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化环境
    env = ReasoningEnv(config)

    # 加载数据
    loader = GSM8KLoader()
    train_questions = loader.load_split("train")

    # 使用简单规则策略
    rule_policy = SimpleRule()

    # 收集数据
    num_episodes = config["stage0"].get("num_episodes", 3000)
    trajectories = collect_bc_data(env, rule_policy, train_questions, num_episodes)

    # 保存数据
    with open("data/bc_trajectories.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    print(f"数据收集完成! 共{len(trajectories)}个episode")
    print(f"已保存到: data/bc_trajectories.pkl")


if __name__ == "__main__":
    main()
