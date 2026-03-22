"""评估脚本 - 在测试集上评估模型"""
import yaml
import torch
from models.high_policy import HighLevelPolicy
from models.low_policy import LowLevelPolicy
from envs.reasoning_env import ReasoningEnv
from utils.data_loader import GSM8KLoader
from evaluation.metrics import compute_metrics


def evaluate(env, high_policy, low_policy, test_questions):
    """评估模型"""
    episodes = []

    print(f"开始评估: {len(test_questions)}个问题")

    for idx, question_data in enumerate(test_questions):
        if idx % 50 == 0:
            print(f"进度: {idx}/{len(test_questions)}")

        question = question_data["question"]
        ground_truth = question_data["answer"]

        # 运行episode
        obs = env.reset(question)
        done = False

        while not done:
            # 高层决策
            with torch.no_grad():
                obs_tensor = torch.tensor(obs["controller"]).unsqueeze(0)
                goal_logits, focus_logits, budget_logits = high_policy(obs_tensor)

                goal = torch.argmax(goal_logits, dim=-1).item()
                budget = torch.argmax(budget_logits, dim=-1).item()

                # 简化: 直接使用argmax
                goal_map = ["EXPLORE-open", "EXPLORE-minority", "CHALLENGE", "DIAGNOSE", "STOP"]
                budget_map = ["light", "standard", "heavy"]
                action = (goal_map[goal], "majority", budget_map[budget])

            obs, reward, done, info = env.step({"controller": action})

        # 记录episode信息
        episode_info = {
            "question": question,
            "final_answer": info.get("answer", ""),
            "correct": info.get("correct", False),
            "total_cost": env.blackboard.messages.__len__() * 400,  # 简化估算
        }
        episodes.append(episode_info)

    return episodes


def main():
    # 加载配置
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 加载模型
    high_policy = HighLevelPolicy(config["high_level"]["obs_dim"], config["high_level"]["hidden_dim"])
    low_policy = LowLevelPolicy(config["low_level"]["obs_dim"], config["low_level"]["hidden_dim"])

    checkpoint = torch.load("checkpoints/stage2_final.pt")
    high_policy.load_state_dict(checkpoint["high_policy"])
    low_policy.load_state_dict(checkpoint["low_policy"])
    high_policy.eval()
    low_policy.eval()

    # 初始化环境
    env = ReasoningEnv(config)

    # 加载测试数据
    loader = GSM8KLoader()
    test_questions = loader.load_split("test")

    # 评估
    episodes = evaluate(env, high_policy, low_policy, test_questions)

    # 计算指标
    ground_truth = [q["answer"] for q in test_questions]
    metrics = compute_metrics(episodes, ground_truth)

    print("\n评估结果:")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"平均成本: {metrics['avg_cost']:.2f}")
    print(f"等价k: {metrics['k_equiv']:.2f}")


if __name__ == "__main__":
    main()
