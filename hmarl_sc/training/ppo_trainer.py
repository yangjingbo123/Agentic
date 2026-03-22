"""PPO训练器"""
import torch
import torch.nn.functional as F
from typing import Dict, List


class PPOTrainer:
    """PPO训练器"""

    def __init__(self, policy, value_net, config):
        self.policy = policy
        self.value_net = value_net
        self.lr = config.get("lr_policy", 3e-4)
        self.clip_epsilon = config.get("clip_epsilon", 0.1)
        self.entropy_coeff = config.get("entropy_coeff", 0.01)

        self.optimizer_policy = torch.optim.Adam(policy.parameters(), lr=self.lr)
        self.optimizer_value = torch.optim.Adam(value_net.parameters(), lr=config.get("lr_value", 1e-3))

    def update(self, trajectories: List[Dict]):
        """PPO更新"""
        # 提取数据
        obs = torch.stack([t["obs"] for t in trajectories])
        actions = torch.stack([t["action"] for t in trajectories])
        old_log_probs = torch.stack([t["log_prob"] for t in trajectories])
        advantages = torch.stack([t["advantage"] for t in trajectories])
        returns = torch.stack([t["return"] for t in trajectories])

        # 策略更新
        self.optimizer_policy.zero_grad()
        logits = self.policy(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        # PPO clip
        ratio = torch.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus
        entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
        loss = policy_loss - self.entropy_coeff * entropy

        loss.backward()
        self.optimizer_policy.step()

        # 价值网络更新
        self.optimizer_value.zero_grad()
        values = self.value_net(obs).squeeze()
        value_loss = F.mse_loss(values, returns)
        value_loss.backward()
        self.optimizer_value.step()

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}
