"""价值网络"""
import torch
import torch.nn as nn


class HighLevelValue(nn.Module):
    """高层价值网络"""

    def __init__(self, obs_dim=271, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs)


class LowLevelValue(nn.Module):
    """低层价值网络 - 集中式Critic"""

    def __init__(self, joint_obs_dim=1536, hidden_dim=128):
        super().__init__()
        # 观测所有3个agent的状态
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs)
