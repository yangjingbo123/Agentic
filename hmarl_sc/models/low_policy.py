"""低层Agent策略网络 - 共享主干"""
import torch
import torch.nn as nn


class RoleHead(nn.Module):
    """角色特定头"""

    def __init__(self, hidden_dim, work_actions, comm_actions):
        super().__init__()
        self.work_head = nn.Linear(hidden_dim, work_actions)
        self.comm_head = nn.Linear(hidden_dim, comm_actions)
        self.pointer_query = nn.Linear(hidden_dim, 64)

    def forward(self, features, targets=None):
        """
        Args:
            features: [batch, hidden_dim]
            targets: [batch, num_targets, 64] (可选,用于pointer)
        Returns:
            work_logits: [batch, work_actions]
            comm_logits: [batch, comm_actions]
            pointer_logits: [batch, num_targets] or None
        """
        work_logits = self.work_head(features)
        comm_logits = self.comm_head(features)

        pointer_logits = None
        if targets is not None:
            query = self.pointer_query(features).unsqueeze(1)
            pointer_logits = torch.bmm(query, targets.transpose(1, 2)).squeeze(1)

        return work_logits, comm_logits, pointer_logits


class LowLevelPolicy(nn.Module):
    """低层策略网络 - 三个agent共享主干"""

    def __init__(self, obs_dim=512, hidden_dim=128):
        super().__init__()
        # 共享主干
        self.shared_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 角色特定头
        self.role_heads = nn.ModuleDict({
            'proposer': RoleHead(hidden_dim, work_actions=5, comm_actions=5),
            'critic': RoleHead(hidden_dim, work_actions=4, comm_actions=5),
            'verifier': RoleHead(hidden_dim, work_actions=3, comm_actions=5),
        })

    def forward(self, obs, role, targets=None):
        """
        Args:
            obs: [batch, obs_dim]
            role: str, one of ['proposer', 'critic', 'verifier']
            targets: [batch, num_targets, 64] (可选)
        Returns:
            work_logits, comm_logits, pointer_logits
        """
        features = self.shared_trunk(obs)
        return self.role_heads[role](features, targets)
