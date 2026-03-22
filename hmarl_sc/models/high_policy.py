"""高层Controller策略网络"""
import torch
import torch.nn as nn


class HighLevelPolicy(nn.Module):
    """高层策略网络 - Controller"""

    def __init__(self, obs_dim=271, hidden_dim=128):
        super().__init__()
        # obs_dim = 15 (统计特征) + 256 (问题embedding)
        self.shared_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Goal head: 5-way (EXPLORE-open, EXPLORE-minority, CHALLENGE, DIAGNOSE, STOP)
        self.goal_head = nn.Linear(hidden_dim, 5)

        # Focus head: Pointer attention over candidate answers
        self.focus_query = nn.Linear(hidden_dim, 64)
        self.focus_key = nn.Linear(256, 64)  # answer embedding

        # Budget head: 3-way (light, standard, heavy)
        self.budget_head = nn.Linear(hidden_dim, 3)

    def forward(self, obs, answer_embeddings=None):
        """
        Args:
            obs: [batch, obs_dim]
            answer_embeddings: [batch, num_answers, 256] (可选)
        Returns:
            goal_logits: [batch, 5]
            focus_logits: [batch, num_answers] or None
            budget_logits: [batch, 3]
        """
        features = self.shared_trunk(obs)

        goal_logits = self.goal_head(features)
        budget_logits = self.budget_head(features)

        # Pointer attention for focus
        focus_logits = None
        if answer_embeddings is not None:
            query = self.focus_query(features).unsqueeze(1)  # [batch, 1, 64]
            keys = self.focus_key(answer_embeddings)  # [batch, num_answers, 64]
            focus_logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # [batch, num_answers]

        return goal_logits, focus_logits, budget_logits
