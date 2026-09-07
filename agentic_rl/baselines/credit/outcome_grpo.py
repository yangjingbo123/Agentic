"""Shared terminal-outcome GRPO for the four-role system."""

from baselines.common import broadcast_scalar, zscores
from baselines.credit.base import CreditAssigner


class OutcomeGRPOAssigner(CreditAssigner):
    name = "outcome_grpo"

    def compute(self, episode_group):
        rewards = [1.0 if ep.get("is_correct", False) else 0.0
                   for ep in episode_group]
        standardized = zscores(rewards, self.delta)
        diagnostics = {
            "credit/outcome/reward_mean": sum(rewards) / max(len(rewards), 1),
            "credit/outcome/coverage": 0.0,
        }
        if standardized is None:
            return [{} for _ in episode_group], diagnostics
        result = [broadcast_scalar(ep, adv)
                  for ep, adv in zip(episode_group, standardized)]
        assigned = sum(len(x) for x in result)
        possible = sum(len(ep.get("messages", [])) for ep in episode_group)
        diagnostics["credit/outcome/coverage"] = assigned / max(possible, 1)
        return result, diagnostics
