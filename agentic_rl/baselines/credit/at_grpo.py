"""AT-GRPO adaptation for the project's fixed four-role workflow.

The original method uses tree sampling and groups K candidate actions by
(environment, agent, turn), with reward ``alpha * team + local``.  This project
retains its existing complete-trajectory sampler, so the paper-facing name must
remain ``AT-GRPO-Adapted``.  Agent/turn grouping and Eq. (3) reward mixing are
implemented directly; the approved default follows the paper's alpha=1.
"""

from collections import defaultdict

from baselines.common import role_occurrences, trainable_turn_ids, zscores
from baselines.credit.base import CreditAssigner
from baselines.credit.role_reward import role_rewards


class ATGRPOAssigner(CreditAssigner):
    name = "at_grpo"

    def __init__(self, config=None):
        super().__init__(config)
        self.team_weight = float(self.config.get("team_weight", 1.0))
        if self.team_weight < 0.0:
            raise ValueError("at_grpo.team_weight must be non-negative")

    def compute(self, episode_group):
        groups = defaultdict(list)
        for ep_idx, ep in enumerate(episode_group):
            team_reward = 1.0 if ep.get("is_correct", False) else 0.0
            local_rewards = role_rewards(ep)
            occurrences = role_occurrences(ep)
            for tid in trainable_turn_ids(ep):
                # Project adaptation of the original role-specific local score.
                # A missing label receives no local shaping, not a duplicate team
                # reward; team credit is already present in the first term.
                local_reward = float(local_rewards.get(tid, 0.0))
                reward = self.team_weight * team_reward + local_reward
                msg_role = next((m.get("role_name", "proposer")
                                 for m in ep.get("messages", [])
                                 if m.get("turn_id") == tid), "proposer")
                key = (msg_role, occurrences.get(tid, 0))
                groups[key].append((ep_idx, tid, reward))

        result = [{} for _ in episode_group]
        live = 0
        for _key, entries in groups.items():
            zs = zscores([x[2] for x in entries], self.delta)
            if zs is None:
                continue
            live += 1
            for (ep_idx, tid, _reward), adv in zip(entries, zs):
                result[ep_idx][tid] = adv

        assigned = sum(len(x) for x in result)
        possible = sum(len(trainable_turn_ids(ep)) for ep in episode_group)
        return result, {
            "credit/at_grpo/team_weight": self.team_weight,
            "credit/at_grpo/groups": len(groups),
            "credit/at_grpo/live_groups": live,
            "credit/at_grpo/coverage": assigned / max(possible, 1),
        }
