"""GiGPO adaptation for open-text blackboard trajectories.

The implementation follows the paper's two-level structure:

- episode advantage from terminal return;
- step advantage from discounted return-to-go among matching anchor states;
- ``A = A_episode + omega * A_step``.

Exact mode hashes the full generation prompt as textual state. The optional
structured mode is diagnostic only. We use sparse terminal correctness, matching
the paper's sparse-reward setting; with gamma=1 it propagates to all preceding
steps through return-to-go.
"""

from collections import defaultdict

from baselines.common import (message_by_turn, prompt_digest, trainable_turn_ids,
                              turn_metadata, zscores)
from baselines.credit.base import CreditAssigner


class GiGPOAssigner(CreditAssigner):
    name = "gigpo"

    def __init__(self, config=None):
        super().__init__(config)
        self.macro_weight = float(self.config.get("macro_weight", 1.0))
        self.micro_weight = float(self.config.get("micro_weight", 1.0))
        self.gamma = float(self.config.get("gamma", 1.0))
        self.anchor_mode = str(self.config.get("anchor_mode", "exact"))
        if self.anchor_mode not in ("exact", "structured"):
            raise ValueError("gigpo.anchor_mode must be exact or structured")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gigpo.gamma must be in (0, 1]")

    def _anchor(self, episode, tid):
        msg = message_by_turn(episode).get(tid, {})
        if self.anchor_mode == "exact":
            return (msg.get("role_name", "proposer"), prompt_digest(msg))
        meta = turn_metadata(episode, tid)
        return (meta["role"], meta["round"], meta["is_response"])

    def _returns(self, episode, terminal_reward):
        tids = trainable_turn_ids(episode)
        returns = {}
        running = 0.0
        # Sparse environment reward: terminal correctness is emitted only after
        # the last generated action, then propagated backward by Eq. (5).
        for rev_idx, tid in enumerate(reversed(tids)):
            step_reward = float(terminal_reward) if rev_idx == 0 else 0.0
            running = step_reward + self.gamma * running
            returns[tid] = running
        return returns

    def compute(self, episode_group):
        terminal = [1.0 if ep.get("is_correct", False) else 0.0
                    for ep in episode_group]
        macro = zscores(terminal, self.delta)
        macro_live = macro is not None
        if macro is None:
            macro = [0.0] * len(episode_group)

        anchor_groups = defaultdict(list)
        exact_groups = defaultdict(list)
        for ep_idx, ep in enumerate(episode_group):
            messages = message_by_turn(ep)
            returns = self._returns(ep, terminal[ep_idx])
            for tid in trainable_turn_ids(ep):
                anchor_groups[self._anchor(ep, tid)].append(
                    (ep_idx, tid, returns[tid]))
                msg = messages.get(tid, {})
                exact_groups[(msg.get("role_name", "proposer"),
                              prompt_digest(msg))].append((ep_idx, tid))

        micro = {}
        live_anchors = 0
        for _key, entries in anchor_groups.items():
            # Same state revisited inside one trajectory counts as a distinct
            # occurrence in the GiGPO paper; keep all occurrences here.
            zs = zscores([x[2] for x in entries], self.delta)
            if zs is None:
                continue
            live_anchors += 1
            for (ep_idx, tid, _return), adv in zip(entries, zs):
                micro[(ep_idx, tid)] = adv

        result = [{} for _ in episode_group]
        for ep_idx, ep in enumerate(episode_group):
            for tid in trainable_turn_ids(ep):
                has_micro = (ep_idx, tid) in micro
                if not macro_live and not has_micro:
                    continue
                value = self.macro_weight * float(macro[ep_idx])
                if has_micro:
                    value += self.micro_weight * float(micro[(ep_idx, tid)])
                result[ep_idx][tid] = value

        possible = sum(len(trainable_turn_ids(ep)) for ep in episode_group)
        exact_usable = sum(len(v) for v in exact_groups.values() if len(v) >= 2)
        return result, {
            "credit/gigpo/anchor_mode_exact": int(self.anchor_mode == "exact"),
            "credit/gigpo/gamma": self.gamma,
            "credit/gigpo/anchors": len(anchor_groups),
            "credit/gigpo/live_anchors": live_anchors,
            "credit/gigpo/micro_coverage": len(micro) / max(possible, 1),
            "credit/gigpo/exact_anchor_coverage": exact_usable / max(possible, 1),
        }
