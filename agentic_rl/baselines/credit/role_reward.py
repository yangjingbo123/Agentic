"""Simple role-wise reward shaping baseline.

The rewards are intentionally simpler than RACA: local task correctness for
proposers, binary review classification for critics, calibration for verifiers,
and final outcome for controllers.  There is no interaction reward, causal
window, primary-correctness layer, or token-channel routing.
"""

from collections import defaultdict

from agents.agentic_executor import math_equal
from agents.parsing import critic_found_errors, parse_reasoning, parse_score
from baselines.common import message_by_turn, trainable_turn_ids, turn_metadata, zscores
from baselines.credit.base import CreditAssigner


def role_rewards(episode):
    gold = episode.get("baseline_gold")
    messages = episode.get("messages", [])
    rewards = {}
    last_answer = ""
    last_correct = False
    for msg in messages:
        tid = msg.get("turn_id")
        if tid is None:
            continue
        role = msg.get("role_name", "proposer")
        text = msg.get("response", "")
        if role == "proposer":
            _reasoning, answer = parse_reasoning(text)
            last_answer = answer
            if gold is not None:
                last_correct = math_equal(answer, gold)
                rewards[tid] = 1.0 if last_correct else 0.0
            else:
                # Synthetic/offline compatibility: consume only the proposer's
                # correctness field, never the composite RACA reward.
                td = episode.get("raca_turn_data", {}).get(tid, {})
                value = td.get("r_prop", td.get("reward", 0.0))
                rewards[tid] = 1.0 if float(value) > 0.5 else 0.0
                last_correct = bool(rewards[tid])
        elif role == "critic":
            flagged = critic_found_errors(text)
            rewards[tid] = 1.0 if flagged == (not last_correct) else 0.0
        elif role == "verifier":
            score = parse_score(text)
            rewards[tid] = (0.0 if score is None else
                            1.0 - abs(float(score) - float(last_correct)))
        elif role == "controller":
            rewards[tid] = 1.0 if episode.get("is_correct", False) else 0.0
    return rewards


class RoleRewardAssigner(CreditAssigner):
    name = "role_reward"

    def compute(self, episode_group):
        per_ep_rewards = [role_rewards(ep) for ep in episode_group]
        grouped = defaultdict(list)
        # Group only by role (and primary/response status), not by RACA sigma,
        # correctness layer, or interaction channel.
        for ep_idx, (ep, rewards) in enumerate(zip(episode_group, per_ep_rewards)):
            for tid in trainable_turn_ids(ep):
                if tid not in rewards:
                    continue
                meta = turn_metadata(ep, tid)
                key = (meta["role"], meta["is_response"])
                grouped[key].append((ep_idx, tid, rewards[tid]))

        result = [{} for _ in episode_group]
        live_groups = 0
        for _key, entries in grouped.items():
            zs = zscores([x[2] for x in entries], self.delta)
            if zs is None:
                continue
            live_groups += 1
            for (ep_idx, tid, _reward), adv in zip(entries, zs):
                result[ep_idx][tid] = adv

        assigned = sum(len(x) for x in result)
        possible = sum(len(trainable_turn_ids(ep)) for ep in episode_group)
        return result, {
            "credit/role/live_groups": live_groups,
            "credit/role/coverage": assigned / max(possible, 1),
        }
