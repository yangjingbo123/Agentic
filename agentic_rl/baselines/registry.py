"""Baseline registry.  RACA is intentionally not registered here."""

from baselines.credit.at_grpo import ATGRPOAssigner
from baselines.credit.gigpo import GiGPOAssigner
from baselines.credit.outcome_grpo import OutcomeGRPOAssigner
from baselines.credit.role_reward import RoleRewardAssigner


CREDIT_ASSIGNERS = {
    "outcome_grpo": OutcomeGRPOAssigner,
    "role_reward": RoleRewardAssigner,
    "at_grpo": ATGRPOAssigner,
    "gigpo": GiGPOAssigner,
    "single_agent_grpo": OutcomeGRPOAssigner,
    "fixed_four_role_grpo": OutcomeGRPOAssigner,
}


def build_credit_assigner(name, config=None):
    key = str(name).strip().lower()
    if key == "raca":
        raise ValueError("RACA is not a baseline and must use the existing train.py")
    if key not in CREDIT_ASSIGNERS:
        raise ValueError(
            "unknown credit baseline %r; choose one of %s" %
            (name, ", ".join(sorted(CREDIT_ASSIGNERS))))
    return CREDIT_ASSIGNERS[key](config)
