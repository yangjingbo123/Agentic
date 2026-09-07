"""Executor adapters used only by baseline jobs."""

from agents.agentic_executor import AgenticExecutor


class BaselineAgenticExecutor(AgenticExecutor):
    """Production four-role rollout with gold attached for baseline rewards.

    The rollout policy and blackboard are inherited unchanged.  Attaching the
    gold answer outside the production executor lets the role-reward baseline
    compute its simple local labels without editing RACA files.
    """

    def run_episodes_batch(self, questions, correct_answers, eps_force=0.0):
        episodes = super().run_episodes_batch(
            questions, correct_answers, eps_force=eps_force)
        for episode, gold in zip(episodes, correct_answers):
            episode["baseline_gold"] = gold
        return episodes
