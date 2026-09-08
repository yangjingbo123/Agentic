"""Single-agent proposer-only GRPO training entry point."""

import os
import sys


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault("BASELINE_NAME", "single_agent_grpo")
    os.environ.setdefault("BASELINE_CONFIG_JSON", "{}")

    from baselines.hydra_runner import run_production_train
    from baselines.systems.runtime import SingleAgentExecutor
    return run_production_train(SingleAgentExecutor)


if __name__ == "__main__":
    main()
