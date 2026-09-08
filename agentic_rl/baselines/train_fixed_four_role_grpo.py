"""Outcome-GRPO training for the fixed five-call four-role pipeline."""

import os
import sys


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault("BASELINE_NAME", "fixed_four_role_grpo")
    os.environ.setdefault("BASELINE_CONFIG_JSON", "{}")

    from baselines.hydra_runner import run_production_train
    from baselines.systems.runtime import FixedPipelineExecutor
    return run_production_train(FixedPipelineExecutor)


if __name__ == "__main__":
    main()
