"""Single-agent proposer-only GRPO training entry point."""

import os
import sys


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    seed = int(os.environ.get("BASELINE_SEED", "1"))
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ.setdefault("BASELINE_NAME", "single_agent_grpo")
    os.environ.setdefault("BASELINE_CONFIG_JSON", "{}")

    from baselines.systems.runtime import SingleAgentExecutor
    from baselines.trainer import BaselineGRPOTrainer
    import agents.agentic_executor as executor_module
    import training.grpo_trainer as production_trainer

    # Process-local substitution. Both the training and eval executors must be
    # single-agent; the production source and regular RACA entry point stay intact.
    executor_module.AgenticExecutor = SingleAgentExecutor
    production_trainer.AgenticExecutor = SingleAgentExecutor
    production_trainer.GRPOAgenticTrainer = BaselineGRPOTrainer
    from train import main as hydra_main
    hydra_main()


if __name__ == "__main__":
    main()
