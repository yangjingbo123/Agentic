"""Outcome-GRPO training for a fixed five-call four-role pipeline."""

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
    os.environ.setdefault("BASELINE_NAME", "fixed_four_role_grpo")
    os.environ.setdefault("BASELINE_CONFIG_JSON", "{}")

    from baselines.systems.runtime import FixedPipelineExecutor
    from baselines.trainer import BaselineGRPOTrainer
    import agents.agentic_executor as executor_module
    import training.grpo_trainer as production_trainer

    executor_module.AgenticExecutor = FixedPipelineExecutor
    production_trainer.AgenticExecutor = FixedPipelineExecutor
    production_trainer.GRPOAgenticTrainer = BaselineGRPOTrainer
    from train import main as hydra_main
    hydra_main()


if __name__ == "__main__":
    main()
