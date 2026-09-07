"""Launch the unchanged four-role training loop with a baseline trainer.

This wrapper monkey-patches only the trainer class imported by ``train.main``.
The production RACA source files remain byte-for-byte untouched.
"""

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
    name = os.environ.get("BASELINE_NAME", "").strip().lower()
    if not name:
        raise SystemExit("BASELINE_NAME is required")
    if name == "raca":
        raise SystemExit("RACA must use the existing submit_primus.sh; it is not rerun here")

    from baselines.executor import BaselineAgenticExecutor
    from baselines.trainer import BaselineGRPOTrainer
    import agents.agentic_executor as executor_module
    import training.grpo_trainer as production_trainer

    # Patch only inside this baseline process.  train.py imports AgenticExecutor
    # again for sharded rollout workers and evaluation, so both module bindings
    # must point at the metadata-preserving subclass. Source files are untouched.
    executor_module.AgenticExecutor = BaselineAgenticExecutor
    production_trainer.AgenticExecutor = BaselineAgenticExecutor
    production_trainer.GRPOAgenticTrainer = BaselineGRPOTrainer
    from train import main as hydra_main
    hydra_main()


if __name__ == "__main__":
    main()
