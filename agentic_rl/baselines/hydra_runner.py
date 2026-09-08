"""Compose the production config from an absolute path and run its task body.

Primus deploys a ZIP archive without Python-package metadata for ``configs``.
Calling the Hydra-decorated ``train.main`` after importing it therefore makes
Hydra search for a nonexistent ``configs`` Python package.  Baseline entry
points instead compose the same YAML tree from its absolute directory, preserve
all command-line overrides, and invoke the unchanged production task function.
"""

import os
import random
import sys

from hydra import compose, initialize_config_dir


CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs"))


def _seed_process():
    seed = int(os.environ.get("BASELINE_SEED", "1"))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Primus dependency bootstrap runs before this entry point.
        pass


def _cli_overrides(argv):
    overrides = []
    for value in argv:
        if value in ("-h", "--help"):
            print("Baseline training accepts Hydra key=value overrides.")
            print("Config directory:", CONFIG_DIR)
            raise SystemExit(0)
        if value.startswith("-") and "=" not in value:
            raise SystemExit("unsupported baseline training option: %s" % value)
        overrides.append(value)
    return overrides


def run_production_train(executor_cls):
    """Patch process-local classes, compose config, and call train.py's body."""
    _seed_process()

    import agents.agentic_executor as executor_module
    import training.grpo_trainer as production_trainer

    # train.py imports AgenticExecutor again for evaluation and shard workers;
    # its trainer constructor also resolves this module-level binding at runtime.
    executor_module.AgenticExecutor = executor_cls
    production_trainer.AgenticExecutor = executor_cls

    from baselines.trainer import BaselineGRPOTrainer
    production_trainer.GRPOAgenticTrainer = BaselineGRPOTrainer

    import train as production_train
    task_function = getattr(production_train.main, "__wrapped__", None)
    if task_function is None:
        raise RuntimeError("train.main no longer exposes its Hydra task function")

    overrides = _cli_overrides(sys.argv[1:])
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return task_function(cfg)
