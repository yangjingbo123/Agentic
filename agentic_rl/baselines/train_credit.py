"""Four-role credit-assignment baseline training entry point."""

import os
import sys


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    name = os.environ.get("BASELINE_NAME", "").strip().lower()
    if not name:
        raise SystemExit("BASELINE_NAME is required")
    if name == "raca":
        raise SystemExit("RACA must use the existing submit_primus.sh; it is not rerun here")

    from baselines.executor import BaselineAgenticExecutor
    from baselines.hydra_runner import run_production_train
    return run_production_train(BaselineAgenticExecutor)


if __name__ == "__main__":
    main()
