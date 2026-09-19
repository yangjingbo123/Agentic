"""Non-invasive entrypoint for the RACA component ablations.

This runner touches **no** existing RACA file. It swaps the module-level
``compute_raca_advantages`` referenced by ``training.grpo_trainer`` for an
ablated variant (see ``training/raca_ablations.py``) and then executes the
unmodified ``train.py`` **as ``__main__``** via :mod:`runpy`, so Hydra resolves
``config_path="configs"`` exactly as in ``python train.py`` and training,
evaluation, logging, and checkpointing behave identically.

Why runpy instead of ``import train; train.main()``: Hydra resolves a relative
``config_path`` by directory only when the decorated function runs as
``__main__``. Importing ``train`` makes ``main.__module__ == "train"``, which
pushes Hydra onto module-based config resolution and fails with
"Primary config module 'configs' not found". Running train.py under
``run_name="__main__"`` restores the directory-based resolution.

Usage (Hydra CLI args pass straight through)::

    RACA_ABLATION=no_role    python train_ablation.py exp_name=abl_no_role
    RACA_ABLATION=no_context python train_ablation.py exp_name=abl_no_context
    RACA_ABLATION=no_channel python train_ablation.py exp_name=abl_no_channel

If ``RACA_ABLATION`` is unset or empty, this behaves identically to
``python train.py`` (no patch applied).
"""

import os
import runpy

# Import the trainer module so the module object exists before we patch its
# ``compute_raca_advantages`` global. train.py imports GRPOAgenticTrainer lazily
# inside main(); it resolves to this same cached, patched module.
import training.grpo_trainer as _grpo_trainer
from training.raca_ablations import make_ablated_advantage_fn


def _apply_ablation_patch():
    ablation = os.environ.get("RACA_ABLATION", "").strip()
    if not ablation:
        print("[ablation] RACA_ABLATION unset -> running unmodified RACA",
              flush=True)
        return
    _grpo_trainer.compute_raca_advantages = make_ablated_advantage_fn(ablation)
    print(f"[ablation] patched compute_raca_advantages -> mode={ablation!r}",
          flush=True)


if __name__ == "__main__":
    _apply_ablation_patch()
    _train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    runpy.run_path(_train_py, run_name="__main__")
