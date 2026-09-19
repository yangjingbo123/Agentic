"""Non-invasive entrypoint for the RACA component ablations.

This runner touches **no** existing RACA file. It swaps the module-level
``compute_raca_advantages`` referenced by ``training.grpo_trainer`` for an
ablated variant (see ``training/raca_ablations.py``) and then delegates to the
unmodified ``train.main`` Hydra entrypoint, so training, evaluation, logging,
and checkpointing behave exactly as in a normal RACA run.

Usage (Hydra CLI args pass straight through to ``train.main``)::

    # Ablate role conditioning
    RACA_ABLATION=no_role    python train_ablation.py exp_name=abl_no_role

    # Ablate semantic-context conditioning
    RACA_ABLATION=no_context python train_ablation.py exp_name=abl_no_context

    # Ablate functional-channel decomposition (single turn-level advantage).
    # Equivalent config-only route: python train.py agentic.token_credit=false
    RACA_ABLATION=no_channel python train_ablation.py exp_name=abl_no_channel

If ``RACA_ABLATION`` is unset or empty, this behaves identically to
``python train.py`` (no patch applied).
"""

import os

# Import the trainer module so the module object exists before we patch its
# ``compute_raca_advantages`` global (grpo_trainer resolves the name at call
# time, so replacing the module attribute reroutes the call site).
import training.grpo_trainer as _grpo_trainer
from training.raca_ablations import make_ablated_advantage_fn

import train


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
    train.main()
