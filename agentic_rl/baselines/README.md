# Baseline experiments

This directory is deliberately isolated from the existing RACA implementation.
The production files `train.py`, `training/grpo_trainer.py`,
`training/raca_adv.py`, `agents/raca_rewards.py`, `agents/agentic_executor.py`,
`configs/agentic/default.yaml`, and `submit_primus.sh` remain unchanged.

## Unified Primus entry point

All methods are launched through one front-end:

```bash
bash baselines/submit_primus.sh --list
bash baselines/submit_primus.sh outcome_grpo
bash baselines/submit_primus.sh role_reward
bash baselines/submit_primus.sh at_grpo
bash baselines/submit_primus.sh gigpo
bash baselines/submit_primus.sh single_agent_grpo
bash baselines/submit_primus.sh fixed_four_role_grpo
bash baselines/submit_primus.sh sft_cot
bash baselines/submit_primus.sh self_consistency
bash baselines/submit_primus.sh self_refine
bash baselines/submit_primus.sh fixed_four_role
```

The front-end dispatches trainable methods to `submit_primus_baseline.sh` and
inference-only methods to `submit_primus_system_eval.sh`. One job runs one
method so training cost, failures, and checkpoint resume remain independently
accounted. `raca` and batch aliases such as `all` are rejected.

## Trainable baselines

The credit-assignment controls reuse the production four-role rollout,
blackboard, SFT checkpoint, PPO loss, KL reference, sampling budget, and
MATH/AIME evaluation. Only advantage construction changes.

```bash
BASELINE_NAME=outcome_grpo bash baselines/submit_primus_baseline.sh
BASELINE_NAME=role_reward  bash baselines/submit_primus_baseline.sh
BASELINE_NAME=at_grpo      bash baselines/submit_primus_baseline.sh
BASELINE_NAME=gigpo        bash baselines/submit_primus_baseline.sh
```

System-level trainable controls:

```bash
BASELINE_NAME=single_agent_grpo    bash baselines/submit_primus_baseline.sh
BASELINE_NAME=fixed_four_role_grpo bash baselines/submit_primus_baseline.sh
```

`BASELINE_NAME=raca` is rejected. Existing RACA results are reused.

- `outcome_grpo`: terminal correctness normalized per question and broadcast to
  all generated turns.
- `role_reward`: simple role-local correctness/calibration rewards; no RACA
  interaction reward, causal window, layers, or token channels.
- `at_grpo`: **AT-GRPO-Adapted**; paper-form $\alpha r^{team}+r^{local}$ reward normalized by role
  and role-local turn index. The existing complete-rollout sampler is retained.
- `gigpo`: **GiGPO-Adapted**; macro terminal GRPO plus micro exact-prompt anchor
  credit using sparse terminal return-to-go. Exact anchor coverage is
  reported; a structured fallback exists only for diagnostics and is disabled.
- `single_agent_grpo`: proposer-only outcome GRPO.
- `fixed_four_role_grpo`: fixed proposer-controller-critic-correction-verifier
  pipeline trained with shared terminal outcome GRPO.

## Non-RL system baselines

```bash
METHOD=sft_cot          bash baselines/submit_primus_system_eval.sh
METHOD=self_consistency bash baselines/submit_primus_system_eval.sh
METHOD=self_refine      bash baselines/submit_primus_system_eval.sh
METHOD=fixed_four_role  bash baselines/submit_primus_system_eval.sh
```

Each evaluation writes per-item JSONL, a summary JSON, accuracy, prompt tokens,
generated tokens, and LLM calls per problem.

## Fairness rules

- Same Qwen3-8B model and SFT checkpoint.
- Same MATH Level-5 1000 and AIME 2022--2026 150 suites.
- Trainable multi-agent baselines use the same 80 steps, batch size, sample
  count, interaction topology, and maximum turns as the existing RACA run.
- Baseline experiment names must begin with `baseline_`; checkpoint manifests
  prevent cross-method or cross-config resume.
- RACA is not rerun by this package.
