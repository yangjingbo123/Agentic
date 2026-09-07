# Baseline source map

The implementations below are isolated adaptations to this project's textual
blackboard and four LoRA roles. The AT-GRPO and GiGPO arXiv papers were audited
against the formulas used here. The upstream repository HEADs used for this
implementation audit are pinned below.

| Baseline | Primary reference | Paper-facing status |
|---|---|---|
| Outcome GRPO | DeepSeekMath / GRPO | Native control |
| Role-wise reward | Project-defined simple shaping | Native control |
| AT-GRPO | Stronger-MAS / PettingLLMs | **AT-GRPO-Adapted** |
| GiGPO | Group-in-Group Policy Optimization / verl-agent | **GiGPO-Adapted**: exact prompt-state sparse-return implementation |
| Self-Consistency | Wang et al., 2022 | Standard implementation |
| Self-Refine | Madaan et al., 2023 | One-cycle implementation |

Upstream repository HEADs pinned on 2026-09-07:

- AT-GRPO / PettingLLMs: `a054fb18b83f4dc7c1d49aba3b71669d1f93ca71`
- GiGPO / verl-agent: `20bd331bdbc9026a5668e11362178e10ab7400c8`
- Self-Refine: `9a206d41e5d2d0c241bb441f41eeadb945afaa55`

Do not remove the `-Adapted` suffix unless the upstream implementation is
reproduced exactly, including its sampler, state grouping, and reward semantics.
