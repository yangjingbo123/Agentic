# AIME evaluation data sources

The repository evaluates on 150 AIME problems (AIME I + II, 2022–2026).
All answers are stored as zero-padded three-digit strings.

- **2022–2024 (90 problems):** `AI-MO/aimo-validation-aime`, revision
  `13f9e12f613e720c2a2b2f345dd04b998a29494d`, Apache-2.0.
- **2025 (30 problems):** `math-ai/aime25`, revision
  `563bb8404243c5f09de6ec262f2db674fe5bce9b`, Apache-2.0.
- **2026 (30 problems):** `math-ai/aime26`, revision
  `79037aebdb6580008fb960d17cb21fd3099083e3`, Apache-2.0. This is the
  dataset revision pinned by the UK AI Security Institute `inspect_evals`
  AIME 2026 task.

Generated files:

- `aime_2022_2024.jsonl`
- `aime_2025.jsonl`
- `aime_2026.jsonl`
- `aime_2022_2026.jsonl`

The older `aime_test.jsonl` is retained for backward compatibility.
