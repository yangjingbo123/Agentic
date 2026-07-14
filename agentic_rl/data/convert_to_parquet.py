"""Convert math jsonl datasets to parquet format for veRL."""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent

SPLITS = [
    ("data/math_train_rl.jsonl",   "data/math_train_rl.parquet"),
    ("data/math_test_clean.jsonl", "data/math_test_clean.parquet"),
    ("data/gsm8k_train.jsonl",     "data/gsm8k_train.parquet"),
    ("data/gsm8k_test.jsonl",      "data/gsm8k_test.parquet"),
]

for src, dst in SPLITS:
    src_path = ROOT / src
    dst_path = ROOT / dst
    if not src_path.exists():
        print(f"SKIP {src} (not found)")
        continue
    rows = []
    with open(src_path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "question":     d["question"],
                "reward_model": {"ground_truth": d.get("answer", d.get("ground_truth", "")),
                                 "style": "rule"},
            })
    pd.DataFrame(rows).to_parquet(dst_path, index=False)
    print(f"OK  {src} → {dst}  ({len(rows)} rows)")
