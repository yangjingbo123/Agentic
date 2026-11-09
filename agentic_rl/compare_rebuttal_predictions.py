"""Paired analysis of two evaluation JSONL files on exactly the same questions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def load_rows(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "question" not in row or "gold" not in row or "is_correct" not in row:
            raise ValueError("Requires evaluation JSONL with question, gold, is_correct")
        key = hashlib.sha256(row["question"].encode()).hexdigest()
        if key in result:
            raise ValueError("Repeated question: analyze each seed separately, not pooled as independent trials")
        result[key] = row
    if not result:
        raise ValueError("Empty evaluation file")
    return result


def paired_stats(a, b, bootstrap=5000, seed=42):
    if a.keys() != b.keys():
        raise ValueError("Question sets differ; refusing to silently compare a subset")
    keys = sorted(a)
    if any(a[k]["gold"] != b[k]["gold"] for k in keys):
        raise ValueError("Gold answers differ between evaluation files")
    n = len(keys)
    diffs = [int(a[k]["is_correct"]) - int(b[k]["is_correct"]) for k in keys]
    wins, losses = diffs.count(1), diffs.count(-1)
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, i)
        for i in range(min(wins, losses) + 1)) / (2 ** discordant)) if discordant else 1.0
    rng = random.Random(seed)
    draws = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(bootstrap))
    return dict(n=n, a_correct=sum(bool(a[k]["is_correct"]) for k in keys),
        b_correct=sum(bool(b[k]["is_correct"]) for k in keys),
        accuracy_difference=sum(diffs) / n,
        a_only_correct=wins, b_only_correct=losses,
        exact_two_sided_mcnemar_p=p,
        paired_bootstrap_95_ci=[draws[int(0.025 * (bootstrap - 1))], draws[int(0.975 * (bootstrap - 1))]],
        bootstrap_samples=bootstrap,
        interpretation="One seed/checkpoint pair; questions are the resampling unit. Not a cross-seed test.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    a, b = load_rows(args.a), load_rows(args.b)
    result = paired_stats(a, b, args.bootstrap, args.seed)
    years = sorted({str(v["year"]) for v in a.values() if v.get("year") is not None})
    result["by_year"] = {}
    for year in years:
        aa = {k: v for k, v in a.items() if str(v.get("year")) == year}
        bb = {k: b[k] for k in aa}
        result["by_year"][year] = paired_stats(aa, bb, args.bootstrap, args.seed)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
