"""Summarize per-item baseline JSONL files into paper-facing metrics."""

import argparse
import json
from collections import defaultdict


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows):
    n = max(len(rows), 1)
    out = {
        "method": rows[0].get("method") if rows else None,
        "suite": rows[0].get("suite") if rows else None,
        "correct": sum(bool(x.get("is_correct")) for x in rows),
        "total": len(rows),
    }
    out["accuracy"] = out["correct"] / n
    for key in ("prompt_tokens", "generated_tokens", "n_calls"):
        values = [float(x[key]) for x in rows if key in x]
        if values:
            out[key + "_per_problem"] = sum(values) / len(values)
    by_year = defaultdict(list)
    for row in rows:
        if row.get("year") is not None:
            by_year[str(row["year"])].append(bool(row.get("is_correct")))
    if by_year:
        out["by_year"] = {year: {"correct": sum(vals), "total": len(vals),
                                  "accuracy": sum(vals) / len(vals)}
                          for year, vals in sorted(by_year.items())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = [summarize(load(path)) for path in args.paths]
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
