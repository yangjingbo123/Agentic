"""CPU-only preflight for baseline experiment definitions."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.registry import CREDIT_ASSIGNERS

TRAINABLE = set(CREDIT_ASSIGNERS)
SYSTEMS = {"sft_cot", "self_consistency", "self_refine", "fixed_four_role"}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=sorted(TRAINABLE | SYSTEMS))
    args = parser.parse_args()
    if args.method == "raca":
        raise SystemExit("RACA is not a baseline")

    math = [x for x in rows(ROOT / "data/math_test.jsonl")
            if x.get("level") == "Level 5"][:1000]
    aime = rows(ROOT / "data/aime_2022_2026.jsonl")
    assert len(math) == 1000, len(math)
    assert len(aime) == 150, len(aime)
    assert Counter(x["year"] for x in aime) == {
        2022: 30, 2023: 30, 2024: 30, 2025: 30, 2026: 30}
    cfg = ROOT / "baselines/configs" / (args.method + ".yaml")
    assert cfg.is_file(), cfg
    forbidden = [ROOT / "baselines/configs/raca.yaml"]
    assert not any(p.exists() for p in forbidden)
    print("baseline preflight OK: method=%s MATH-L5=%d AIME=%d" %
          (args.method, len(math), len(aime)))


if __name__ == "__main__":
    main()
