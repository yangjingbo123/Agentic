"""
下载并转换 GSM8K / MATH / AIME 数据集为统一格式：
{"question": "...", "answer": "..."}
"""
import json
import re
import os
from datasets import load_dataset

MATH_SUBSETS = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def extract_gsm8k_answer(solution: str) -> str:
    m = re.search(r"####\s*(.+)", solution)
    return m.group(1).strip().replace(",", "") if m else ""


def extract_math_answer(solution: str) -> str:
    """Extract the content inside \\boxed{...}, correctly handling nested braces.

    The old regex r"\\boxed\{([^}]+)\}" breaks on nested braces like
    \\boxed{\\frac{1}{2}} — it captures only \\frac{1 because [^}] stops at
    the first }. This caused ~25% of MATH answers to be truncated.
    """
    marker = "\\boxed{"
    idx = solution.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    depth = 1
    pos = start
    while pos < len(solution) and depth > 0:
        ch = solution[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        pos += 1
    if depth != 0:
        return ""
    return solution[start:pos].strip()


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} items to {path}")


def prepare_gsm8k():
    ds = load_dataset("openai/gsm8k", "main")
    for split, path in [("train", "data/gsm8k_train.jsonl"),
                        ("test",  "data/gsm8k_test.jsonl")]:
        data = [{"question": item["question"],
                 "answer":   extract_gsm8k_answer(item["answer"])}
                for item in ds[split]]
        save_jsonl(data, path)


def prepare_math(levels=(3, 4, 5)):
    """合并所有学科 subset，只取 Level 3-5"""
    for split, path in [("train", "data/math_train.jsonl"),
                        ("test",  "data/math_test.jsonl")]:
        data = []
        for subset in MATH_SUBSETS:
            ds = load_dataset("EleutherAI/hendrycks_math", subset)
            for item in ds[split]:
                try:
                    lvl = int(item["level"].replace("Level ", ""))
                except ValueError:
                    continue
                if lvl in levels:
                    data.append({
                        "question": item["problem"],
                        "answer":   extract_math_answer(item["solution"]),
                        "level":    item["level"],
                    })
        save_jsonl(data, path)


def prepare_aime():
    """AIME 2024/2025，共90题，全部作为测试集"""
    ds = load_dataset("AI-MO/aimo-validation-aime")
    data = [{"question": item["problem"], "answer": str(item["answer"])}
            for item in ds["train"]]
    save_jsonl(data, "data/aime_test.jsonl")


if __name__ == "__main__":
    print("Preparing GSM8K...")
    prepare_gsm8k()
    print("Preparing MATH Level 3-5...")
    prepare_math()
    print("Preparing AIME 2024/2025...")
    prepare_aime()
    print("Done.")

