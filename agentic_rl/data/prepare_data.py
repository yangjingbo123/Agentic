"""
下载并转换 GSM8K 和 MATH 数据集为统一格式：
{"question": "...", "answer": "..."}
"""
import json
import re
import os
from datasets import load_dataset


def extract_gsm8k_answer(solution: str) -> str:
    """GSM8K答案在####后面"""
    m = re.search(r"####\s*(.+)", solution)
    return m.group(1).strip().replace(",", "") if m else ""


def extract_math_answer(solution: str) -> str:
    """MATH答案在\boxed{}里"""
    m = re.search(r"\\boxed\{([^}]+)\}", solution)
    return m.group(1).strip() if m else ""


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
    """只取Level 3-5（难度适中，适合主实验）"""
    ds = load_dataset("lighteval/MATH", "all")
    for split, path in [("train", "data/math_train.jsonl"),
                        ("test",  "data/math_test.jsonl")]:
        data = [{"question": item["problem"],
                 "answer":   extract_math_answer(item["solution"]),
                 "level":    item["level"]}
                for item in ds[split]
                if int(item["level"].replace("Level ", "")) in levels]
        save_jsonl(data, path)


if __name__ == "__main__":
    print("Preparing GSM8K...")
    prepare_gsm8k()
    print("Preparing MATH Level 3-5...")
    prepare_math()
    print("Done.")
