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
    """AIME I/II 2022--2026，共150题，全部作为测试集。

    2022--2024 使用 AI-MO validation 集；2025/2026 使用 Math-AI 发布集。
    revision 与 data/AIME_SOURCES.md 对齐，避免上游更新造成静默漂移。
    """
    specs = [
        ("AI-MO/aimo-validation-aime", "train",
         "13f9e12f613e720c2a2b2f345dd04b998a29494d", None),
        ("math-ai/aime25", "test",
         "563bb8404243c5f09de6ec262f2db674fe5bce9b", 2025),
        ("math-ai/aime26", "test",
         "79037aebdb6580008fb960d17cb21fd3099083e3", 2026),
    ]
    all_data = []
    for dataset_name, split, revision, fixed_year in specs:
        ds = load_dataset(dataset_name, revision=revision)[split]
        part = []
        for item in ds:
            if fixed_year is None:
                m = re.search(
                    r"/(20\d{2})_AIME_(I|II)_Problems/Problem_(\d+)",
                    item["url"],
                )
                if not m:
                    raise ValueError(f"无法从 URL 解析 AIME 元数据: {item['url']}")
                year, exam, problem = int(m.group(1)), m.group(2), int(m.group(3))
            else:
                idx = int(item["id"])
                # aime25 id=0..29；aime26 id=1..30。
                zero = idx if fixed_year == 2025 else idx - 1
                year = fixed_year
                exam = "I" if zero < 15 else "II"
                problem = zero + 1 if zero < 15 else zero - 14
            part.append({
                "question": item["problem"],
                "answer": str(item["answer"]).zfill(3),
                "year": year,
                "exam": exam,
                "problem": problem,
                "source": dataset_name,
                "source_revision": revision,
            })
        part.sort(key=lambda x: (x["year"], x["exam"], x["problem"]))
        save_jsonl(part, f"data/aime_{fixed_year or '2022_2024'}.jsonl")
        all_data.extend(part)

    if len(all_data) != 150 or len({x["question"] for x in all_data}) != 150:
        raise ValueError("AIME 2022--2026 应恰好包含 150 道唯一题目")
    save_jsonl(all_data, "data/aime_2022_2026.jsonl")


if __name__ == "__main__":
    print("Preparing GSM8K...")
    prepare_gsm8k()
    print("Preparing MATH Level 3-5...")
    prepare_math()
    print("Preparing AIME 2022--2026...")
    prepare_aime()
    print("Done.")
