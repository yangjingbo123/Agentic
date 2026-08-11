"""从主训练文件派生 RL 训练集（math_train_rl.jsonl）。

历史背景（见 RACA_ALGORITHM.md「遗留问题」节）：
旧的 math_train_rl.jsonl 是一次性手工派生的，仓库里没有生成脚本。
2026-08-11 实测破案：其派生语义是 "RL 训练集 = 主训练集 − SFT 用过的题"
（防止 SFT 数据泄漏进 RL 训练）。本脚本把该规则固化入库。

派生规则：
1. 过滤空 answer —— 修复后的 extract_math_answer 对花括号不平衡的
   \\boxed{...} 返回空串，这类条目没有可验证的 ground truth，必须剔除；
2. 剔除 sft_train.jsonl 中实际出现的 question（精确匹配）—— SFT 阶段
   见过的题不能再进 RL 训练集，否则奖励信号被记忆污染。

用法：
    python data/derive_rl_split.py
    python data/derive_rl_split.py --train data/math_train.jsonl \
        --sft data/sft_train.jsonl --output data/math_train_rl.jsonl
"""
import argparse
import json


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def sft_questions(sft_path: str) -> set:
    """收集 SFT 数据中实际出现过的 question（精确文本）。"""
    questions = set()
    for item in load_jsonl(sft_path):
        q = item.get("question", "").strip()
        if q:
            questions.add(q)
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",  default="data/math_train.jsonl")
    parser.add_argument("--sft",    default="data/sft_train.jsonl")
    parser.add_argument("--output", default="data/math_train_rl.jsonl")
    args = parser.parse_args()

    train = load_jsonl(args.train)
    sft_qs = sft_questions(args.sft)
    print(f"主训练集: {len(train)} 条；SFT 题目: {len(sft_qs)} 个")

    kept, n_empty, n_leak, n_unbalanced = [], 0, 0, 0
    for item in train:
        ans = item.get("answer", "").strip()
        if not ans:
            n_empty += 1
            continue
        # 防御性检查：answer 花括号必须平衡（修复后解析器不应产出，但留一道闸）
        if ans.count("{") != ans.count("}"):
            n_unbalanced += 1
            continue
        if item["question"].strip() in sft_qs:
            n_leak += 1
            continue
        kept.append(item)

    save_jsonl(kept, args.output)
    print(f"剔除: 空答案 {n_empty}，花括号不平衡 {n_unbalanced}，SFT 泄漏 {n_leak}")
    print(f"输出: {len(kept)} 条 → {args.output}")

    # 验收自检
    assert all(it["answer"].count("{") == it["answer"].count("}") for it in kept)
    assert all(it["answer"].strip() for it in kept)
    assert not any(it["question"].strip() in sft_qs for it in kept)
    print("验收通过：无空答案、花括号全平衡、与 SFT 题目零交集")


if __name__ == "__main__":
    main()
