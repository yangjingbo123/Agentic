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
import os


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def question_set(path: str) -> set:
    """收集一份 jsonl 里出现过的 question（精确文本，strip 后）。"""
    questions = set()
    for item in load_jsonl(path):
        q = item.get("question", "").strip()
        if q:
            questions.add(q)
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",  default="data/math_train.jsonl")
    parser.add_argument("--sft",    default="data/sft_train.jsonl")
    # 评测集去重。hendrycks_math 的 train/test 自带 1 题重复，旧版只查 SFT
    # 没查 test，导致 math_train_rl ∩ math_test = 1（那题在 Level 3/4，不在
    # eval 取的 Level-5 池里，所以历史 eval 数字未受污染）。
    parser.add_argument("--test",   default="data/math_test.jsonl")
    parser.add_argument("--output", default="data/math_train_rl.jsonl")
    args = parser.parse_args()

    train = load_jsonl(args.train)
    sft_qs = question_set(args.sft)
    test_qs = question_set(args.test) if os.path.isfile(args.test) else set()
    print(f"主训练集: {len(train)} 条；SFT 题目: {len(sft_qs)} 个；"
          f"评测题目: {len(test_qs)} 个")

    kept, n_empty, n_leak, n_unbalanced, n_test = [], 0, 0, 0, 0
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
        if item["question"].strip() in test_qs:
            n_test += 1
            continue
        kept.append(item)

    save_jsonl(kept, args.output)
    print(f"剔除: 空答案 {n_empty}，花括号不平衡 {n_unbalanced}，"
          f"SFT 泄漏 {n_leak}，评测集重复 {n_test}")
    print(f"输出: {len(kept)} 条 → {args.output}")

    # 验收自检
    assert all(it["answer"].count("{") == it["answer"].count("}") for it in kept)
    assert all(it["answer"].strip() for it in kept)
    assert not any(it["question"].strip() in sft_qs for it in kept)
    assert not any(it["question"].strip() in test_qs for it in kept)
    print("验收通过：无空答案、花括号全平衡、与 SFT 及评测集题目零交集")


if __name__ == "__main__":
    main()
