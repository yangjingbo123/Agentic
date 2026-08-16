"""grader.math_equal 回归测试。

运行：python test_grader.py 或 pytest test_grader.py
用例大量取自 data/math_test.jsonl 的真实答案形态。
"""

from agents.grader import math_equal


# (a, b, expected)
CASES = [
    # ── 旧实现的假阳性：必须判 False（核心回归目标） ──────────────────────
    ("2\\sqrt{3}", "3", False),                 # 旧："取最后一个数字" → 3 == 3
    ("x^2+4x+4", "4", False),
    ("9\\pi", "9", False),
    ("\\sqrt{13}", "13", False),
    ("4\\pi - 2\\sqrt{3}", "3", False),
    ("(9,11)", "11", False),
    ("\\frac{2\\sqrt{53}+53}{53}", "53", False),
    ("2x^9 - 8x^7 + 9x^6", "9", False),

    # ── 旧实现的假阴性方向：符号一致时必须判 True ────────────────────────
    ("\\frac{\\pi}{2}", "\\frac{\\pi}{2}", True),
    ("\\dfrac{9}{7}", "\\frac{9}{7}", True),            # dfrac 归一
    ("\\tfrac{1}{3}", "\\frac{1}{3}", True),            # tfrac 归一
    ("\\left[\\frac{1}{2}, \\frac{4}{3}\\right]", "[\\frac{1}{2},\\frac{4}{3}]", True),
    ("\\left(-\\sqrt{3}, \\sqrt{3}\\right)", "(-\\sqrt{3},\\sqrt{3})", True),
    ("14 \\pi", "14\\pi", True),                        # 空格
    ("\\sqrt2", "\\sqrt{2}", True),                     # sqrt 简写
    ("\\frac12", "\\frac{1}{2}", True),                 # frac 简写
    ("\\frac{68}{3}\\text{ pounds}", "\\frac{68}{3}", True),   # 右侧单位
    ("\\$40", "40", True),                              # 美元符号
    ("40\\%", "40", True),                              # 百分号
    ("x \\in [-2,7]", "[-2,7]", True),                  # "x = ..." 前缀类（k= 剥离）
    ("12, 10, 6", "12,10,6", True),                     # 列表空格
    ("(-\\infty,-8)\\cup (8,\\infty)", "(-\\infty,-8)\\cup(8,\\infty)", True),

    # ── 数值等价（严格 fullmatch 路径） ──────────────────────────────────
    ("0.5", "\\frac{1}{2}", True),
    ("1/2", "\\frac{1}{2}", True),
    ("-\\frac{1}{8}", "-0.125", True),
    ("3,600", "3600", True),                            # 千分位
    ("42", "42.0", True),
    ("\\text{7}", "7", True),
    ("7", "8", False),
    ("0.5", "0.6", False),

    # ── 模型输出常见包裹 ─────────────────────────────────────────────────
    ("\\boxed{42}", "42", True),
    ("$\\frac{3}{4}$", "\\frac{3}{4}", True),
    ("**42**", "42", True),
    ("42。", "42", True),

    # ── 顺序/结构敏感：不同就是不同 ──────────────────────────────────────
    ("(9,11)", "(11,9)", False),
    ("(2,12)", "(2,11)", False),
    ("i", "1", False),
    ("\\text{(C)}", "\\text{(B)}", False),

    # ── 边界 ─────────────────────────────────────────────────────────────
    ("", "42", False),
    (None, "42", False),
    ("nan", "nan", True),                               # 不走数值路径，但同字符串判等
    ("nan", "0", False),                                # float("nan") 不算数值
]


def test_math_equal():
    failures = []
    for a, b, expected in CASES:
        got = math_equal(a, b)
        if got != expected:
            failures.append(f"  math_equal({a!r}, {b!r}) = {got}, expected {expected}")
    assert not failures, "\n" + "\n".join(failures)


def test_symmetry():
    """判等必须对称。"""
    for a, b, expected in CASES:
        if a is None or b is None:
            continue
        assert math_equal(b, a) == expected, f"asymmetric: ({a!r}, {b!r})"


if __name__ == "__main__":
    test_math_equal()
    test_symmetry()
    print(f"All {len(CASES)} cases passed (+ symmetry).")
