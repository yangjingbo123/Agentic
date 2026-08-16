"""MATH 答案判等 grader（纯字符串，零第三方依赖）。

移植自 verl/utils/reward_score/math_reward.py（Apache 2.0），
其源头是 Hendrycks MATH / lm-evaluation-harness 官方判分器。

math_equal 分层判等策略：
1. 清洗（去 \\boxed、markdown 加粗、$、尾部标点）
2. 两边【整个字符串】都能严格解析为数值 → 容差数值比较
   （fullmatch，绝不从符号表达式里抠数字）
3. Hendrycks strip_string 归一化 → 严格字符串比较
4. 无法证明等价 → False

替代旧实现的动机：旧 _extract_number 的"取最后一个数字"兜底会把
"2\\sqrt{3}" 压成 3 造成奖励假阳性，把 "\\frac{\\pi}{2}" 压成 2 造成
假阴性——Level 5 的符号型答案受害最重（详见 RACA_ALGORITHM.md）。
"""

import math
import re


# ── Hendrycks strip_string 及其辅助函数（原样移植） ─────────────────────────

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # X/Y --> \frac{X}{Y} for simple integer cases
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2) -> bool:
    if str1 is None or str2 is None:
        return False
    try:
        return strip_string(str1) == strip_string(str2)
    except Exception:
        return str1 == str2


# ── \boxed 提取（清洗模型输出用） ────────────────────────────────────────────

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


# ── 严格数值解析（fullmatch，替代旧的"取最后一个数字"） ──────────────────────

_FRAC_RE  = re.compile(r"(-?)\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}")
_SLASH_RE = re.compile(r"(-?[\d.]+)\s*/\s*(-?[\d.]+)")
_TEXT_RE  = re.compile(r"\\text\{([^{}]+)\}")


def _parse_number(s: str, _depth: int = 0):
    """整个字符串是数值时返回 float，否则返回 None。

    支持：整数/小数、千分位逗号、\\frac{a}{b}（a、b 为数值）、a/b、
    \\text{...} 包裹。绝不从符号表达式中抽取局部数字。
    """
    if _depth > 2:
        return None
    s = s.strip().strip("$").strip()
    if not s:
        return None
    m = _TEXT_RE.fullmatch(s)
    if m:
        return _parse_number(m.group(1), _depth + 1)
    t = s.replace(",", "")
    try:
        v = float(t)
        # 拒绝 "nan"/"inf" 这类能被 float() 接受的非数值答案
        return v if math.isfinite(v) else None
    except ValueError:
        pass
    m = _FRAC_RE.fullmatch(s)
    if m:
        try:
            v = float(m.group(2)) / float(m.group(3))
            return -v if m.group(1) else v
        except (ValueError, ZeroDivisionError):
            return None
    m = _SLASH_RE.fullmatch(s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    return None


# ── 对外主入口 ───────────────────────────────────────────────────────────────

def _clean(s: str) -> str:
    """判等前的轻量清洗：\\boxed 提取、markdown 加粗、$、尾部标点。"""
    s = s.strip()
    boxed = last_boxed_only_string(s)
    if boxed is not None:
        s = remove_boxed(boxed).strip()
    s = s.strip("*").strip()          # markdown **42**
    s = s.strip("$").strip()
    s = re.sub(r"^[a-zA-Z]\s*\\in\s*", "", s)   # "x \in [-2,7]" → "[-2,7]"
    s = s.rstrip("。．.,，;；:：!！")    # 尾部标点（不影响 "0.5" 这类小数）
    return s.strip()


def math_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    """MATH 答案判等：严格数值比较 → Hendrycks 归一化字符串比较。

    无法证明等价的一律 False（宁可漏判，不做"最后一个数字"式的猜测）。
    """
    if a is None or b is None:
        return False
    a, b = _clean(str(a)), _clean(str(b))
    if not a or not b:
        return False
    va, vb = _parse_number(a), _parse_number(b)
    if va is not None and vb is not None:
        return abs(va - vb) < tol
    return is_equiv(a, b)
