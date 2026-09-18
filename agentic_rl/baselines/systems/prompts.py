"""Prompts used only by system-level baselines.

Fixed-role baselines reuse the production role system prompts verbatim so their
SFT adapters see the format on which they were trained. The pipeline ignores
any generated routing action and executes the predetermined next role.
"""

from llm.prompt_templates import PromptTemplates
from agents.parsing import parse_reasoning

NO_INTERACTION = """<interaction>
action: none
target: none
reason: baseline uses no adaptive interaction
</interaction>"""

SINGLE_SYSTEM = PromptTemplates.proposer_system()

FEEDBACK_SYSTEM = """你是同一个数学解题模型的自我审查阶段。请检查给定解法中的逻辑、计算和最终答案，指出最关键的问题；如果没有发现问题，明确说明。最后输出固定结束块：
<interaction>
action: none
target: none
reason: baseline uses no adaptive interaction
</interaction>
不要直接重写完整答案。"""

REFINE_SYSTEM = PromptTemplates.proposer_system()

COORDINATOR_SYSTEM = PromptTemplates.controller_system()

CRITIC_SYSTEM = PromptTemplates.critic_system()

CORRECTION_SYSTEM = PromptTemplates.proposer_system()

VERIFIER_SYSTEM = PromptTemplates.verifier_system()

PROPOSAL_VOTING_SYSTEM = PromptTemplates.proposer_system() + """

本次运行属于 proposer-only 的候选投票基线。你的唯一任务是生成一份完整、可独立检查的候选解答，而不是评价、批评或打分其他候选。

要求：
1. 从题目条件重新推导，不要机械复制已有答案；
2. 可以参考候选池中的分歧，但必须自行完成计算；
3. 即使同意已有答案，也必须给出独立推理；
4. 严格使用“推理过程：”和“最终答案：”输出；
5. 输出末尾使用固定块：
<interaction>
action: none
target: none
reason: proposal-only baseline
</interaction>"""


def single_user(question):
    return "问题：%s\n请独立求解。" % question


def feedback_user(question, answer_text):
    return "问题：%s\n候选解答：\n%s\n请审查。" % (question, answer_text)


def refine_user(question, answer_text, feedback):
    return ("问题：%s\n原解答：\n%s\n自我反馈：\n%s\n"
            "请给出修正后的完整解答。" % (question, answer_text, feedback))


def coordinator_user(question, answer_text):
    return "问题：%s\n当前候选解答：\n%s\n请判断下一步。" % (question, answer_text)


def critic_user(question, answer_text):
    return "问题：%s\n待审查解法：\n%s\n请审查。" % (question, answer_text)


def correction_user(question, answer_text, critique):
    return ("问题：%s\n原解答：\n%s\n审查意见：\n%s\n"
            "请给出修正后的完整解答。" % (question, answer_text, critique))


def verifier_user(question, answer_text):
    return "问题：%s\n待验证解答：\n%s\n请独立核验。" % (question, answer_text)


def independent_proposal_user(question, proposal_index):
    return ("问题：%s\n这是第 %d 个独立候选。请在不参考其他候选的情况下，"
            "从头生成一份完整的新解答。" % (question, proposal_index))


def _reasoning_excerpt(text, max_chars):
    reasoning, _answer = parse_reasoning(text)
    reasoning = (reasoning or text or "").strip()
    max_chars = max(80, int(max_chars))
    if len(reasoning) <= max_chars:
        return reasoning
    half = max_chars // 2
    return reasoning[:half] + "\n...[中间推理省略]...\n" + reasoning[-half:]


def proposal_pool_user(question, candidates, proposal_index,
                       max_reasoning_chars=600, tie_break=False):
    blocks = []
    for idx, candidate in enumerate(candidates, 1):
        answer = candidate.get("answer") or "[未解析出答案]"
        excerpt = _reasoning_excerpt(
            candidate.get("text", ""), max_reasoning_chars)
        blocks.append(
            "Candidate %d\n推理摘要：%s\n最终答案：%s" %
            (idx, excerpt, answer))
    pool = "\n\n".join(blocks) if blocks else "[候选池为空]"
    purpose = (
        "当前候选出现平票。请重新独立求解并提供一张新的决胜选票。"
        if tie_break else
        "请利用候选之间的信息和分歧重新独立求解，但不要直接服从多数答案。"
    )
    return (
        "问题：%s\n\n已有候选池：\n%s\n\n%s\n"
        "这是第 %d 个候选；必须输出完整的新推理和最终答案。" %
        (question, pool, purpose, proposal_index)
    )
