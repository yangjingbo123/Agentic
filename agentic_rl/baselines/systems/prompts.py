"""Prompts used only by system-level baselines.

Fixed-role baselines reuse the production role system prompts verbatim so their
SFT adapters see the format on which they were trained. The pipeline ignores
any generated routing action and executes the predetermined next role.
"""

from llm.prompt_templates import PromptTemplates

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
