"""
用 DeepSeek-V4-Pro 生成 SFT 训练数据。
每条数据是一个完整的多角色协作 episode，格式与 prompt_templates.py 完全一致。
"""
import json
import os
import random
import time
from openai import OpenAI

API_KEY = "sk-YOCWM1eLkRSrDRH0160c881f47B2475aA7Fe5aD8AbF48eD5"
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.vveai.com/v1")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """你是一个数学推理多智能体系统的数据生成器。
给定一道数学题和正确答案，你需要生成一个完整的多角色协作推理过程。

系统包含4个角色：
1. Controller：高层协调者，输出 <meta-plan> 标签
2. Proposer：解题专家，输出 <interaction> 标签 + 推理过程 + 最终答案
3. Critic：审查员，输出 <interaction> 标签 + 错误分析
4. Verifier：验证专家，输出 <interaction> 标签 + 分数 + 验证说明

输出一个 JSON，包含完整的对话轮次列表。每个轮次包含：
- role_name: controller/proposer/critic/verifier
- system: 该角色的系统提示
- user: 该角色收到的用户消息
- response: 该角色的输出（必须严格遵守格式）

Controller 格式：
<meta-plan>
strategy: [explore|refine|verify|stop]
focus: [proposer|critic|verifier|balanced]
reason: [一句话说明]
</meta-plan>

Proposer 格式：
<interaction>
action: [none|request_critic|request_verifier|support:<答案>|challenge:<指出的问题>]
target: [critic|verifier|none]
reason: [一句话]
</interaction>
推理过程：[逐步推导]
最终答案：[数值或表达式]

Critic 格式：
<interaction>
action: [none|request_proposer|request_verifier|support:<答案>|challenge:<指出的问题>]
target: [proposer|verifier|none]
reason: [一句话]
</interaction>
错误分析：[有错误则描述，无错误则写"无错误"]

Verifier 格式：
<interaction>
action: [none|request_proposer|request_critic|support:<答案>|challenge:<指出的问题>]
target: [proposer|critic|none]
reason: [一句话]
</interaction>
分数: [0.0-1.0]
验证说明：[简要说明]

生成要求：
- 推理过程要真实、正确，最终答案必须与给定答案一致
- 交互动作要多样，不要全是 none，适当加入 request_verifier、support 等
- 生成 1-2 轮协作（不要太长），最后 Controller 输出 stop
- 输出纯 JSON，不要有其他文字
"""

USER_TEMPLATE = """数学题：{question}
正确答案：{answer}

请生成一个完整的多角色协作推理 episode，输出格式：
{{
  "question": "题目",
  "answer": "正确答案",
  "turns": [
    {{"role_name": "controller", "system": "...", "user": "...", "response": "..."}},
    {{"role_name": "proposer", "system": "...", "user": "...", "response": "..."}},
    ...
  ]
}}"""


def generate_episode(question: str, answer: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="DeepSeek-V4-Pro",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        question=question, answer=answer
                    )},
                ],
                temperature=0.7,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            if "turns" in data and len(data["turns"]) >= 2:
                return data
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/math_train.jsonl")
    parser.add_argument("--output", default="data/sft_train.jsonl")
    parser.add_argument("--n", type=int, default=500, help="Number of examples to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not API_KEY:
        print("Error: set OPENAI_API_KEY environment variable")
        return

    random.seed(args.seed)
    with open(args.input) as f:
        dataset = [json.loads(line) for line in f]

    samples = random.sample(dataset, min(args.n, len(dataset)))
    print(f"Generating {len(samples)} SFT examples...")

    success = 0
    with open(args.output, "w") as out:
        for i, item in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] {item['question'][:60]}...", end=" ", flush=True)
            episode = generate_episode(item["question"], item["answer"])
            if episode:
                out.write(json.dumps(episode, ensure_ascii=False) + "\n")
                out.flush()
                success += 1
                print(f"OK (turns={len(episode['turns'])})")
            else:
                print("FAILED")
            time.sleep(0.5)  # rate limit

    print(f"\nDone: {success}/{len(samples)} examples saved to {args.output}")


if __name__ == "__main__":
    main()
