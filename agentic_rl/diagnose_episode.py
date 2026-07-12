"""诊断 seq_input_ids 是否有重复 prompt，不用 vLLM"""
import sys
sys.path.insert(0, '/cephfs/volumes/hpc_home/k24104674/aed22256-9e0b-4f4f-86c1-c56793988876/jingbo/marl/Agentic/agentic_rl')

if __name__ == '__main__':
    from collections import Counter
    from llm.trainable_llm import load_trainable_models
    from agents.agentic_executor import AgenticExecutor

    model_path = "/scratch/users/k24104674/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    model, tokenizer = load_trainable_models(model_path, sft_checkpoint="checkpoints/sft")

    call_count = [0]

    class MockVLLM:
        def generate(self, role, prompt):
            call_count[0] += 1
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if role == "controller":
                text = "<meta-plan>\nstrategy: explore\nfocus: proposer\nreason: test\n</meta-plan>"
            elif role == "proposer":
                text = "<interaction>\naction: none\ntarget: none\nreason: test\n</interaction>\n推理过程：16-3-4=9, 9*2=18\n最终答案：18"
            else:
                text = "<interaction>\naction: none\ntarget: none\nreason: ok\n</interaction>\n无错误"
            ids = tokenizer.encode(text, add_special_tokens=False)
            lps = [-1.0] * len(ids)
            print(f"  call {call_count[0]}: role={role} prompt_len={len(prompt_ids)} resp_len={len(ids)}", flush=True)
            return text, lps, ids

    config = {"max_tokens": 256, "max_rounds": 2, "max_interactions": 1}
    executor = AgenticExecutor(model, tokenizer, config, vllm_engine=MockVLLM())

    ep = executor.run_episode(
        "Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4. Sells rest at $2. How much per day?",
        "18"
    )

    seq  = ep["seq_input_ids"]
    step = ep["seq_step_ids"]
    lps  = ep["log_probs"]

    n_resp = sum(1 for s in step if s >= 0)
    n_prompt = sum(1 for s in step if s < 0)
    print(f"\nseq_len={len(seq)} prompt_tokens={n_prompt} resp_tokens={n_resp} log_probs={len(lps)}")
    print(f"match: {n_resp == len(lps)}")
    print(f"is_correct={ep['is_correct']} final_answer={repr(ep['final_answer'])}")

    # 检查序列是否有异常重复（相邻100个token的重复率）
    if len(seq) > 0:
        unique = len(set(seq)) / len(seq)
        print(f"token uniqueness ratio: {unique:.3f} (low = lots of repetition)")
