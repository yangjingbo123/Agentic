"""CPU-only contract tests for baseline algorithms and isolation."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.credit.at_grpo import ATGRPOAssigner
from baselines.credit.gigpo import GiGPOAssigner
from baselines.credit.outcome_grpo import OutcomeGRPOAssigner
from baselines.credit.role_reward import RoleRewardAssigner
from baselines.registry import build_credit_assigner
from baselines.systems.runtime import majority_answer


def approx(a, b, eps=1e-7):
    return abs(float(a) - float(b)) <= eps


def msg(tid, role, text="x", prompt=None):
    return {"turn_id": tid, "role_name": role, "response": text,
            "response_ids": [10 + tid], "prompt_text": prompt or (role + str(tid))}


def episode(correct, messages, meta=None, gold="1"):
    td = {}
    for i, m in enumerate(messages):
        data = {"role": m["role_name"], "round": i // 4,
                "sigma": "verify", "is_response": i > 0, "reward": 0.0}
        if meta and m["turn_id"] in meta:
            data.update(meta[m["turn_id"]])
        td[m["turn_id"]] = data
    return {"is_correct": correct, "messages": messages,
            "raca_turn_data": td, "baseline_gold": gold}


def test_registry_rejects_raca():
    try:
        build_credit_assigner("raca")
    except ValueError as exc:
        assert "not a baseline" in str(exc)
    else:
        raise AssertionError("RACA must never be launched as a baseline")


def test_outcome_grpo_broadcasts_one_trajectory_advantage():
    group = [episode(True, [msg(0, "proposer"), msg(1, "critic")]),
             episode(False, [msg(0, "proposer"), msg(1, "critic")])]
    adv, diag = OutcomeGRPOAssigner().compute(group)
    assert adv[0][0] > 0 and adv[0][0] == adv[0][1]
    assert adv[1][0] < 0 and adv[1][0] == adv[1][1]
    assert approx(diag["credit/outcome/coverage"], 1.0)


def test_outcome_grpo_drops_zero_variance():
    group = [episode(True, [msg(0, "proposer")]),
             episode(True, [msg(0, "proposer")])]
    adv, _ = OutcomeGRPOAssigner().compute(group)
    assert adv == [{}, {}]


def test_role_reward_groups_roles_separately():
    p_good = "推理过程：ok\n最终答案：1"
    p_bad = "推理过程：bad\n最终答案：2"
    c_ok = "错误分析：无错误"
    c_flag = "错误分析：答案错误"
    group = [episode(True, [msg(0, "proposer", p_good), msg(1, "critic", c_ok)]),
             episode(False, [msg(0, "proposer", p_bad), msg(1, "critic", c_flag)])]
    adv, diag = RoleRewardAssigner().compute(group)
    assert adv[0][0] > 0 and adv[1][0] < 0
    # Both critics classified their reviewed answer correctly, so their group
    # has zero variance and receives no fabricated gradient.
    assert 1 not in adv[0] and 1 not in adv[1]
    assert diag["credit/role/live_groups"] == 1


def test_at_grpo_uses_role_local_turn_index():
    group = [episode(True, [msg(0, "proposer"), msg(1, "critic"),
                            msg(2, "proposer")]),
             episode(False, [msg(0, "proposer"), msg(1, "critic"),
                             msg(2, "proposer")])]
    adv, diag = ATGRPOAssigner().compute(group)
    assert set(adv[0]) == {0, 1, 2}
    assert adv[0][0] > 0 and adv[1][0] < 0
    assert diag["credit/at_grpo/live_groups"] == 3


def test_gigpo_macro_plus_micro_and_coverage():
    meta = {0: {"round": 0, "sigma": "solve", "is_response": False}}
    group = [episode(True, [msg(0, "proposer",
                               "推理过程：ok\n最终答案：1", prompt="same")], meta),
             episode(False, [msg(0, "proposer",
                                "推理过程：bad\n最终答案：2", prompt="same")], meta)]
    exact = GiGPOAssigner({"anchor_mode": "exact"})
    adv, diag = exact.compute(group)
    # macro and micro have the same sign and both are active.
    assert adv[0][0] > 1.0 and adv[1][0] < -1.0
    assert approx(diag["credit/gigpo/micro_coverage"], 1.0)
    assert approx(diag["credit/gigpo/exact_anchor_coverage"], 1.0)


def test_gigpo_does_not_invent_micro_credit_for_unique_exact_states():
    group = [episode(True, [msg(0, "proposer", prompt="a")]),
             episode(False, [msg(0, "proposer", prompt="b")])]
    adv, diag = GiGPOAssigner({"anchor_mode": "exact"}).compute(group)
    assert approx(abs(adv[0][0]), 1.0) and approx(abs(adv[1][0]), 1.0)
    assert diag["credit/gigpo/micro_coverage"] == 0.0


def test_majority_answer_has_stable_tie_break():
    assert majority_answer(["2", "1", "1", "2"]) == "2"
    assert majority_answer(["", "3", ""]) == "3"
    assert majority_answer(["080", "80", "079"]) in ("080", "80")


def test_baseline_configs_exist_and_do_not_include_raca():
    names = {p.stem for p in (ROOT / "baselines" / "configs").glob("*.yaml")}
    expected = {"outcome_grpo", "role_reward", "at_grpo", "gigpo",
                "single_agent_grpo", "fixed_four_role_grpo", "sft_cot", "self_consistency",
                "self_refine", "fixed_four_role"}
    assert expected <= names
    assert "raca" not in names


def test_raca_source_files_are_not_under_baselines():
    for path in (ROOT / "baselines").rglob("*"):
        assert "training/raca_adv.py" not in str(path)
        assert "agents/raca_rewards.py" not in str(path)


def run():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("  ✓", test.__name__)
    print("All %d baseline tests passed." % len(tests))


class _FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "SYS=" + messages[0]["content"] + "\nUSR=" + messages[1]["content"]

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class _Result:
    def __init__(self, text):
        self.text = text
        self.log_probs = [-0.1] * max(len(text.split()), 1)
        self.token_ids = list(range(1, len(self.log_probs) + 1))

    def __iter__(self):
        yield self.text
        yield self.log_probs
        yield self.token_ids


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def generate_batch(self, requests):
        self.calls.extend(requests)
        out = []
        for request_ in requests:
            role = request_["role"]
            prompt = request_["prompt"]
            if role == "critic":
                text = "错误分析：无错误"
            elif role == "verifier":
                text = "分数: 0.9\n验证说明：通过"
            elif "你是同一个数学解题模型的自我审查阶段" in prompt:
                text = "没有发现关键错误。"
            else:
                text = "推理过程：测试\n最终答案：1"
            out.append(_Result(text))
        return out


def test_single_agent_executor_contract():
    from baselines.systems.runtime import SingleAgentExecutor
    engine = _FakeEngine()
    ex = SingleAgentExecutor(None, _FakeTokenizer(), {}, engine, eval_mode=False)
    rows = ex.run_episodes_batch(["q"], ["1"])
    assert len(rows) == 1 and rows[0]["is_correct"]
    assert len(rows[0]["turn_ids"]) == len(rows[0]["log_probs"])
    assert rows[0]["baseline_gold"] == "1"
    assert set(rows[0]["raca_turn_data"]) == {0}


def test_system_baseline_call_budgets():
    from baselines.systems.runtime import SystemEvaluator
    expected = {"sft_cot": 1, "self_consistency": 4,
                "self_refine": 3, "fixed_four_role": 5}
    for method, calls in expected.items():
        engine = _FakeEngine()
        evaluator = SystemEvaluator(
            _FakeTokenizer(), engine, method, n_samples=4, temperature=0.7)
        rows = evaluator.run_batch(["q"], ["1"])
        assert rows[0]["n_calls"] == calls, (method, rows[0]["n_calls"])
        assert len(engine.calls) == calls
        assert rows[0]["is_correct"]


def test_fixed_pipeline_training_executor_contract():
    from baselines.systems.runtime import FixedPipelineExecutor
    engine = _FakeEngine()
    ex = FixedPipelineExecutor(None, _FakeTokenizer(), {}, engine, eval_mode=False)
    rows = ex.run_episodes_batch(["q"], ["1"])
    ep = rows[0]
    assert ep["is_correct"] and len(ep["messages"]) == 5
    assert [m["role_name"] for m in ep["messages"]] == [
        "proposer", "controller", "critic", "proposer", "verifier"]
    assert len(ep["turn_ids"]) == len(ep["log_probs"])
    assert set(ep["raca_turn_data"]) == {0, 1, 2, 3, 4}


def test_at_grpo_uses_paper_reward_equation():
    # alpha * team + local with alpha=1: episode 0 has team=0/local=1;
    # episode 1 has team=1/local=0, so both rewards tie and yield no advantage.
    good_local_bad_team = episode(False, [msg(0, "proposer",
        "推理过程：ok\n最终答案：1")], gold="1")
    bad_local_good_team = episode(True, [msg(0, "proposer",
        "推理过程：bad\n最终答案：2")], gold="1")
    adv, diag = ATGRPOAssigner({"team_weight": 1.0}).compute(
        [good_local_bad_team, bad_local_good_team])
    assert adv == [{}, {}]
    assert diag["credit/at_grpo/team_weight"] == 1.0


def test_gigpo_discounted_terminal_return():
    ep = episode(True, [msg(0, "proposer", prompt="s0"),
                        msg(1, "critic", prompt="s1"),
                        msg(2, "verifier", prompt="s2")])
    assigner = GiGPOAssigner({"gamma": 0.5, "anchor_mode": "exact"})
    returns = assigner._returns(ep, 1.0)
    assert approx(returns[2], 1.0)
    assert approx(returns[1], 0.5)
    assert approx(returns[0], 0.25)


def test_production_raca_files_have_no_worktree_diff():
    import subprocess
    if os.environ.get("BASELINE_ALLOW_RACA_CHANGES") == "1":
        print("  [skip] RACA worktree diff check: intentional v34 development")
        return
    # Local development runs inside a Git worktree, but Primus deploys a ZIP
    # archive without .git metadata.  The isolation assertion is meaningful
    # only in a worktree; all algorithm/data contract tests still run on Primus.
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(ROOT), text=True, capture_output=True)
    except FileNotFoundError:
        probe = None
    if (probe is None or probe.returncode != 0
            or probe.stdout.strip() != "true"):
        print("  [skip] RACA worktree diff check: Git metadata unavailable")
        return

    protected = [
        "train.py", "training/grpo_trainer.py", "training/raca_adv.py",
        "agents/raca_rewards.py", "agents/agentic_executor.py",
        "configs/agentic/default.yaml", "submit_primus.sh",
    ]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--"] + protected,
        cwd=str(ROOT), text=True).strip()
    assert not changed, "RACA files changed: " + changed


def test_unified_launcher_dispatch_and_guards():
    script = ROOT / "baselines" / "submit_primus.sh"
    listed = subprocess.check_output(
        ["bash", str(script), "--list"], cwd=str(ROOT), text=True).splitlines()
    expected = {
        "outcome_grpo", "role_reward", "at_grpo", "gigpo",
        "single_agent_grpo", "fixed_four_role_grpo", "sft_cot",
        "self_consistency", "self_refine", "fixed_four_role",
    }
    assert set(listed) == expected

    train = subprocess.check_output(
        ["bash", str(script), "--dry-run", "at_grpo"],
        cwd=str(ROOT), text=True)
    assert "BASELINE_NAME=at_grpo" in train
    assert "submit_primus_baseline.sh" in train

    evaluation = subprocess.check_output(
        ["bash", str(script), "--dry-run", "self_consistency"],
        cwd=str(ROOT), text=True)
    assert "METHOD=self_consistency" in evaluation
    assert "submit_primus_system_eval.sh" in evaluation

    for forbidden in ("raca", "all", "not_a_method"):
        result = subprocess.run(
            ["bash", str(script), "--dry-run", forbidden],
            cwd=str(ROOT), text=True, capture_output=True)
        assert result.returncode == 2, (forbidden, result.returncode)


def test_baseline_hydra_entry_uses_absolute_config_directory():
    source = (ROOT / "baselines" / "hydra_runner.py").read_text(encoding="utf-8")
    assert 'initialize_config_dir(config_dir=CONFIG_DIR' in source
    assert 'compose(config_name="config", overrides=overrides)' in source
    assert 'production_train.main, "__wrapped__"' in source
    assert '@hydra.main' not in source
    for name in ("train_credit.py", "train_single_agent_grpo.py",
                 "train_fixed_four_role_grpo.py"):
        wrapper = (ROOT / "baselines" / name).read_text(encoding="utf-8")
        assert "run_production_train" in wrapper
        assert "from train import main as hydra_main" not in wrapper


if __name__ == "__main__":
    run()
