"""CPU tests for measurement denominators and observational invariance."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from training.raca_adv import compute_raca_advantages
from training.raca_ablations import make_ablated_advantage_fn
from training.rebuttal_diagnostics import (
    DiagnosticWriter, behavior_report, credit_report, episode_diagnostics,
)


def proposer(reward, context="explore", round=0, forced=False):
    return dict(role="proposer", sigma=context, round=round, is_response=False,
                reward=reward, r_prop=reward, r_int=None if forced else -0.02,
                r_int_w=None if forced else -0.02, layer_key=int(reward),
                lambda_int=1.0, token_credit=True)


def episode(turns):
    return dict(raca_turn_data=turns, is_correct=False,
        messages=[dict(turn_id=tid, role_name=t["role"], response_ids=[1, 2],
            credit_spans={"solution": (0, 1), "interaction": (1, 2)})
            for tid, t in turns.items()],
        turn_ids=[tid for tid in turns for _ in range(2)])


class Measurements(unittest.TestCase):
    def test_auditing_does_not_change_advantages_or_inputs(self):
        turns = [{0: proposer(0), 1: proposer(0)}, {0: proposer(1)}]
        before = json.dumps(turns, sort_keys=True)
        for mode in (None, "no_role", "no_context", "no_channel"):
            fn = make_ablated_advantage_fn(mode)
            expected = fn(turns)
            rows = []
            self.assertEqual(expected, fn(turns, diagnostics=rows))
            self.assertTrue(rows)
        self.assertEqual(before, json.dumps(turns, sort_keys=True))

    def test_deduplication_preserves_turn_coverage_and_absence_is_distinct(self):
        groups = [[episode({0: proposer(0), 1: proposer(0)}),
                   episode({0: proposer(1)})]]
        anchors = []
        adv = compute_raca_advantages([e["raca_turn_data"] for e in groups[0]], diagnostics=anchors)
        prop = next(r for r in anchors if r["context"] == "explore" and r["channel"] == "prop")
        self.assertEqual((prop["entries"], prop["representatives"]), (3, 2))
        self.assertEqual(prop["status"], "valid")
        statuses = {r["status"] for r in anchors}
        self.assertTrue({"valid", "absent", "too_small"} <= statuses)
        report = credit_report(groups, [adv], anchors, vocab_size=10)
        self.assertEqual((report["worker_assigned"], report["worker_total"]), (3, 3))
        self.assertEqual(report["worker_structural_valid"], 3)
        # Missing interaction credit must not erase independently valid solution credit.
        self.assertTrue(all(set(a[0]) == {"solution"} for a in adv))

    def test_zero_advantage_is_valid_and_alignment_loss_is_separate(self):
        turns = [{0: dict(role="critic", sigma="refine", round=0,
                  is_response=True, reward=r)} for r in (0.0, 1.0, 2.0)]
        eps = [episode(t) for t in turns]
        eps[0]["messages"][0]["logprob_aligned"] = False
        rows = []
        adv = compute_raca_advantages(turns, diagnostics=rows)
        self.assertEqual(adv[1][0], 0.0)
        report = credit_report([eps], [adv], rows, 10)
        cell = report["turns"][0]
        self.assertEqual((cell["assigned"], cell["nonzero_assigned"], cell["structural_valid"]), (3, 2, 2))

    def test_forced_turns_do_not_enter_interaction_denominator(self):
        turns = [{0: proposer(0, forced=True)}, {0: proposer(1)}]
        rows = []
        compute_raca_advantages(turns, diagnostics=rows)
        self.assertEqual(sum(r["entries"] for r in rows if r["channel"] == "int"), 1)

    def test_one_missing_span_only_removes_that_channel_from_routed_coverage(self):
        turns = [{0: proposer(0)}, {0: proposer(0)},
                 {0: proposer(1)}, {0: proposer(1)}]
        for td, reward in zip(turns, (-0.1, -0.02, -0.07, 0.0)):
            td[0]["r_int"] = reward
        eps = [episode(td) for td in turns]
        eps[0]["messages"][0]["credit_spans"] = {"solution": (0, 2)}
        rows = []
        adv = compute_raca_advantages(turns, diagnostics=rows)
        report = credit_report([eps], [adv], rows, 10)
        channel = next(r for r in report["channels"] if r["context"] == "explore"
                       and r["channel"] == "int" and r["p"] == 0)
        self.assertEqual((channel["assigned"], channel["structural_assigned"], channel["eligible"]), (2, 1, 2))
        self.assertEqual(report["worker_structural_valid"], 4)

    def test_unknown_ablation_rejected_without_pytest_dependency(self):
        from training.raca_ablations import compute_raca_advantages_ablated
        with self.assertRaises(ValueError):
            make_ablated_advantage_fn("unknown")
        with self.assertRaises(ValueError):
            compute_raca_advantages_ablated([], ablation="unknown")

    def test_constant_rewards_are_low_variance_not_small(self):
        rows = []
        turns = [{0: proposer(1)}, {0: proposer(1)}]
        adv = compute_raca_advantages(turns, diagnostics=rows)
        self.assertEqual(adv, [{}, {}])
        prop = next(r for r in rows if r["context"] == "explore" and r["channel"] == "prop")
        self.assertEqual(prop["status"], "low_variance")

    def test_verifier_is_scored_against_reviewed_answer_not_final_outcome(self):
        rnd = dict(sigma="explore", primary_answer="3", corrected_answer="4",
            primary_tid=1, u=True, forced=False, target="critic",
            critic_turns=[dict(tid=2, hop=1, flagged=True, reviewed_answer="3", correction_followed=True)],
            verifier_turns=[dict(tid=4, score=0.9, reviewed_answer="3")],
            correction_turns=[dict(tid=3, answer="4")])
        td = {1: {"r_int": 0.28}, 2: {"reward": 0.3}, 4: {"reward": 0.1}}
        diag = episode_diagnostics([rnd], "4", td, [], [("", "3"), ("", "4")], [])
        ep = dict(is_correct=True, initial_is_correct=False,
            last_candidate_is_correct=True, last_primary_is_correct=False,
            is_correct_uni_incl=True, rebuttal=diag)
        report = behavior_report([ep])
        self.assertAlmostEqual(report["verifier"]["brier"], 0.81)
        self.assertEqual(report["critics"][0]["tp"], 1)
        self.assertEqual(report["corrections"]["fix"], 1)
        self.assertEqual(report["predictions"]["final_fix_rate"], 1.0)
        self.assertIsNone(report["predictions"]["final_break_rate"])

    def test_stop_proposals_and_execution_are_counted_separately(self):
        events = [dict(round=0, context="explore", score_available=False,
            latest_score=None, decision="stop", accepted=False, blocked=True),
            dict(round=1, context="verify", score_available=True,
            latest_score=0.2, decision="continue", accepted=False, blocked=False),
            dict(round=2, context="verify", score_available=True,
            latest_score=0.9, decision="stop", accepted=True, blocked=False)]
        report = behavior_report([dict(is_correct=True, rebuttal={"controller": events})])
        rows = report["controller"]
        self.assertEqual(sum(x["proposed_stop"] for x in rows), 2)
        self.assertEqual(sum(x["accepted_stop"] for x in rows), 1)
        self.assertEqual(sum(x["blocked_stop"] for x in rows), 1)
        self.assertEqual(sum(x["continue_with_score"] for x in rows), 1)

    def test_persistent_log_includes_skips_and_uses_unique_directories(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = {"agentic": {"rebuttal_print_every": 5}}
            a, b = DiagnosticWriter(directory, config), DiagnosticWriter(directory, config)
            self.assertNotEqual(a.directory, b.directory)
            a.write("train", 0, [], skipped=True)
            row = json.loads((a.directory / "diagnostics.jsonl").read_text())
            self.assertTrue(row["skipped"])
            self.assertEqual(row["behavior"]["diagnostic_episodes"], 0)
            manifest = json.loads((a.directory / "manifest.json").read_text())
            self.assertIn("agents/raca_rewards.py", manifest["source_hashes"])

    def test_csv_pools_counts_not_batch_percentages(self):
        from summarize_rebuttal import summarize
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            writer = DiagnosticWriter(directory, {"agentic": {}})
            for rewards in ((0, 1), (0, 0, 0, 0)):
                eps = [episode({0: dict(role="critic", sigma="refine", round=0,
                    is_response=True, reward=r)}) for r in rewards]
                rows = []
                adv = compute_raca_advantages([e["raca_turn_data"] for e in eps], diagnostics=rows)
                credit = credit_report([eps], [adv], rows, 10)
                writer.write("train", 1, eps, credit=credit)
            output = Path(directory) / "csv"
            summarize([writer.directory / "diagnostics.jsonl"], output, "train")
            with (output / "turns.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]["coverage"]), 2 / 6)
            self.assertEqual(int(rows[0]["total"]), 6)

    def test_executor_instrumentation_preserves_actions_and_rewards(self):
        from test_raca_v2 import FakeEngine, _mk_executor
        script = dict(controller=["decision: continue"], proposer=[
            "推理过程：1+3=4\n最终答案：4\n<interaction>\naction: none\ntarget: none\nreason: x\n</interaction>"])
        results = []
        for enabled in (False, True):
            engine = FakeEngine({k: list(v) for k, v in script.items()})
            executor = _mk_executor(engine, max_rounds=1, stop_gate=False,
                                    rebuttal_diagnostics=enabled)
            results.append(executor.run_episodes_batch(["1+3=?"], ["4"], eps_force=0)[0])
        for key in ("final_answer", "is_correct", "raca_turn_data", "raca_round_meta", "turn_ids", "log_probs"):
            self.assertEqual(results[0][key], results[1][key])
        self.assertNotIn("rebuttal", results[0])
        self.assertEqual(results[1]["rebuttal"]["controller"][0]["decision"], "continue")

    def test_paired_test_uses_discordant_pairs_and_rejects_unmatched_sets(self):
        from compare_rebuttal_predictions import paired_stats
        a = {str(i): dict(gold="4", is_correct=True) for i in range(10)}
        b = {str(i): dict(gold="4", is_correct=False) for i in range(10)}
        result = paired_stats(a, b, bootstrap=100)
        self.assertAlmostEqual(result["exact_two_sided_mcnemar_p"], 2 / 1024)
        self.assertEqual(result["a_only_correct"], 10)
        self.assertEqual(paired_stats(a, a, 100)["exact_two_sided_mcnemar_p"], 1.0)
        with self.assertRaises(ValueError):
            paired_stats(a, {"0": b["0"]}, 100)


if __name__ == "__main__":
    unittest.main()
