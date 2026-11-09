"""Read-only rebuttal measurements. No sampling, rewards, or policy updates here.

Counts are additive; zero denominators produce None, never fabricated 0%.
Anchor coverage precedes token/logprob guards. Structural coverage is separate.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

SCHEMA_VERSION = 1


def ratio(n, d):
    return n / d if d else None


def _moments(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def anchor_record(key, entries, rewards, delta):
    mean, std = _moments(rewards)
    status = ("too_small" if len(rewards) < 2 else
              "valid" if std > delta else "low_variance")
    return dict(role=key[0], context=key[1], response=bool(key[2]),
                channel=key[3], p=key[4], entries=len(entries),
                representatives=len(rewards), mean=mean, std=std, status=status,
                reward_histogram=dict(Counter(str(x) for x in rewards)))


def add_absent_anchors(rows, observed):
    # Empty eligible anchors (unreached context or no eligible turns) are distinct
    # from reached-but-degenerate groups.
    channels = [("proposer", False, "prop", None),
                ("proposer", False, "int", 0), ("proposer", False, "int", 1),
                ("proposer", True, "rew", None),
                ("critic", True, "rew", None), ("verifier", True, "rew", None)]
    for context in ("explore", "refine", "verify"):
        for role, response, channel, p in channels:
            if (role, context, response, channel, p) not in observed:
                rows.append(dict(role=role, context=context, response=response,
                                 channel=channel, p=p, entries=0, representatives=0,
                                 mean=None, std=None, status="absent",
                                 reward_histogram={}))


def credit_report(groups, advantages, anchors, vocab_size=None):
    """Use exact estimator results; do not recompute or change advantages."""
    from training.raca_adv import token_credit_components
    cells = {}
    compact = []
    routed = Counter()
    anchor_status = {}
    for row in anchors:
        key = tuple(row[k] for k in ("role", "context", "response", "channel", "p"))
        anchor_status[(row.get("question_index", 0), key)] = row["status"]
    drop_role = any(row["role"] == "_all_" for row in anchors)
    drop_context = any(row["context"] == "_all_" for row in anchors)
    for qi, (episodes, specs) in enumerate(zip(groups, advantages)):
        for ei, (ep, adv) in enumerate(zip(episodes, specs)):
            messages = {m.get("turn_id"): m for m in ep.get("messages", [])}
            lp_counts = Counter(ep.get("turn_ids", []))
            for tid, turn in ep.get("raca_turn_data", {}).items():
                role = turn["role"]
                key = (role, turn.get("sigma", "explore"),
                       bool(turn.get("is_response", False)))
                cell = cells.setdefault(key, dict(role=key[0], context=key[1],
                    response=key[2], total=0, assigned=0, structural_valid=0,
                    nonzero_assigned=0, structural_failures=Counter()))
                cell["total"] += 1
                spec = adv.get(tid)
                assigned = tid in adv
                failure = None
                msg = messages.get(tid, {})
                ids = msg.get("response_ids", [])
                routed_names = set()
                if assigned:
                    cell["assigned"] += 1
                    vals = spec.values() if isinstance(spec, dict) else [spec]
                    cell["nonzero_assigned"] += int(any(x != 0 for x in vals))
                    if not msg:
                        failure = "missing_message"
                    elif not msg.get("logprob_aligned", True):
                        failure = "logprob_alignment"
                    elif not ids:
                        failure = "empty_response"
                    elif len(ids) != lp_counts.get(tid, 0):
                        failure = "logprob_length"
                    elif vocab_size is not None and any(
                            x < 0 or x >= vocab_size for x in ids):
                        failure = "token_out_of_vocab"
                    elif isinstance(spec, dict) and not token_credit_components(
                            spec, msg.get("credit_spans", msg.get("interaction_span")),
                            range(len(ids))):
                        failure = "no_routable_span"
                    if failure:
                        cell["structural_failures"][failure] += 1
                    else:
                        cell["structural_valid"] += 1
                        if isinstance(spec, dict):
                            routed_names = {part[0] for part in token_credit_components(
                                spec, msg.get("credit_spans", msg.get("interaction_span")),
                                range(len(ids)))}
                        primary = role == "proposer" and not key[2] and "r_prop" in turn
                        channels = ([("prop", None, "solution"),
                                     ("int", turn.get("layer_key"), "interaction")]
                                    if primary else [("rew", turn.get("layer_key"), "default")])
                        for channel, p, name in channels:
                            if channel == "int":
                                eligible = (turn.get("r_int") is not None
                                            and float(turn.get("lambda_int", 1.0)) != 0.0
                                            if turn.get("token_credit", False)
                                            else turn.get("r_int_w") is not None)
                                if not eligible:
                                    continue
                            anchor_key = ("_all_" if drop_role else role,
                                "_all_" if drop_context else key[1], key[2], channel, p)
                            if (anchor_status.get((qi, anchor_key)) == "valid"
                                    and (not isinstance(spec, dict) or name in routed_names)):
                                routed[anchor_key] += 1
                compact.append(dict(question_index=qi, episode_index=ei, tid=tid,
                    **turn, assigned=assigned, advantage=spec,
                    structural_failure=failure, routable_channels=sorted(routed_names),
                    response_tokens=len(ids), logprob_count=lp_counts.get(tid, 0),
                    logprob_aligned=msg.get("logprob_aligned", True),
                    credit_spans=msg.get("credit_spans", msg.get("interaction_span"))))
    turns = list(cells.values())
    for cell in turns:
        cell["coverage"] = ratio(cell["assigned"], cell["total"])
        cell["structural_coverage"] = ratio(cell["structural_valid"], cell["total"])
    acells = {}
    for row in anchors:
        key = tuple(row[k] for k in ("role", "context", "response", "channel", "p"))
        cell = acells.setdefault(key, dict(zip(
            ("role", "context", "response", "channel", "p"), key)))
        if "groups" not in cell:
            cell.update(groups=0, valid_groups=0, absent=0, too_small=0,
                        low_variance=0, eligible=0, assigned=0,
                        representatives=0, n2=0, valid_n2=0,
                        representative_size_histogram=Counter())
        cell["groups"] += 1
        cell[row["status"] if row["status"] != "valid" else "valid_groups"] += 1
        cell["eligible"] += row["entries"]
        cell["assigned"] += row["entries"] if row["status"] == "valid" else 0
        cell["representatives"] += row["representatives"]
        cell["n2"] += int(row["representatives"] == 2)
        cell["valid_n2"] += int(row["representatives"] == 2 and row["status"] == "valid")
        cell["representative_size_histogram"][str(row["representatives"])] += 1
    for cell in acells.values():
        cell["coverage"] = ratio(cell["assigned"], cell["eligible"])
        key = tuple(cell[k] for k in ("role", "context", "response", "channel", "p"))
        cell["structural_assigned"] = routed[key]
        cell["structural_coverage"] = ratio(routed[key], cell["eligible"])
    workers = [row for row in turns if row["role"] != "controller"]
    return dict(worker_total=sum(x["total"] for x in workers),
                worker_assigned=sum(x["assigned"] for x in workers),
                worker_structural_valid=sum(x["structural_valid"] for x in workers),
                turns=turns, channels=list(acells.values()), anchors=anchors,
                turn_records=compact)


def episode_diagnostics(records, gold, turn_data, controller, candidates, messages):
    from agents.grader import math_equal
    def correct(answer):
        return bool(answer) and bool(math_equal(answer, gold))
    rounds, critics, verifiers, corrections = [], [], [], []
    for t, rnd in enumerate(records):
        before = correct(rnd["primary_answer"])
        end = correct(rnd["corrected_answer"]) if rnd["corrected_answer"] is not None else before
        rounds.append(dict(round=t, context=rnd["sigma"], p=int(before),
            end_correct=end, request=bool(rnd["u"]), forced=bool(rnd["forced"]),
            target=rnd.get("target"),
            interaction_reward=turn_data.get(rnd["primary_tid"], {}).get("r_int")))
        for ct in rnd["critic_turns"]:
            reviewed = correct(ct["reviewed_answer"])
            available = bool(ct["correction_followed"]) or t + 1 < len(records)
            outcome = (end if ct["correction_followed"] else
                       correct(records[t + 1]["primary_answer"]) if t + 1 < len(records)
                       else None)
            critics.append(dict(round=t, context=rnd["sigma"], tid=ct["tid"],
                hop=ct.get("hop"), flagged=bool(ct["flagged"]),
                reviewed_correct=reviewed, repair_available=available,
                repair_correct=outcome,
                terminal_flag=bool(ct["flagged"] and not reviewed and not available),
                reward=turn_data[ct["tid"]]["reward"]))
        for vt in rnd["verifier_turns"]:
            verifiers.append(dict(round=t, context=rnd["sigma"], tid=vt["tid"],
                score=vt["score"], reviewed_correct=correct(vt["reviewed_answer"]),
                source=vt.get("source", "chain"),
                reward=turn_data[vt["tid"]]["reward"]))
        previous = before
        for ct in rnd["correction_turns"]:
            now = correct(ct["answer"])
            corrections.append(dict(round=t, tid=ct["tid"], before=previous,
                after=now, verified=bool(ct.get("verified")),
                verifier_score=ct.get("verifier_score")))
            previous = now
    return dict(controller=controller, rounds=rounds, critics=critics,
                verifiers=verifiers, corrections=corrections,
                candidates=[dict(answer=a, correct=correct(a)) for _r, a in candidates],
                input_tokens=sum(m.get("input_token_count", 0) for m in messages),
                input_tokens_complete=all("input_token_count" in m for m in messages),
                output_tokens=sum(len(m.get("response_ids", [])) for m in messages))


def behavior_report(episodes):
    """All rates retain their numerator and denominator. Associations are not causality."""
    interactions, stops, critics = {}, {}, {}
    scores = []
    score_bins = [dict(bin=i, lower=i / 10, upper=(i + 1) / 10,
                       n=0, score_sum=0.0, correct=0) for i in range(10)]
    correct_scores, wrong_scores = [], []
    verifier = Counter()
    prediction = Counter()
    cost = Counter()
    correction = Counter()
    stopping = Counter()
    for ep in episodes:
        diag = ep.get("rebuttal", {})
        cost["episodes"] += 1
        stopping["episodes"] += 1
        stopping["mechanical_stop_episodes"] += int(ep.get("auto_stopped", False))
        stopping["exhausted_episodes"] += int(not ep.get("stopped", False))
        cost["calls"] += len(ep.get("messages", []))
        cost["input_tokens"] += diag.get("input_tokens", 0)
        cost["output_tokens"] += diag.get("output_tokens", ep.get("generated_tokens", 0))
        cost["input_complete_episodes"] += int(diag.get("input_tokens_complete", False))
        for role, count in ep.get("role_calls", {}).items():
            cost[f"calls_{role}"] += count
        keys = {"final": "is_correct", "initial": "initial_is_correct",
                "last_candidate": "last_candidate_is_correct",
                "last_primary": "last_primary_is_correct",
                "uniform_all": "is_correct_uni_incl"}
        for label, key in keys.items():
            if key not in ep:
                continue
            prediction[f"{label}_n"] += 1
            prediction[f"{label}_correct"] += int(ep[key])
            if "initial_is_correct" in ep and label != "initial":
                start = bool(ep["initial_is_correct"])
                prediction[f"{label}_initial_wrong"] += int(not start)
                prediction[f"{label}_initial_correct"] += int(start)
                prediction[f"{label}_fix"] += int(not start and ep[key])
                prediction[f"{label}_break"] += int(start and not ep[key])
        for rnd in diag.get("rounds", []):
            # Forced calls are reported separately, never credited as chosen requests.
            target = rnd.get("target") if rnd["request"] or rnd["forced"] else "none"
            key = (rnd["context"], rnd["p"], target, rnd["forced"])
            row = interactions.setdefault(key, dict(context=key[0], p=key[1],
                target=key[2], forced=key[3], n=0, requested=0,
                end_correct=0, fix=0, harm=0, failed_request=0))
            row["n"] += 1
            row["requested"] += int(rnd["request"])
            row["end_correct"] += int(rnd["end_correct"])
            row["fix"] += int(not rnd["p"] and rnd["end_correct"])
            row["harm"] += int(rnd["p"] and not rnd["end_correct"])
            row["failed_request"] += int(rnd["request"] and not rnd["p"] and not rnd["end_correct"])
        for event in diag.get("controller", []):
            stopping["decisions"] += 1
            stopping["proposed_stop"] += int(event["decision"] == "stop")
            stopping["accepted_stop"] += int(event["accepted"])
            stopping["blocked_stop"] += int(event["blocked"])
            stopping["with_score"] += int(event["score_available"])
            stopping["continue_with_score"] += int(event["score_available"] and event["decision"] != "stop")
            score = event.get("latest_score")
            bucket = "missing" if score is None else str(min(9, int(score * 10)))
            displayed = event.get("displayed_best_score")
            displayed_bin = "missing" if displayed is None else str(min(9, int(displayed * 10)))
            key = (event["round"], event["context"], event["score_available"], bucket, displayed_bin)
            row = stops.setdefault(key, dict(round=key[0], context=key[1],
                score_available=key[2], score_bin=key[3], displayed_score_bin=key[4], n=0,
                proposed_stop=0, accepted_stop=0, blocked_stop=0,
                continue_with_score=0, accepted_stop_correct=0))
            row["n"] += 1
            row["proposed_stop"] += int(event["decision"] == "stop")
            row["accepted_stop"] += int(event["accepted"])
            row["blocked_stop"] += int(event["blocked"])
            row["continue_with_score"] += int(event["score_available"] and event["decision"] != "stop")
            row["accepted_stop_correct"] += int(event["accepted"] and ep["is_correct"])
        for event in diag.get("critics", []):
            key = (event["round"], event["context"], event.get("hop"))
            row = critics.setdefault(key, dict(round=key[0], context=key[1], hop=key[2],
                n=0, tp=0, fp=0, tn=0, fn=0, terminal_flags=0,
                repair_success=0, repair_failure=0, reward_sum=0.0))
            row["n"] += 1
            row[("fp" if event["reviewed_correct"] else "tp") if event["flagged"]
                else ("tn" if event["reviewed_correct"] else "fn")] += 1
            row["terminal_flags"] += int(event["terminal_flag"])
            if event["flagged"] and not event["reviewed_correct"] and event["repair_available"]:
                row["repair_success" if event["repair_correct"] else "repair_failure"] += 1
            row["reward_sum"] += event["reward"]
        for event in diag.get("verifiers", []):
            verifier["total"] += 1
            score = event.get("score")
            if score is None or not math.isfinite(score):
                verifier["unparsed"] += 1
                continue
            truth = int(event["reviewed_correct"])
            scores.append(score)
            (correct_scores if truth else wrong_scores).append(score)
            verifier["parsed"] += 1
            verifier["exact_half"] += int(score == 0.5)
            verifier["near_half"] += int(abs(score - 0.5) <= 0.05 + 1e-12)
            verifier["brier_sum"] += (score - truth) ** 2
            verifier["reward_sum"] += event["reward"]
            verifier["decision_correct"] += int((score > 0.5) == bool(truth))
            verifier[f"source_{event['source']}"] += 1
            bin_row = score_bins[min(9, max(0, int(score * 10)))]
            bin_row["n"] += 1
            bin_row["score_sum"] += score
            bin_row["correct"] += truth
        for event in diag.get("corrections", []):
            correction["total"] += 1
            correction["before_correct"] += int(event["before"])
            correction["before_wrong"] += int(not event["before"])
            correction["fix"] += int(not event["before"] and event["after"])
            correction["break"] += int(event["before"] and not event["after"])
    mean, std = _moments(scores)
    verifier = dict(verifier, parsed=verifier["parsed"],
                    brier_sum=verifier["brier_sum"], near_half=verifier["near_half"])
    verifier.update(score_mean=mean, score_std=std,
        correct_n=len(correct_scores), wrong_n=len(wrong_scores),
        correct_score_mean=_moments(correct_scores)[0], wrong_score_mean=_moments(wrong_scores)[0],
        brier=ratio(verifier["brier_sum"], verifier["parsed"]),
        near_half_rate=ratio(verifier["near_half"], verifier["parsed"]))
    verifier["calibration_bins"] = score_bins
    verifier["ece"] = (sum(abs(x["score_sum"] - x["correct"]) for x in score_bins)
                       / len(scores) if scores else None)
    for label in ("final", "last_candidate", "last_primary", "uniform_all"):
        prediction[f"{label}_fix_rate"] = ratio(prediction[f"{label}_fix"], prediction[f"{label}_initial_wrong"])
        prediction[f"{label}_break_rate"] = ratio(prediction[f"{label}_break"], prediction[f"{label}_initial_correct"])
    return dict(n_episodes=len(episodes), diagnostic_episodes=sum("rebuttal" in e for e in episodes),
                interactions=list(interactions.values()), controller=list(stops.values()),
                controller_summary=dict(stopping),
                critics=list(critics.values()), verifier=dict(verifier),
                predictions=dict(prediction), corrections=dict(correction), cost=dict(cost))


def prediction_fields(ep):
    keys = ("initial_answer", "initial_is_correct", "last_candidate_answer",
            "last_candidate_is_correct", "last_primary_answer", "last_primary_is_correct",
            "is_correct_uni_excl", "is_correct_wt_excl", "is_correct_uni_incl", "is_correct_wt_incl")
    result = {k: ep[k] for k in keys if k in ep}
    if "rebuttal" in ep:
        result["rebuttal"] = ep["rebuttal"]
    return result


def _coverage(n, d):
    return f"{n}/{d} ({n/d:.1%})" if d else "0/0 (NA)"


def print_report(report, label):
    credit = report.get("credit")
    if credit:
        print(f"[rebuttal:{label}:worker] assigned=" + _coverage(
            credit["worker_assigned"], credit["worker_total"]) + " structural=" + _coverage(
            credit["worker_structural_valid"], credit["worker_total"]), flush=True)
        for row in credit["channels"]:
            print(f"[rebuttal:{label}:anchor] {row['role']}/{row['context']}/"
                  f"{'response' if row['response'] else 'primary'}/{row['channel']}/p={row['p']} "
                  f"credit={_coverage(row['assigned'], row['eligible'])} "
                  f"routable={_coverage(row['structural_assigned'], row['eligible'])} "
                  f"groups={row['valid_groups']}/{row['groups']} absent={row['absent']} "
                  f"small={row['too_small']} lowvar={row['low_variance']} "
                  f"n2={row['n2']} sizes={dict(row['representative_size_histogram'])}", flush=True)
        for row in credit["turns"]:
            print(f"[rebuttal:{label}:turn] {row['role']}/{row['context']}/"
                  f"response={int(row['response'])} assigned={_coverage(row['assigned'], row['total'])} "
                  f"structural={_coverage(row['structural_valid'], row['total'])} "
                  f"failures={dict(row['structural_failures'])}", flush=True)
    behavior = report["behavior"]
    for name in ("controller", "interactions", "critics"):
        for row in behavior[name]:
            print(f"[rebuttal:{label}:{name}] " + json.dumps(row, ensure_ascii=False), flush=True)
    for name in ("controller_summary", "verifier", "predictions", "corrections", "cost"):
        print(f"[rebuttal:{label}:{name}] " + json.dumps(behavior[name], ensure_ascii=False), flush=True)


class DiagnosticWriter:
    """Append-only JSONL on the same persistent mount as checkpoints."""
    def __init__(self, base_dir, config, run_name="run"):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        self.directory = Path(base_dir) / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.print_every = max(1, int(config.get("agentic", {}).get("rebuttal_print_every", 1)))
        self.save_turns = bool(config.get("agentic", {}).get("rebuttal_save_turns", True))
        self.attempt = 0
        self.train_attempt = 0
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            revision = None
        source_hashes = {}
        source_root = Path(__file__).resolve().parents[1]
        for name in ("agents/agentic_executor.py", "agents/raca_rewards.py",
                     "training/raca_adv.py", "training/raca_ablations.py",
                     "training/grpo_trainer.py", "training/rebuttal_diagnostics.py", "train.py"):
            source = source_root / name
            source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = dict(schema_version=SCHEMA_VERSION, run_id=self.run_id,
            run_name=run_name, config=config, git_revision=revision, source_hashes=source_hashes,
            ablation=os.environ.get("RACA_ABLATION"),
            verifier_training_reward="1-abs(score-reviewed_correct)",
            verifier_diagnostic_brier="(score-reviewed_correct)^2; measurement only",
            coverage_definition="assigned/eligible before token guards; structural separately",
            notes="Observed repair and same-trajectory aggregation are not causal ablations.")
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[rebuttal] persistent diagnostics: {self.directory}", flush=True)

    def write(self, phase, step, episodes, credit=None, **metadata):
        self.attempt += 1
        if phase == "train":
            self.train_attempt += 1
        report = dict(schema_version=SCHEMA_VERSION, run_id=self.run_id,
            phase=phase, step=step, attempt=self.attempt, **metadata,
            behavior=behavior_report(episodes))
        if credit is not None:
            report["credit"] = {k: v for k, v in credit.items()
                                if self.save_turns or k != "turn_records"}
        if self.save_turns:
            report["episodes"] = [dict(question_id=hashlib.sha256(
                str(ep.get("question", "")).encode()).hexdigest(),
                is_correct=bool(ep["is_correct"]), **prediction_fields(ep)) for ep in episodes]
        line = json.dumps(report, ensure_ascii=False, allow_nan=False)
        with (self.directory / "diagnostics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        if phase != "train" or self.train_attempt % self.print_every == 0:
            print_report(report, f"{phase}:{step}:attempt{self.attempt}")
        return report
