"""Export persistent diagnostics as pooled-count CSV tables; retain run/phase."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from training.rebuttal_diagnostics import ratio


def summarize(paths, output, phase=None):
    tables = {k: {} for k in ("anchors", "turns", "controller", "interactions", "critics",
                              "predictions", "cost", "verifier", "controller_summary")}
    dims = dict(anchors=("role", "context", "response", "channel", "p"),
                turns=("role", "context", "response"),
                controller=("round", "context", "score_available", "score_bin", "displayed_score_bin"),
                interactions=("context", "p", "target", "forced"),
                critics=("round", "context", "hop"))
    seen = set()
    records = 0
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                report = json.loads(line)
                if phase and report["phase"] != phase:
                    continue
                unique = (report["run_id"], report["attempt"])
                if unique in seen:
                    raise ValueError(f"Duplicate diagnostic record: {unique}")
                seen.add(unique)
                records += 1
                base = dict(run_id=report["run_id"], phase=report["phase"])
                values = report["behavior"]
                credit = report.get("credit", {})
                for table, fields in dims.items():
                    rows = credit.get("channels" if table == "anchors" else "turns", []) if table in ("anchors", "turns") else values[table]
                    for row in rows:
                        key = (base["run_id"], base["phase"], *(row.get(k) for k in fields))
                        pooled = tables[table].setdefault(key, dict(base, **{k: row.get(k) for k in fields}))
                        for name, value in row.items():
                            if name in fields or name.endswith("coverage"):
                                continue
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                pooled[name] = pooled.get(name, 0) + value
                            elif isinstance(value, dict):
                                counts = pooled.setdefault(name, Counter())
                                counts.update(value)
                key = (base["run_id"], base["phase"])
                for table in ("predictions", "cost", "controller_summary"):
                    pooled = tables[table].setdefault(key, dict(base))
                    for name, value in values[table].items():
                        if name.endswith("_rate") or value is None:
                            continue
                        pooled[name] = pooled.get(name, 0) + value
                verifier = values["verifier"]
                pooled = tables["verifier"].setdefault(key, dict(base))
                n = verifier["parsed"]
                for name in ("total", "unparsed", "parsed", "exact_half", "near_half", "brier_sum", "correct_n", "wrong_n", "decision_correct", "reward_sum"):
                    pooled[name] = pooled.get(name, 0) + verifier.get(name, 0)
                pooled["score_sum"] = pooled.get("score_sum", 0) + (verifier["score_mean"] or 0) * n
                pooled["score_square_sum"] = pooled.get("score_square_sum", 0) + ((verifier["score_std"] or 0) ** 2 + (verifier["score_mean"] or 0) ** 2) * n
                for condition in ("correct", "wrong"):
                    name = f"{condition}_score_sum"
                    pooled[name] = pooled.get(name, 0) + (verifier[f"{condition}_score_mean"] or 0) * verifier[f"{condition}_n"]
                bins = pooled.setdefault("_calibration_bins", [dict(n=0, score_sum=0.0, correct=0) for _ in range(10)])
                for destination, source in zip(bins, verifier["calibration_bins"]):
                    for name in ("n", "score_sum", "correct"):
                        destination[name] += source[name]
    if not records:
        raise ValueError("No records match the requested phase")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for table, cells in tables.items():
        rows = list(cells.values())
        for row in rows:
            if table in ("anchors", "turns"):
                row["coverage"] = ratio(row["assigned"], row["eligible" if table == "anchors" else "total"])
                if table == "turns":
                    row["structural_coverage"] = ratio(row["structural_valid"], row["total"])
                else:
                    row["structural_coverage"] = ratio(row["structural_assigned"], row["eligible"])
            elif table == "verifier":
                row["brier"] = ratio(row["brier_sum"], row["parsed"])
                row["near_half_rate"] = ratio(row["near_half"], row["parsed"])
                row["score_mean"] = ratio(row["score_sum"], row["parsed"])
                row["score_std"] = (math.sqrt(max(0, row["score_square_sum"] / row["parsed"] - row["score_mean"] ** 2)) if row["parsed"] else None)
                for condition in ("correct", "wrong"):
                    row[f"{condition}_score_mean"] = ratio(row[f"{condition}_score_sum"], row[f"{condition}_n"])
                bins = row.pop("_calibration_bins")
                row["ece"] = ratio(sum(abs(x["score_sum"] - x["correct"]) for x in bins), row["parsed"])
                row["calibration_bins"] = json.dumps(bins, ensure_ascii=False)
            elif table == "predictions":
                for label in ("initial", "final", "last_candidate", "last_primary", "uniform_all"):
                    row[f"{label}_accuracy"] = ratio(row.get(f"{label}_correct", 0), row.get(f"{label}_n", 0))
                    for change, condition in (("fix", "wrong"), ("break", "correct")):
                        if label != "initial":
                            row[f"{label}_{change}_rate"] = ratio(row.get(f"{label}_{change}", 0), row.get(f"{label}_initial_{condition}", 0))
            for name, value in list(row.items()):
                if isinstance(value, dict):
                    row[name] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        path = output / f"{table}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            if fields:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        print(f"{table}: {len(rows)} rows -> {path}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", help="e.g. train, checkpoint_probe, eval_aime")
    args = parser.parse_args()
    summarize(args.logs, args.output, args.phase)


if __name__ == "__main__":
    main()
