"""Measure C2 label coverage without making an optimizer update."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=707)
    return parser.parse_args()


def _last_round(metrics_path):
    last = None
    with metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record.get("kind") == "round":
                last = record
    if last is None:
        raise RuntimeError("The label probe did not record a completed round")
    return last


def main():
    args = _arguments()
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")

    repository_root = Path(__file__).resolve().parents[4]
    with tempfile.TemporaryDirectory(prefix="m2_4_teacher_probe_") as directory:
        root = Path(directory)
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "SEAT_D3QN_VARIANT": "m2_4",
                "SEAT_M2_4_BRANCH": "c2",
                "SEAT_M2_4_PROBE_ROOT": str(root),
                "SEAT_D3QN_SESSION_TRANSITION_LIMIT": "4999",
                "SEAT_D3QN_CHECKPOINT_AFTER_ROUNDS": str(args.rounds),
                "SEAT_M2_4_CURRICULUM_BLOCK": "label_probe",
            }
        )
        command = [
            sys.executable,
            "main.py",
            "play",
            "--agents",
            "seat_d3qn",
            "peaceful_agent",
            "coin_collector_agent",
            "rule_based_agent",
            "--train",
            "1",
            "--scenario",
            "classic",
            "--n-rounds",
            str(args.rounds),
            "--no-gui",
            "--seed",
            str(args.seed),
            "--save-stats",
            str(root / "probe.json"),
        ]
        subprocess.run(command, cwd=repository_root, env=environment, check=True)
        result = _last_round(root / "experiment" / "metrics" / "training-metrics.jsonl")

    queries = int(result["teacher_queries"])
    accepted = int(result["teacher_accepted"])
    if int(result["optimizer_step"]) != 0:
        raise RuntimeError("The label-only probe unexpectedly updated the network")
    if queries == 0:
        raise RuntimeError("The rule teacher was not queried")
    coverage = accepted / float(queries)
    print("Teacher queries: {}".format(queries))
    print("Accepted labels: {}".format(accepted))
    print("Label coverage: {:.2%}".format(coverage))
    print("Agreement rejections: {}".format(result["teacher_agreement"]))
    print("Low-confidence rejections: {}".format(result["teacher_low_confidence"]))
    print("Safety rejections: {}".format(result["teacher_unsafe"]))
    print("Optimizer updates: 0")
    if coverage < 0.005:
        print("Warning: coverage is too sparse for useful teacher supervision")
    elif coverage > 0.50:
        print("Warning: coverage is high; inspect whether the gate is too permissive")


if __name__ == "__main__":
    main()
