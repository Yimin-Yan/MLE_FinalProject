"""Validate matched-control snapshots and install the best checkpoint."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import (
    BEST_MODEL_PATH,
    EVALUATION_MODEL_ENV,
    MODEL_PATH,
    MODEL_SNAPSHOT_DIR,
    REPOSITORY_ROOT,
    RESULTS_DIR,
    VARIANT_ENV,
)


AGENT_NAME = "seat_d3qn"
OPPONENTS = ("peaceful_agent", "coin_collector_agent", "rule_based_agent")
DEFAULT_VALIDATION_SEEDS = (1101, 2202, 3303)
DEFAULT_ROUNDS_PER_SEED = 100
VALIDATION_DIR = RESULTS_DIR / "validation"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds-per-seed", type=int, default=DEFAULT_ROUNDS_PER_SEED)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    return parser.parse_args()


def _transition_from_name(snapshot_path):
    marker = snapshot_path.stem.rsplit("_t", 1)
    if len(marker) != 2 or not marker[1].isdigit():
        raise ValueError("Unexpected control snapshot name: {}".format(snapshot_path.name))
    return int(marker[1])


def _run_one_validation(snapshot_path, seed, rounds):
    result_path = VALIDATION_DIR / "{}_seed{}.json".format(snapshot_path.stem, seed)
    environment = os.environ.copy()
    environment[VARIANT_ENV] = "m2_2_control"
    environment[EVALUATION_MODEL_ENV] = str(snapshot_path.resolve())
    command = [
        sys.executable,
        "main.py",
        "play",
        "--agents",
        AGENT_NAME,
        *OPPONENTS,
        "--scenario",
        "classic",
        "--n-rounds",
        str(rounds),
        "--no-gui",
        "--seed",
        str(seed),
        "--save-stats",
        str(result_path),
    ]
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)

    with open(result_path, "r", encoding="utf-8") as file:
        result = json.load(file)
    statistics = result["by_agent"][AGENT_NAME]
    if int(statistics.get("rounds", 0)) != rounds:
        raise RuntimeError("Control validation did not finish all rounds")
    return {
        "seed": seed,
        "rounds": rounds,
        "score": int(statistics.get("score", 0)),
        "coins": int(statistics.get("coins", 0)),
        "kills": int(statistics.get("kills", 0)),
        "suicides": int(statistics.get("suicides", 0)),
        "invalid": int(statistics.get("invalid", 0)),
        "steps": int(statistics.get("steps", 0)),
        "result_file": str(result_path.relative_to(REPOSITORY_ROOT)),
    }


def _validate_snapshot(snapshot_path, seeds, rounds_per_seed):
    runs = [
        _run_one_validation(snapshot_path, seed, rounds_per_seed)
        for seed in seeds
    ]
    total_rounds = sum(run["rounds"] for run in runs)
    total_score = sum(run["score"] for run in runs)
    total_suicides = sum(run["suicides"] for run in runs)
    total_kills = sum(run["kills"] for run in runs)
    return {
        "snapshot": str(snapshot_path.resolve()),
        "transition_count": _transition_from_name(snapshot_path),
        "validation_rounds": total_rounds,
        "mean_official_score": total_score / total_rounds,
        "suicide_rate": total_suicides / total_rounds,
        "kills_per_round": total_kills / total_rounds,
        "runs": runs,
    }


def _write_summary(summary):
    summary_path = VALIDATION_DIR / "selection-summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, summary_path)
    return summary_path


def main():
    args = _arguments()
    if args.rounds_per_seed <= 0:
        raise ValueError("rounds-per-seed must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Validation seeds must be non-empty and unique")

    snapshots = sorted(MODEL_SNAPSHOT_DIR.glob("m2_2_control_t*.pt"))
    if not snapshots:
        raise FileNotFoundError("No matched-control snapshots were found")
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [
        _validate_snapshot(snapshot, args.seeds, args.rounds_per_seed)
        for snapshot in snapshots
    ]
    best = max(
        candidates,
        key=lambda item: (
            item["mean_official_score"],
            -item["suicide_rate"],
            item["kills_per_round"],
            item["transition_count"],
        ),
    )

    shutil.copy2(best["snapshot"], BEST_MODEL_PATH)
    shutil.copy2(best["snapshot"], MODEL_PATH)
    summary = {
        "selection_metric": "mean_official_score",
        "tie_breakers": [
            "lower_suicide_rate",
            "higher_kills_per_round",
            "later_transition",
        ],
        "validation_seeds": list(args.seeds),
        "rounds_per_seed": args.rounds_per_seed,
        "opponents": list(OPPONENTS),
        "scenario": "classic",
        "experiment_role": "m2_2_matched_control",
        "best_snapshot": best,
        "candidates": candidates,
    }
    summary_path = _write_summary(summary)
    print(
        "Selected {} with validation score {:.4f}".format(
            Path(best["snapshot"]).name,
            best["mean_official_score"],
        )
    )
    print("Installed best control weights at {}".format(MODEL_PATH))
    print("Selection summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
