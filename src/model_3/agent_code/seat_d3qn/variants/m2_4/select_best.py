"""Validate every snapshot in one M2-4 branch and install its best model."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import (
    BEST_MODEL_PATH,
    BRANCH,
    C3_SCORE_EXCEPTION,
    C3_SCORE_TIE_WINDOW,
    C3_SUICIDE_TOLERANCE,
    EVALUATION_MODEL_ENV,
    MODEL_PATH,
    MODEL_SNAPSHOT_DIR,
    REPOSITORY_ROOT,
    RESULTS_DIR,
)


AGENT_NAME = "seat_d3qn"
OPPONENTS = ("peaceful_agent", "coin_collector_agent", "rule_based_agent")
DEFAULT_VALIDATION_SEEDS = (1101, 2202, 3303)
DEFAULT_CONFIRMATION_SEEDS = (4404, 5505, 6606)
VALIDATION_DIR = RESULTS_DIR / "validation"
C1_SUMMARY_PATH = REPOSITORY_ROOT / "results/m2_4/c1/validation/selection-summary.json"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds-per-seed", type=int, default=100)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_VALIDATION_SEEDS)
    )
    return parser.parse_args()


def _transition_from_name(path):
    marker = path.stem.rsplit("_t", 1)
    if len(marker) != 2 or not marker[1].isdigit():
        raise ValueError("Unexpected M2-4 snapshot name: {}".format(path.name))
    return int(marker[1])


def _run_validation(snapshot, seed, rounds):
    result_path = VALIDATION_DIR / "{}_seed{}.json".format(snapshot.stem, seed)
    if result_path.is_file():
        try:
            with result_path.open("r", encoding="utf-8") as file:
                result = json.load(file)
            statistics = result["by_agent"][AGENT_NAME]
            if int(statistics.get("rounds", 0)) == rounds:
                print(
                    "Validation {} seed {} already complete; skipping".format(
                        snapshot.stem, seed
                    )
                )
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
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        result_path.unlink()

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SEAT_D3QN_VARIANT"] = "m2_4"
    environment["SEAT_M2_4_BRANCH"] = BRANCH
    environment[EVALUATION_MODEL_ENV] = str(snapshot.resolve())
    for name in (
        "SEAT_D3QN_RESUME",
        "SEAT_D3QN_SESSION_TRANSITION_LIMIT",
        "SEAT_D3QN_CHECKPOINT_AFTER_ROUNDS",
        "SEAT_M2_4_CURRICULUM_BLOCK",
    ):
        environment.pop(name, None)
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
    with result_path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    statistics = result["by_agent"][AGENT_NAME]
    if int(statistics.get("rounds", 0)) != rounds:
        raise RuntimeError("Validation did not finish all rounds")
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


def _summarise(snapshot, runs):
    rounds = sum(run["rounds"] for run in runs)
    return {
        "snapshot": str(snapshot.resolve()),
        "transition_count": _transition_from_name(snapshot),
        "validation_rounds": rounds,
        "mean_official_score": sum(run["score"] for run in runs) / rounds,
        "kills_per_round": sum(run["kills"] for run in runs) / rounds,
        "suicide_rate": sum(run["suicides"] for run in runs) / rounds,
        "runs": runs,
    }


def _combine(candidate, extra_runs):
    return _summarise(
        Path(candidate["snapshot"]), list(candidate["runs"]) + list(extra_runs)
    )


def _c1_baseline():
    if not C1_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            "C3 selection needs the completed C1 validation summary"
        )
    with C1_SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    return C1_SUMMARY_PATH, summary["best_snapshot"]


def _passes_c3_guardrail(candidate, baseline):
    suicide_limit = float(baseline["suicide_rate"]) + C3_SUICIDE_TOLERANCE
    score_gain = float(candidate["mean_official_score"]) - float(
        baseline["mean_official_score"]
    )
    return not (
        float(candidate["suicide_rate"]) > suicide_limit
        and score_gain < C3_SCORE_EXCEPTION
    )


def _choose_c3(confirmation):
    baseline_path, baseline = _c1_baseline()
    eligible = [
        candidate
        for candidate in confirmation
        if _passes_c3_guardrail(candidate, baseline)
    ]
    if not eligible:
        raise RuntimeError("Every C3 confirmation candidate failed the safety guardrail")

    highest_score = max(item["mean_official_score"] for item in eligible)
    close = [
        item
        for item in eligible
        if item["mean_official_score"] >= highest_score - C3_SCORE_TIE_WINDOW
    ]
    best = min(
        close,
        key=lambda item: (
            item["suicide_rate"],
            -item["mean_official_score"],
            -item["kills_per_round"],
            -item["transition_count"],
        ),
    )
    rejected = [
        candidate
        for candidate in confirmation
        if not _passes_c3_guardrail(candidate, baseline)
    ]
    return best, {
        "baseline_summary": str(baseline_path.relative_to(REPOSITORY_ROOT)),
        "baseline_score": float(baseline["mean_official_score"]),
        "baseline_suicide_rate": float(baseline["suicide_rate"]),
        "suicide_tolerance": C3_SUICIDE_TOLERANCE,
        "score_exception": C3_SCORE_EXCEPTION,
        "score_tie_window": C3_SCORE_TIE_WINDOW,
        "rejected_transitions": [item["transition_count"] for item in rejected],
    }


def main():
    args = _arguments()
    if args.rounds_per_seed <= 0:
        raise ValueError("rounds-per-seed must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Validation seeds must be non-empty and unique")
    snapshots = sorted(MODEL_SNAPSHOT_DIR.glob("m2_4_{}_t*.pt".format(BRANCH)))
    if not snapshots:
        raise FileNotFoundError("Finish M2-4 {} training first".format(BRANCH.upper()))

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    candidates = []
    for snapshot in snapshots:
        runs = [
            _run_validation(snapshot, seed, args.rounds_per_seed)
            for seed in args.seeds
        ]
        candidates.append(_summarise(snapshot, runs))

    top_two = sorted(
        candidates,
        key=lambda item: (
            item["mean_official_score"],
            item["kills_per_round"],
            -item["suicide_rate"],
            item["transition_count"],
        ),
        reverse=True,
    )[:2]
    if BRANCH == "c3":
        zero = next(item for item in candidates if item["transition_count"] == 0)
        if all(item["transition_count"] != 0 for item in top_two):
            top_two.append(zero)
    confirmation = []
    for candidate in top_two:
        snapshot = Path(candidate["snapshot"])
        extra = [
            _run_validation(snapshot, seed, args.rounds_per_seed)
            for seed in DEFAULT_CONFIRMATION_SEEDS
        ]
        confirmation.append(_combine(candidate, extra))

    guardrail = None
    if BRANCH == "c3":
        best, guardrail = _choose_c3(confirmation)
    else:
        best = max(
            confirmation,
            key=lambda item: (
                item["mean_official_score"],
                item["kills_per_round"],
                -item["suicide_rate"],
                item["transition_count"],
            ),
        )
    shutil.copy2(best["snapshot"], BEST_MODEL_PATH)
    shutil.copy2(best["snapshot"], MODEL_PATH)
    summary = {
        "branch": BRANCH,
        "selection_metric": "mean_official_score",
        "tie_breakers": [
            "higher_kills_per_round",
            "lower_suicide_rate",
            "later_transition",
        ],
        "validation_seeds": list(args.seeds),
        "confirmation_seeds": list(DEFAULT_CONFIRMATION_SEEDS),
        "rounds_per_seed": args.rounds_per_seed,
        "opponents": list(OPPONENTS),
        "scenario": "classic",
        "best_snapshot": best,
        "candidates": candidates,
        "confirmation_candidates": confirmation,
    }
    if guardrail is not None:
        summary["c3_guardrail"] = guardrail
        summary["selection_metric"] = "score_with_c1_suicide_guardrail"
        summary["tie_breakers"] = [
            "within_0.10_score_choose_lower_suicide",
            "higher_score",
            "higher_kills_per_round",
            "later_transition",
        ]
    summary_path = VALIDATION_DIR / "selection-summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, summary_path)
    print(
        "Selected {} with validation score {:.4f}".format(
            Path(best["snapshot"]).name, best["mean_official_score"]
        )
    )
    print("Installed best weights at {}".format(MODEL_PATH))
    print("Selection summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
