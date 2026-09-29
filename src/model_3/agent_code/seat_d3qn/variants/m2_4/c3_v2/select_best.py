"""Validate all C3-v2 snapshots and install the guarded best model."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .checks import load_torch, verify_all
from .config import (
    BEST_MODEL_PATH,
    EVALUATION_MODEL_ENV,
    MAX_TRAINING_TRANSITIONS,
    MODEL_PATH,
    MODEL_SNAPSHOT_DIR,
    REPOSITORY_ROOT,
    RESULTS_DIR,
    TRAINING_CHECKPOINT_PATH,
)


AGENT_NAME = "seat_d3qn"
OPPONENTS = ("peaceful_agent", "coin_collector_agent", "rule_based_agent")
VALIDATION_SEEDS = (8_301, 8_302, 8_303)
CONFIRMATION_SEEDS = (8_401, 8_402, 8_403)
VALIDATION_DIR = RESULTS_DIR / "validation"
SUICIDE_TOLERANCE = 0.05
SCORE_EXCEPTION = 0.30
SCORE_TIE_WINDOW = 0.10


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds-per-seed", type=int, default=100)
    return parser.parse_args()


def _transition(path):
    value = path.stem.rsplit("_t", 1)[-1]
    if not value.isdigit():
        raise ValueError("Unexpected C3-v2 snapshot name: " + path.name)
    return int(value)


def _read(path, rounds):
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    statistics = result.get("by_agent", {}).get(AGENT_NAME, {})
    if int(statistics.get("rounds", 0)) != rounds:
        raise ValueError("Incomplete validation result: {}".format(path))
    return {
        "rounds": rounds,
        "score": int(statistics.get("score", 0)),
        "kills": int(statistics.get("kills", 0)),
        "coins": int(statistics.get("coins", 0)),
        "suicides": int(statistics.get("suicides", 0)),
        "invalid": int(statistics.get("invalid", 0)),
        "steps": int(statistics.get("steps", 0)),
    }


def _run(snapshot, seed, rounds):
    result_path = VALIDATION_DIR / "{}_seed{}.json".format(snapshot.stem, seed)
    if result_path.is_file():
        try:
            return _read(result_path, rounds)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            result_path.unlink()

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "SEAT_D3QN_VARIANT": "m2_4_c3_v2",
            EVALUATION_MODEL_ENV: str(snapshot.resolve()),
        }
    )
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
    return _read(result_path, rounds)


def _summarise(snapshot, runs):
    rounds = sum(run["rounds"] for run in runs)
    return {
        "snapshot": str(snapshot.resolve()),
        "transition_count": _transition(snapshot),
        "validation_rounds": rounds,
        "mean_official_score": sum(run["score"] for run in runs) / rounds,
        "kills_per_round": sum(run["kills"] for run in runs) / rounds,
        "suicide_rate": sum(run["suicides"] for run in runs) / rounds,
        "runs": runs,
    }


def _passes_guardrail(candidate, baseline):
    score_gain = (
        candidate["mean_official_score"] - baseline["mean_official_score"]
    )
    return not (
        candidate["suicide_rate"] > baseline["suicide_rate"] + SUICIDE_TOLERANCE
        and score_gain < SCORE_EXCEPTION
    )


def main():
    args = _arguments()
    if args.rounds_per_seed < 1:
        raise ValueError("rounds-per-seed must be positive")
    verify_all()
    if not TRAINING_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError("Finish C3-v2 training before validation")
    checkpoint = load_torch(TRAINING_CHECKPOINT_PATH, weights_only=False)
    if int(checkpoint.get("transition_count", -1)) != MAX_TRAINING_TRANSITIONS:
        raise RuntimeError("C3-v2 did not reach the complete 500k budget")

    snapshots = sorted(MODEL_SNAPSHOT_DIR.glob("m2_4_c3_t*.pt"))
    expected = set(range(0, MAX_TRAINING_TRANSITIONS + 1, 25_000))
    actual = {_transition(path) for path in snapshots}
    if actual != expected:
        raise RuntimeError("C3-v2 snapshots do not cover every 25k checkpoint")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    candidates = []
    for snapshot in snapshots:
        runs = [
            _run(snapshot, seed, args.rounds_per_seed)
            for seed in VALIDATION_SEEDS
        ]
        candidates.append(_summarise(snapshot, runs))

    first_pass = sorted(
        candidates,
        key=lambda item: (
            item["mean_official_score"],
            item["kills_per_round"],
            -item["suicide_rate"],
            item["transition_count"],
        ),
        reverse=True,
    )[:2]
    zero = next(item for item in candidates if item["transition_count"] == 0)
    if all(item["transition_count"] != 0 for item in first_pass):
        first_pass.append(zero)

    confirmation = []
    for candidate in first_pass:
        snapshot = Path(candidate["snapshot"])
        extra = [
            _run(snapshot, seed, args.rounds_per_seed)
            for seed in CONFIRMATION_SEEDS
        ]
        confirmation.append(
            _summarise(snapshot, list(candidate["runs"]) + extra)
        )
    baseline = next(
        item for item in confirmation if item["transition_count"] == 0
    )
    eligible = [
        item for item in confirmation if _passes_guardrail(item, baseline)
    ]
    if not eligible:
        raise RuntimeError("Every C3-v2 candidate failed the C1 guardrail")
    highest = max(item["mean_official_score"] for item in eligible)
    close = [
        item
        for item in eligible
        if item["mean_official_score"] >= highest - SCORE_TIE_WINDOW
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
    shutil.copy2(best["snapshot"], BEST_MODEL_PATH)
    shutil.copy2(best["snapshot"], MODEL_PATH)
    summary = {
        "branch": "c3_v2",
        "selection_metric": "score_with_frozen_c1_safety_guardrail",
        "validation_seeds": list(VALIDATION_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "rounds_per_seed": args.rounds_per_seed,
        "opponents": list(OPPONENTS),
        "guardrail": {
            "baseline_transition": 0,
            "suicide_tolerance": SUICIDE_TOLERANCE,
            "score_exception": SCORE_EXCEPTION,
            "score_tie_window": SCORE_TIE_WINDOW,
        },
        "best_snapshot": best,
        "candidates": candidates,
        "confirmation_candidates": confirmation,
    }
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
