"""Run formal M2-3 training, validation, and held-out multi-seed testing."""

import argparse
import hashlib
import json
import os
import subprocess
import sys

from .config import (
    EVALUATION_MODEL_ENV,
    MAX_TRAINING_TRANSITIONS,
    MODEL_PATH,
    MODEL_SNAPSHOT_DIR,
    REPOSITORY_ROOT,
    RESULTS_DIR,
    VARIANT_ID,
)
from .select_best import DEFAULT_CONFIRMATION_SEEDS, DEFAULT_VALIDATION_SEEDS


AGENT_NAME = "seat_d3qn"
OPPONENTS = ("peaceful_agent", "coin_collector_agent", "rule_based_agent")
DEFAULT_TEST_SEEDS = tuple(range(8101, 8131))
TEST_DIR = RESULTS_DIR / "test"
TEST_SUMMARY_PATH = RESULTS_DIR / "m2_3_test_1500_multiseed.json"
TEST_MODEL_MARKER = TEST_DIR / "model-sha256.txt"
VALIDATION_SUMMARY_PATH = RESULTS_DIR / "validation" / "selection-summary.json"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rounds-per-session", type=int, default=500)
    parser.add_argument("--validation-rounds-per-seed", type=int, default=100)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    parser.add_argument("--test-rounds-per-seed", type=int, default=50)
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TEST_SEEDS),
    )
    parser.add_argument("--rerun-validation", action="store_true")
    parser.add_argument("--rerun-test", action="store_true")
    return parser.parse_args()


def _run(command, environment=None):
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def _validate_arguments(args):
    if args.max_rounds_per_session <= 0:
        raise ValueError("max-rounds-per-session must be positive")
    if args.validation_rounds_per_seed <= 0:
        raise ValueError("validation-rounds-per-seed must be positive")
    if args.test_rounds_per_seed <= 0:
        raise ValueError("test-rounds-per-seed must be positive")
    if not args.validation_seeds or len(set(args.validation_seeds)) != len(
        args.validation_seeds
    ):
        raise ValueError("Validation seeds must be non-empty and unique")
    if not args.test_seeds or len(set(args.test_seeds)) != len(args.test_seeds):
        raise ValueError("Test seeds must be non-empty and unique")
    selection_seeds = set(args.validation_seeds) | set(DEFAULT_CONFIRMATION_SEEDS)
    overlap = selection_seeds & set(args.test_seeds)
    if overlap:
        raise ValueError(
            "Test seeds overlap checkpoint-selection seeds: {}".format(
                sorted(overlap)
            )
        )


def _train(args):
    _run(
        [
            sys.executable,
            "-m",
            "agent_code.seat_d3qn.variants.m2_3.run_curriculum",
            "--max-rounds-per-session",
            str(args.max_rounds_per_session),
        ]
    )
    prefix = VARIANT_ID.lower().replace("-", "_")
    final_snapshot = MODEL_SNAPSHOT_DIR / "{}_t{:07d}.pt".format(
        prefix,
        MAX_TRAINING_TRANSITIONS,
    )
    if not final_snapshot.is_file():
        raise RuntimeError("Training finished without the 250k snapshot")


def _validation_is_complete(args):
    if not VALIDATION_SUMMARY_PATH.is_file() or not MODEL_PATH.is_file():
        return False
    try:
        with open(VALIDATION_SUMMARY_PATH, "r", encoding="utf-8") as file:
            summary = json.load(file)
        transitions = {
            int(candidate.get("transition_count", -1))
            for candidate in summary.get("candidates", [])
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        MAX_TRAINING_TRANSITIONS in transitions
        and summary.get("validation_seeds") == list(args.validation_seeds)
        and int(summary.get("rounds_per_seed", -1))
        == args.validation_rounds_per_seed
    )


def _validate(args):
    if not args.rerun_validation and _validation_is_complete(args):
        print("Validation already complete; using the installed best checkpoint")
        return
    _run(
        [
            sys.executable,
            "-m",
            "agent_code.seat_d3qn.variants.m2_3.select_best",
            "--rounds-per-seed",
            str(args.validation_rounds_per_seed),
            "--seeds",
            *[str(seed) for seed in args.validation_seeds],
        ]
    )
    if not _validation_is_complete(args):
        raise RuntimeError("Checkpoint selection did not produce a complete summary")


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_path(seed, rounds):
    return TEST_DIR / "m2_3_seed{}_{}.json".format(seed, rounds)


def _read_result(path, expected_rounds):
    with open(path, "r", encoding="utf-8") as file:
        result = json.load(file)
    by_agent = result.get("by_agent", {})
    expected_agents = {AGENT_NAME, *OPPONENTS}
    if set(by_agent) != expected_agents:
        raise ValueError("Unexpected agent set in {}".format(path))
    for name in expected_agents:
        if int(by_agent[name].get("rounds", 0)) != expected_rounds:
            raise ValueError("Incomplete test result in {}".format(path))
    return result


def _prepare_test_directory(model_hash, rerun):
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    previous_hash = None
    if TEST_MODEL_MARKER.is_file():
        previous_hash = TEST_MODEL_MARKER.read_text(encoding="utf-8").strip()
    if rerun or previous_hash != model_hash:
        for path in TEST_DIR.glob("m2_3_seed*_*.json"):
            path.unlink()
        if TEST_SUMMARY_PATH.is_file():
            TEST_SUMMARY_PATH.unlink()
    temporary = TEST_MODEL_MARKER.with_suffix(".tmp")
    temporary.write_text(model_hash + "\n", encoding="utf-8")
    os.replace(temporary, TEST_MODEL_MARKER)


def _run_test_seed(seed, rounds):
    result_path = _result_path(seed, rounds)
    if result_path.is_file():
        try:
            _read_result(result_path, rounds)
            print("Test seed {} already complete; skipping".format(seed))
            return result_path
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            result_path.unlink()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SEAT_D3QN_VARIANT"] = "m2_3"
    environment[EVALUATION_MODEL_ENV] = str(MODEL_PATH.resolve())
    for name in (
        "SEAT_D3QN_RESUME",
        "SEAT_D3QN_SESSION_TRANSITION_LIMIT",
        "SEAT_D3QN_CHECKPOINT_AFTER_ROUNDS",
        "SEAT_M2_3_CURRICULUM_BLOCK",
    ):
        environment.pop(name, None)
    _run(
        [
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
        ],
        environment,
    )
    _read_result(result_path, rounds)
    return result_path


def _sum_agent_statistics(results):
    totals = {}
    for result in results:
        for agent, statistics in result["by_agent"].items():
            aggregate = totals.setdefault(agent, {})
            for name, value in statistics.items():
                if isinstance(value, (int, float)):
                    aggregate[name] = aggregate.get(name, 0) + value
    return totals


def _write_test_summary(args, model_hash, paths):
    results = [_read_result(path, args.test_rounds_per_seed) for path in paths]
    totals = _sum_agent_statistics(results)
    derived = {}
    for agent, statistics in totals.items():
        rounds = int(statistics["rounds"])
        derived[agent] = {
            "score_per_round": statistics.get("score", 0) / rounds,
            "kills_per_round": statistics.get("kills", 0) / rounds,
            "suicide_rate": statistics.get("suicides", 0) / rounds,
        }
    summary = {
        "protocol": {
            "scenario": "classic",
            "opponents": list(OPPONENTS),
            "seeds": list(args.test_seeds),
            "rounds_per_seed": args.test_rounds_per_seed,
            "total_rounds": len(args.test_seeds) * args.test_rounds_per_seed,
            "checkpoint_selected_without_test_seeds": True,
        },
        "model_path": str(MODEL_PATH.resolve()),
        "model_sha256": model_hash,
        "by_agent": totals,
        "derived": derived,
        "result_files": [str(path.relative_to(REPOSITORY_ROOT)) for path in paths],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TEST_SUMMARY_PATH.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, TEST_SUMMARY_PATH)
    return summary


def _test(args):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Validation did not install the selected model")
    model_hash = _file_sha256(MODEL_PATH)
    _prepare_test_directory(model_hash, args.rerun_test)
    paths = [
        _run_test_seed(seed, args.test_rounds_per_seed)
        for seed in args.test_seeds
    ]
    summary = _write_test_summary(args, model_hash, paths)
    ranking = sorted(
        summary["derived"].items(),
        key=lambda item: item[1]["score_per_round"],
        reverse=True,
    )
    print("\nFormal multi-seed ranking")
    for position, (agent, metrics) in enumerate(ranking, start=1):
        print(
            "{}. {} score/game={:.4f} kills/game={:.4f} suicide={:.2%}".format(
                position,
                agent,
                metrics["score_per_round"],
                metrics["kills_per_round"],
                metrics["suicide_rate"],
            )
        )
    print("Test summary: {}".format(TEST_SUMMARY_PATH))


def main():
    args = _arguments()
    _validate_arguments(args)
    print("\n=== Training: 250k interleaved curriculum ===", flush=True)
    _train(args)
    print("\n=== Validation: held-out checkpoint selection ===", flush=True)
    _validate(args)
    print(
        "\n=== Test: {} held-out seeds x {} rounds ===".format(
            len(args.test_seeds),
            args.test_rounds_per_seed,
        ),
        flush=True,
    )
    _test(args)


if __name__ == "__main__":
    main()
