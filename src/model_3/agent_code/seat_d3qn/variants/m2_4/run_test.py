"""Evaluate the installed model for one M2-4 branch on 30 held-out seeds."""

import argparse
import hashlib
import json
import os
import subprocess
import sys

from .config import BRANCH, EVALUATION_MODEL_ENV, MODEL_PATH, REPOSITORY_ROOT, RESULTS_DIR


AGENT_NAME = "seat_d3qn"
OPPONENTS = ("peaceful_agent", "coin_collector_agent", "rule_based_agent")
if BRANCH == "c3":
    DEFAULT_TEST_SEEDS = tuple(range(10_001, 10_031))
else:
    DEFAULT_TEST_SEEDS = tuple(range(8_101, 8_131))
TEST_DIR = RESULTS_DIR / "test"
SUMMARY_PATH = RESULTS_DIR / "m2_4_{}_test_1500.json".format(BRANCH)
MODEL_MARKER = TEST_DIR / "model-sha256.txt"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds-per-seed", type=int, default=50)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_TEST_SEEDS)
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_path(seed, rounds):
    return TEST_DIR / "m2_4_{}_seed{}_{}.json".format(BRANCH, seed, rounds)


def _read(path, rounds):
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    statistics = result.get("by_agent", {}).get(AGENT_NAME, {})
    if int(statistics.get("rounds", 0)) != rounds:
        raise ValueError("Incomplete test result in {}".format(path))
    return result


def _prepare(model_hash, rerun):
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    old_hash = MODEL_MARKER.read_text(encoding="utf-8").strip() if MODEL_MARKER.is_file() else None
    if rerun or old_hash != model_hash:
        for path in TEST_DIR.glob("m2_4_{}_seed*_*.json".format(BRANCH)):
            path.unlink()
        if SUMMARY_PATH.is_file():
            SUMMARY_PATH.unlink()
    temporary = MODEL_MARKER.with_suffix(".tmp")
    temporary.write_text(model_hash + "\n", encoding="utf-8")
    os.replace(temporary, MODEL_MARKER)


def _run_seed(seed, rounds):
    result_path = _result_path(seed, rounds)
    if result_path.is_file():
        try:
            _read(result_path, rounds)
            print("Test seed {} already complete; skipping".format(seed))
            return result_path
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            result_path.unlink()

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SEAT_D3QN_VARIANT"] = "m2_4"
    environment["SEAT_M2_4_BRANCH"] = BRANCH
    environment[EVALUATION_MODEL_ENV] = str(MODEL_PATH.resolve())
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
    _read(result_path, rounds)
    return result_path


def _sum(results):
    totals = {}
    for result in results:
        for agent, statistics in result["by_agent"].items():
            target = totals.setdefault(agent, {})
            for key, value in statistics.items():
                if isinstance(value, (int, float)):
                    target[key] = target.get(key, 0) + value
    return totals


def main():
    args = _arguments()
    if args.rounds_per_seed <= 0:
        raise ValueError("rounds-per-seed must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Test seeds must be non-empty and unique")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Select the best M2-4 {} model first".format(BRANCH.upper()))

    model_hash = _sha256(MODEL_PATH)
    _prepare(model_hash, args.rerun)
    paths = [_run_seed(seed, args.rounds_per_seed) for seed in args.seeds]
    totals = _sum([_read(path, args.rounds_per_seed) for path in paths])
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
            "seeds": list(args.seeds),
            "rounds_per_seed": args.rounds_per_seed,
            "total_rounds": len(args.seeds) * args.rounds_per_seed,
        },
        "branch": BRANCH,
        "model_path": str(MODEL_PATH.resolve()),
        "model_sha256": model_hash,
        "by_agent": totals,
        "derived": derived,
        "result_files": [str(path.relative_to(REPOSITORY_ROOT)) for path in paths],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, SUMMARY_PATH)
    ranking = sorted(
        derived.items(), key=lambda item: item[1]["score_per_round"], reverse=True
    )
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
    print("Test summary: {}".format(SUMMARY_PATH))


if __name__ == "__main__":
    main()
