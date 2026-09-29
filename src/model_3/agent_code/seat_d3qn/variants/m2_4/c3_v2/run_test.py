"""Test the selected C3-v2 model on held-out seeds with seed-level CIs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from .config import EVALUATION_MODEL_ENV, MODEL_PATH, REPOSITORY_ROOT, RESULTS_DIR


AGENT_NAME = "seat_d3qn"
RULE_AGENT = "rule_based_agent"
OPPONENTS = ("peaceful_agent", "coin_collector_agent", RULE_AGENT)
DEFAULT_TEST_SEEDS = tuple(range(10_001, 10_016))
TEST_DIR = RESULTS_DIR / "test"
MODEL_MARKER = TEST_DIR / "model-sha256.txt"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds-per-seed", type=int, default=100)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_TEST_SEEDS)
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path, rounds):
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    for agent in (AGENT_NAME, RULE_AGENT):
        if int(result.get("by_agent", {}).get(agent, {}).get("rounds", 0)) != rounds:
            raise ValueError("Incomplete test result: {}".format(path))
    return result


def _run(seed, rounds):
    path = TEST_DIR / "m2_4_c3_seed{}_{}.json".format(seed, rounds)
    if path.is_file():
        try:
            _read(path, rounds)
            return path
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            path.unlink()
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "SEAT_D3QN_VARIANT": "m2_4_c3_v2",
            EVALUATION_MODEL_ENV: str(MODEL_PATH.resolve()),
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
        str(path),
    ]
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)
    _read(path, rounds)
    return path


def _totals(results):
    output = {}
    for result in results:
        for agent, statistics in result["by_agent"].items():
            target = output.setdefault(agent, {})
            for name, value in statistics.items():
                if isinstance(value, (int, float)):
                    target[name] = target.get(name, 0) + value
    return output


def _seed_metric(result, agent, name):
    statistics = result["by_agent"][agent]
    rounds = float(statistics["rounds"])
    if name == "score":
        return statistics.get("score", 0) / rounds
    if name == "kills":
        return statistics.get("kills", 0) / rounds
    if name == "suicides":
        return statistics.get("suicides", 0) / rounds
    raise ValueError("Unknown test metric: " + name)


def _bootstrap_interval(values, rng, repeats=10_000):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _confidence_intervals(results):
    rng = np.random.default_rng(20_260_919)
    output = {}
    for agent in (AGENT_NAME, RULE_AGENT):
        output[agent] = {}
        for metric in ("score", "kills", "suicides"):
            values = [_seed_metric(result, agent, metric) for result in results]
            output[agent][metric] = {
                "mean": float(np.mean(values)),
                "ci95": _bootstrap_interval(values, rng),
            }
    paired = [
        _seed_metric(result, AGENT_NAME, "score")
        - _seed_metric(result, RULE_AGENT, "score")
        for result in results
    ]
    output["seat_minus_rule_score"] = {
        "mean": float(np.mean(paired)),
        "ci95": _bootstrap_interval(paired, rng),
    }
    return output


def main():
    args = _arguments()
    if args.rounds_per_seed < 1:
        raise ValueError("rounds-per-seed must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Test seeds must be non-empty and unique")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Select the best C3-v2 model first")

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    total_rounds = len(args.seeds) * args.rounds_per_seed
    summary_path = RESULTS_DIR / "m2_4_c3_test_{}.json".format(total_rounds)
    model_hash = _sha(MODEL_PATH)
    old_hash = MODEL_MARKER.read_text().strip() if MODEL_MARKER.is_file() else None
    if args.rerun or old_hash != model_hash:
        for path in TEST_DIR.glob("m2_4_c3_seed*_*.json"):
            path.unlink()
        for path in RESULTS_DIR.glob("m2_4_c3_test_*.json"):
            path.unlink()
    MODEL_MARKER.write_text(model_hash + "\n", encoding="utf-8")

    paths = [_run(seed, args.rounds_per_seed) for seed in args.seeds]
    results = [_read(path, args.rounds_per_seed) for path in paths]
    totals = _totals(results)
    derived = {}
    for agent, statistics in totals.items():
        rounds = int(statistics["rounds"])
        derived[agent] = {
            "score_per_round": statistics.get("score", 0) / rounds,
            "kills_per_round": statistics.get("kills", 0) / rounds,
            "suicide_rate": statistics.get("suicides", 0) / rounds,
        }
    intervals = _confidence_intervals(results)
    summary = {
        "protocol": {
            "scenario": "classic",
            "opponents": list(OPPONENTS),
            "seeds": list(args.seeds),
            "rounds_per_seed": args.rounds_per_seed,
            "total_rounds": total_rounds,
        },
        "model_path": str(MODEL_PATH.resolve()),
        "model_sha256": model_hash,
        "by_agent": totals,
        "derived": derived,
        "confidence_intervals": intervals,
        "result_files": [str(path.relative_to(REPOSITORY_ROOT)) for path in paths],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, summary_path)

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
    paired = intervals["seat_minus_rule_score"]
    print(
        "seat-rule score/game={:+.4f}, 95% CI [{:+.4f}, {:+.4f}]".format(
            paired["mean"], paired["ci95"][0], paired["ci95"][1]
        )
    )
    print("Test summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
