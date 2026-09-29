"""Run the final paired official test for C3-v2 and the M2-5 gate."""

import json
import os
import subprocess
import sys

import numpy as np

from .checks import verify_preflight
from .config import (
    COLLECTION_SEED_ENV,
    DISABLE_GATE_ENV,
    DOMAIN_ID_ENV,
    MODE_ENV,
    MODEL_ENV,
    REPOSITORY_ROOT,
    RESULTS_DIR,
    RISK_CHECKPOINT_PATH,
    SHARD_ENV,
)


AGENT = "seat_d3qn"
RULE_AGENT = "rule_based_agent"
OPPONENTS = ("peaceful_agent", "coin_collector_agent", RULE_AGENT)
TEST_SEEDS = tuple(range(61_001, 61_016))
ROUNDS_PER_SEED = 100
METRICS = ("score", "coins", "kills", "suicides", "steps", "bombs", "crates", "invalid")
TEST_DIR = RESULTS_DIR / "test" / "official_paired"


def _read(path):
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    for name in (AGENT, *OPPONENTS):
        rounds = result.get("by_agent", {}).get(name, {}).get("rounds", 0)
        if int(rounds) != ROUNDS_PER_SEED:
            raise ValueError("Incomplete official test result: {}".format(path))
    return result


def _run(condition, seed):
    directory = TEST_DIR / condition
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "seed{}.json".format(seed)
    if path.is_file():
        try:
            return _read(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            path.unlink()

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "SEAT_D3QN_VARIANT": "m2_5",
            MODEL_ENV: str(RISK_CHECKPOINT_PATH.resolve()),
            DISABLE_GATE_ENV: "1" if condition == "source" else "0",
        }
    )
    for name in (MODE_ENV, SHARD_ENV, DOMAIN_ID_ENV, COLLECTION_SEED_ENV):
        environment.pop(name, None)

    command = [
        sys.executable,
        "main.py",
        "play",
        "--agents",
        AGENT,
        *OPPONENTS,
        "--scenario",
        "classic",
        "--n-rounds",
        str(ROUNDS_PER_SEED),
        "--no-gui",
        "--seed",
        str(seed),
        "--save-stats",
        str(path),
    ]
    print("{} seed {}".format(condition.upper(), seed), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)
    return _read(path)


def _per_round(result, agent, metric):
    stats = result["by_agent"][agent]
    return float(stats.get(metric, 0)) / float(stats["rounds"])


def _interval(values, rng, repeats=100_000):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return [float(lower), float(upper)]


def _agent_summary(results, agent, rng):
    output = {}
    for metric in METRICS:
        values = [_per_round(result, agent, metric) for result in results]
        output[metric + "_per_round"] = {
            "mean": float(np.mean(values)),
            "ci95": _interval(values, rng),
        }
    return output


def _condition_summary(results, rng):
    agents = (AGENT, *OPPONENTS)
    return {agent: _agent_summary(results, agent, rng) for agent in agents}


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def main():
    preflight = verify_preflight(require_risk=True)
    raw = {
        condition: [_run(condition, seed) for seed in TEST_SEEDS]
        for condition in ("source", "recovery")
    }
    rng = np.random.default_rng(2_026_092_011)
    summaries = {
        condition: _condition_summary(results, rng)
        for condition, results in raw.items()
    }

    differences = {}
    for metric in METRICS:
        values = [
            _per_round(recovery, AGENT, metric) - _per_round(source, AGENT, metric)
            for source, recovery in zip(raw["source"], raw["recovery"])
        ]
        differences[metric + "_per_round"] = {
            "mean": float(np.mean(values)),
            "ci95": _interval(values, rng),
        }

    score_ci = differences["score_per_round"]["ci95"]
    if score_ci[0] > 0.0:
        decision = "stable_official_gain"
    elif score_ci[1] < 0.0:
        decision = "stable_official_loss"
    else:
        decision = "not_distinguishable"

    summary = {
        "protocol": {
            "scenario": "classic",
            "opponents": list(OPPONENTS),
            "seeds": list(TEST_SEEDS),
            "rounds_per_seed": ROUNDS_PER_SEED,
            "rounds_per_condition": len(TEST_SEEDS) * ROUNDS_PER_SEED,
            "bootstrap_unit": "seed",
            "source_checkpoint_sha256": preflight["source_sha256"],
            "risk_checkpoint": str(RISK_CHECKPOINT_PATH.resolve()),
        },
        "source_c3_v2": summaries["source"],
        "recovery_m2_5": summaries["recovery"],
        "m2_5_minus_c3_v2": differences,
        "decision": decision,
    }
    output = RESULTS_DIR / "test" / "m2_5_official_paired_test_1500.json"
    _write_json(output, summary)

    print("\nM2-5 minus frozen C3-v2 on the official test")
    for metric in ("score", "coins", "kills", "suicides", "steps", "bombs"):
        item = differences[metric + "_per_round"]
        print(
            "{:<10s} {:+.4f}  95% CI [{:+.4f}, {:+.4f}]".format(
                metric, item["mean"], item["ci95"][0], item["ci95"][1]
            )
        )
    print("Decision: {}".format(decision))
    print("Official test summary: {}".format(output))


if __name__ == "__main__":
    main()
