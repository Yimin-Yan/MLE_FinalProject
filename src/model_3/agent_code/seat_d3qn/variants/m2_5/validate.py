"""Validate the fitted recovery gate against its exact frozen source."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .checks import verify_preflight
from .config import (
    BEST_MODEL_PATH,
    C1_MODEL_PATH,
    COLLECTION_SEED_ENV,
    DISABLE_GATE_ENV,
    DOMAIN_ID_ENV,
    MODE_ENV,
    MODEL_ENV,
    MODEL_PATH,
    REPOSITORY_ROOT,
    RESULTS_DIR,
    RISK_CHECKPOINT_PATH,
    SHARD_ENV,
    VALIDATION_DIR,
)


AGENT = "seat_d3qn"
VALIDATION_SEEDS = (56_001, 56_002, 56_003, 56_004, 56_005)
ROUNDS_PER_BLOCK = 100
DOMAINS = {
    "official": ("peaceful_agent", "coin_collector_agent", "rule_based_agent"),
    "pressure": ("rule_based_agent", "rule_based_agent", "coin_collector_agent"),
    # Different composition and more noise than the 15% collection domain.
    "heldout_noisy": ("m2_4_noisy_agent", "peaceful_agent", "rule_based_agent"),
}
METRICS = ("score", "coins", "kills", "suicides", "steps", "bombs", "crates", "invalid")


def _read(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    stats = data.get("by_agent", {}).get(AGENT, {})
    if int(stats.get("rounds", 0)) != ROUNDS_PER_BLOCK:
        raise ValueError("Incomplete M2-5 validation result: {}".format(path))
    return {name: float(stats.get(name, 0)) / ROUNDS_PER_BLOCK for name in METRICS}


def _run(condition, domain, seed):
    directory = VALIDATION_DIR / condition / domain
    directory.mkdir(parents=True, exist_ok=True)
    result = directory / "seed{}.json".format(seed)
    if result.is_file():
        try:
            return _read(result)
        except (OSError, ValueError, json.JSONDecodeError):
            result.unlink()

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "SEAT_D3QN_VARIANT": "m2_5",
            MODEL_ENV: str(RISK_CHECKPOINT_PATH.resolve()),
            DISABLE_GATE_ENV: "1" if condition == "source" else "0",
            "SEAT_M2_4_OPPONENT_MODEL_PATH": str(C1_MODEL_PATH.resolve()),
            "SEAT_M2_4_NOISY_SEED": str(seed + 900_000_000),
            "SEAT_M2_4_NOISE_PROBABILITY": "0.25",
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
        *DOMAINS[domain],
        "--scenario",
        "classic",
        "--n-rounds",
        str(ROUNDS_PER_BLOCK),
        "--no-gui",
        "--seed",
        str(seed),
        "--save-stats",
        str(result),
    ]
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)
    return _read(result)


def _bootstrap_ci(values, seed=550_055, samples=100_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return [float(lower), float(upper)]


def _summarize(records):
    output = {}
    for metric in METRICS:
        values = [item[metric] for item in records]
        output[metric + "_per_round"] = float(np.mean(values))
    return output


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def main():
    preflight = verify_preflight(require_risk=True)
    raw = {"source": {}, "recovery": {}}
    for condition in raw:
        for domain in DOMAINS:
            raw[condition][domain] = [
                _run(condition, domain, seed) for seed in VALIDATION_SEEDS
            ]

    source_blocks = []
    recovery_blocks = []
    domain_delta = {}
    for domain in DOMAINS:
        source_blocks.extend(raw["source"][domain])
        recovery_blocks.extend(raw["recovery"][domain])
        source_score = np.mean([item["score"] for item in raw["source"][domain]])
        recovery_score = np.mean([item["score"] for item in raw["recovery"][domain]])
        domain_delta[domain] = float(recovery_score - source_score)

    metric_delta = {}
    metric_ci = {}
    for metric in METRICS:
        differences = np.asarray(
            [
                recovery[metric] - source[metric]
                for source, recovery in zip(source_blocks, recovery_blocks)
            ],
            dtype=np.float64,
        )
        metric_delta[metric] = float(differences.mean())
        metric_ci[metric] = _bootstrap_ci(differences, seed=550_055 + len(metric))

    nonnegative_domains = sum(value >= 0.0 for value in domain_delta.values())
    checks = {
        "aggregate_score_nonnegative": metric_delta["score"] >= 0.0,
        "official_score_nonnegative": domain_delta["official"] >= 0.0,
        "worst_domain_above_minus_0_15": min(domain_delta.values()) >= -0.15,
        "at_least_two_domains_nonnegative": nonnegative_domains >= 2,
        "kills_not_lost": metric_delta["kills"] >= -0.03,
        "coins_not_lost": metric_delta["coins"] >= -0.05,
        "suicide_not_increased": metric_delta["suicides"] <= 0.02,
    }
    passed = all(checks.values())
    summary = {
        "protocol": {
            "seeds": list(VALIDATION_SEEDS),
            "rounds_per_seed_domain": ROUNDS_PER_BLOCK,
            "domains": {name: list(agents) for name, agents in DOMAINS.items()},
            "source_checkpoint": preflight["source_sha256"],
            "risk_checkpoint": str(RISK_CHECKPOINT_PATH.resolve()),
        },
        "source": _summarize(source_blocks),
        "recovery": _summarize(recovery_blocks),
        "recovery_minus_source": metric_delta,
        "paired_bootstrap_95_ci": metric_ci,
        "domain_score_delta": domain_delta,
        "guardrails": checks,
        "passed": passed,
    }
    output = VALIDATION_DIR / "selection-summary.json"
    _write_json(output, summary)

    print("\nM2-5 recovery minus frozen C3-v2")
    for metric in ("score", "coins", "kills", "suicides", "steps", "bombs"):
        lower, upper = metric_ci[metric]
        print(
            "{:<10s} {:+.4f}  95% CI [{:+.4f}, {:+.4f}]".format(
                metric, metric_delta[metric], lower, upper
            )
        )
    print("Domain score deltas: {}".format(domain_delta))
    if not passed:
        print("M2-5 recovery gate failed validation; C3-v2 remains the fallback")
        print("Validation summary: {}".format(output))
        return

    shutil.copy2(RISK_CHECKPOINT_PATH, BEST_MODEL_PATH)
    shutil.copy2(RISK_CHECKPOINT_PATH, MODEL_PATH)
    print("M2-5 recovery gate passed every guardrail")
    print("Installed best weights at {}".format(MODEL_PATH))
    print("Validation summary: {}".format(output))


if __name__ == "__main__":
    main()

