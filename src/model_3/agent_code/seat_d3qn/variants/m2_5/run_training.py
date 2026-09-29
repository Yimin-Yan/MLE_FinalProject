"""Collect 250k transitions in ten domains blocks, then fit the risk model."""

import json
import math
import os
import subprocess
import sys

from .checks import verify_preflight
from .config import (
    BLOCK_INDEX_ENV,
    BLOCK_TRANSITIONS,
    COLLECTION_BLOCKS,
    COLLECTION_ROUNDS_PER_SESSION,
    COLLECTION_SEED_ENV,
    DATASET_DIR,
    DOMAIN_ID_ENV,
    MASTER_SEED,
    MODE_ENV,
    REPOSITORY_ROOT,
    RESULTS_DIR,
    SHARD_ENV,
    STOCHASTIC_OPPONENT_MODEL_PATH,
)
from .dataset import shard_count
from .fit_risk import main as fit_risk, shard_path


def _completed_rounds(directory):
    rounds = 0
    for path in directory.glob("session_*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                stats = json.load(file).get("by_agent", {}).get("seat_d3qn", {})
            rounds += int(stats.get("rounds", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return rounds


def _round_estimate(current, completed_rounds):
    remaining = BLOCK_TRANSITIONS - current
    if current and completed_rounds:
        average = max(40.0, current / completed_rounds)
    else:
        average = 180.0
    estimate = int(math.ceil(1.08 * remaining / average)) + 1
    return max(1, min(COLLECTION_ROUNDS_PER_SESSION, estimate))


def _run_block(block_index, domain, opponents):
    path = shard_path(block_index, domain)
    session_root = RESULTS_DIR / ".collection_sessions" / "block_{:02d}_{}".format(
        block_index, domain
    )
    session_root.mkdir(parents=True, exist_ok=True)

    while True:
        current = shard_count(path)
        if current == BLOCK_TRANSITIONS:
            print(
                "block={:02d} domain={} reached {} transitions".format(
                    block_index, domain, current
                ),
                flush=True,
            )
            return
        if not 0 <= current < BLOCK_TRANSITIONS:
            raise RuntimeError("M2-5 block count is outside its budget")

        session_index = len(list(session_root.glob("session_*.json")))
        completed_rounds = _completed_rounds(session_root)
        rounds = _round_estimate(current, completed_rounds)
        seed = MASTER_SEED * 1_000_000 + block_index * 10_000 + session_index
        result = session_root / "session_{:03d}_{:05d}.json".format(
            session_index, current
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "SEAT_D3QN_VARIANT": "m2_5",
                MODE_ENV: "collect",
                SHARD_ENV: str(path.resolve()),
                BLOCK_INDEX_ENV: str(block_index),
                DOMAIN_ID_ENV: domain,
                COLLECTION_SEED_ENV: str(seed),
                "SEAT_M2_4_OPPONENT_MODEL_PATH": str(
                    STOCHASTIC_OPPONENT_MODEL_PATH.resolve()
                ),
                "SEAT_M2_4_NOISY_SEED": str(seed + 700_000_000),
                "SEAT_M2_4_NOISE_PROBABILITY": "0.15",
            }
        )
        environment.pop("SEAT_D3QN_MODEL_PATH", None)
        command = [
            sys.executable,
            "main.py",
            "play",
            "--agents",
            "seat_d3qn",
            *opponents,
            "--train",
            "1",
            "--scenario",
            "classic",
            "--n-rounds",
            str(rounds),
            "--no-gui",
            "--seed",
            str(seed),
            "--save-stats",
            str(result),
        ]
        print(
            "block={:02d} domain={} transitions={}->{} opponents={}".format(
                block_index,
                domain,
                current,
                BLOCK_TRANSITIONS,
                ",".join(opponents),
            ),
            flush=True,
        )
        subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)
        updated = shard_count(path)
        if updated <= current:
            raise RuntimeError("M2-5 collection made no progress")


def main():
    verify_preflight(require_risk=False)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for block_index, (domain, opponents) in enumerate(COLLECTION_BLOCKS):
        _run_block(block_index, domain, opponents)
    fit_risk()


if __name__ == "__main__":
    main()
