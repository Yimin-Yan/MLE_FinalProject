"""Evaluate one C3-v2 checkpoint against its frozen C1 starting policy."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import (
    EVALUATION_MODEL_ENV,
    GATE_KILL_TOLERANCE,
    GATE_MODE,
    GATE_ROUNDS_PER_SEED,
    GATE_SCORE_TOLERANCE,
    GATE_SEEDS,
    GATE_SUICIDE_TOLERANCE,
    MODEL_SNAPSHOT_DIR,
    REPOSITORY_ROOT,
    RESULTS_DIR,
)


AGENT_NAME = "seat_d3qn"
OPPONENTS = ("peaceful_agent", "coin_collector_agent", "rule_based_agent")
GATE_DIR = RESULTS_DIR / "gates"


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_result(path, rounds):
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    statistics = result.get("by_agent", {}).get(AGENT_NAME, {})
    if int(statistics.get("rounds", 0)) != rounds:
        raise ValueError("Incomplete gate result: {}".format(path))
    return {
        "rounds": rounds,
        "score": int(statistics.get("score", 0)),
        "kills": int(statistics.get("kills", 0)),
        "suicides": int(statistics.get("suicides", 0)),
    }


def _run(snapshot, seed, rounds):
    result_dir = GATE_DIR / snapshot.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "seed{}_{}.json".format(seed, rounds)
    if result_path.is_file():
        try:
            return _read_result(result_path, rounds)
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
    return _read_result(result_path, rounds)


def _summary(snapshot, runs):
    rounds = sum(run["rounds"] for run in runs)
    return {
        "snapshot": str(snapshot.resolve()),
        "sha256": _hash(snapshot),
        "rounds": rounds,
        "score_per_round": sum(run["score"] for run in runs) / rounds,
        "kills_per_round": sum(run["kills"] for run in runs) / rounds,
        "suicide_rate": sum(run["suicides"] for run in runs) / rounds,
    }


def evaluate_gate(transition_count):
    transition_count = int(transition_count)
    baseline = MODEL_SNAPSHOT_DIR / "m2_4_c3_t0000000.pt"
    candidate = MODEL_SNAPSHOT_DIR / "m2_4_c3_t{:07d}.pt".format(
        transition_count
    )
    if not baseline.is_file() or not candidate.is_file():
        raise FileNotFoundError("Gate snapshots are incomplete")

    baseline_runs = [
        _run(baseline, seed, GATE_ROUNDS_PER_SEED) for seed in GATE_SEEDS
    ]
    candidate_runs = [
        _run(candidate, seed, GATE_ROUNDS_PER_SEED) for seed in GATE_SEEDS
    ]
    baseline_summary = _summary(baseline, baseline_runs)
    candidate_summary = _summary(candidate, candidate_runs)
    deltas = {
        "score_per_round": candidate_summary["score_per_round"]
        - baseline_summary["score_per_round"],
        "kills_per_round": candidate_summary["kills_per_round"]
        - baseline_summary["kills_per_round"],
        "suicide_rate": candidate_summary["suicide_rate"]
        - baseline_summary["suicide_rate"],
    }
    passed = (
        deltas["score_per_round"] >= GATE_SCORE_TOLERANCE
        and deltas["kills_per_round"] >= GATE_KILL_TOLERANCE
        and deltas["suicide_rate"] <= GATE_SUICIDE_TOLERANCE
    )
    payload = {
        "transition_count": transition_count,
        "passed": passed,
        "mode": GATE_MODE,
        "protocol": {
            "seeds": list(GATE_SEEDS),
            "rounds_per_seed": GATE_ROUNDS_PER_SEED,
            "opponents": list(OPPONENTS),
        },
        "thresholds": {
            "minimum_score_delta": GATE_SCORE_TOLERANCE,
            "minimum_kill_delta": GATE_KILL_TOLERANCE,
            "maximum_suicide_delta": GATE_SUICIDE_TOLERANCE,
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": deltas,
    }
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = GATE_DIR / "gate-t{:07d}.json".format(transition_count)
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, summary_path)
    print(
        "Gate t{:07d}: score {:+.4f}, kills {:+.4f}, suicide {:+.2%} -> {}".format(
            transition_count,
            deltas["score_per_round"],
            deltas["kills_per_round"],
            deltas["suicide_rate"],
            "PASS" if passed else "FLAG",
        )
    )
    return passed, summary_path
