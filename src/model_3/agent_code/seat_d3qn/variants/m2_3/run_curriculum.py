"""Train M2-3 with the 250k interleaved combat curriculum."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import torch

from .config import (
    CURRICULUM_STAGES,
    CURRICULUM_TARGETS,
    MAX_TRAINING_TRANSITIONS,
    REPOSITORY_ROOT,
    TRAINING_RESULT_PATH,
    TRAINING_CHECKPOINT_PATH,
    TRAINING_SEED,
)


OPPONENTS = {
    "weak": ("peaceful_agent", "peaceful_agent", "coin_collector_agent"),
    "official": ("peaceful_agent", "coin_collector_agent", "rule_based_agent"),
    "rule_heavy": ("rule_based_agent", "rule_based_agent", "coin_collector_agent"),
}
SESSION_STATS_DIR = TRAINING_RESULT_PATH.parent / ".m2_3_training_sessions"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rounds-per-session", type=int, default=500)
    parser.add_argument("--scenario", default="classic")
    return parser.parse_args()


def _load_checkpoint():
    if not TRAINING_CHECKPOINT_PATH.is_file():
        return None
    try:
        return torch.load(
            TRAINING_CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(TRAINING_CHECKPOINT_PATH, map_location="cpu")


def _training_state():
    checkpoint = _load_checkpoint()
    if checkpoint is None:
        return {"transitions": 0, "rounds": 0}
    metadata = checkpoint.get("metadata", {})
    if metadata.get("variant_id") != "M2-3":
        raise ValueError("The training checkpoint does not belong to M2-3")
    return {
        "transitions": int(checkpoint["transition_count"]),
        "rounds": int(checkpoint["rounds_completed"]),
    }


def _current_block(transition_count):
    start = 0
    for target, stage in zip(CURRICULUM_TARGETS, CURRICULUM_STAGES):
        if transition_count < target:
            return start, target, stage, OPPONENTS[stage]
        start = target
    return None


def _session_rounds(state, target, maximum):
    remaining = target - state["transitions"]
    average_steps = state["transitions"] / float(max(state["rounds"], 1))
    # Early games are short, so this floor gives the first block enough room.
    average_steps = max(average_steps, 80.0)
    estimate = int(math.ceil(1.25 * remaining / average_steps)) + 3
    return min(maximum, max(1, estimate))


def _merge_training_results():
    session_files = sorted(SESSION_STATS_DIR.glob("m2_3_session_*.json"))
    if not session_files:
        raise FileNotFoundError("No M2-3 session statistics were recorded")

    combined_agents = {}
    combined_rounds = {}
    round_index = 1
    for path in session_files:
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)

        for name, statistics in result.get("by_agent", {}).items():
            target = combined_agents.setdefault(name, {})
            for key, value in statistics.items():
                if isinstance(value, (int, float)):
                    target[key] = target.get(key, 0) + value

        for round_data in result.get("by_round", {}).values():
            combined_rounds["Round {:05d}".format(round_index)] = round_data
            round_index += 1

    payload = {"by_agent": combined_agents, "by_round": combined_rounds}
    TRAINING_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = TRAINING_RESULT_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, TRAINING_RESULT_PATH)
    shutil.rmtree(SESSION_STATS_DIR)


def main():
    args = _arguments()
    if args.max_rounds_per_session <= 0:
        raise ValueError("max-rounds-per-session must be positive")

    while True:
        state = _training_state()
        if state["transitions"] >= MAX_TRAINING_TRANSITIONS:
            if SESSION_STATS_DIR.is_dir():
                _merge_training_results()
            elif not TRAINING_RESULT_PATH.is_file():
                raise FileNotFoundError("The completed run has no training statistics")
            print("M2-3 reached 250000 curriculum transitions")
            print("Training statistics: {}".format(TRAINING_RESULT_PATH))
            return

        block = _current_block(state["transitions"])
        if block is None:
            raise RuntimeError("The curriculum ended before the training budget")
        start, target, stage, opponents = block
        rounds = _session_rounds(state, target, args.max_rounds_per_session)
        SESSION_STATS_DIR.mkdir(parents=True, exist_ok=True)
        result_path = SESSION_STATS_DIR / "m2_3_session_{:05d}_{:05d}.json".format(
            state["transitions"],
            target,
        )

        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["SEAT_D3QN_VARIANT"] = "m2_3"
        environment["SEAT_D3QN_RESUME"] = "1" if state["transitions"] else "0"
        environment["SEAT_D3QN_SESSION_TRANSITION_LIMIT"] = str(target)
        environment["SEAT_D3QN_CHECKPOINT_AFTER_ROUNDS"] = str(rounds)
        environment["SEAT_M2_3_CURRICULUM_BLOCK"] = stage

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
            args.scenario,
            "--n-rounds",
            str(rounds),
            "--no-gui",
            "--seed",
            str(TRAINING_SEED + start),
            "--save-stats",
            str(result_path),
        ]
        print(
            "stage={} transitions={}->{} opponents={}".format(
                stage,
                state["transitions"],
                target,
                ",".join(opponents),
            ),
            flush=True,
        )
        subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
