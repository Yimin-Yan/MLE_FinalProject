"""Train all C3-v2 stages and record a diagnostic gate every 25k."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

from .checks import load_torch, verify_all
from .config import (
    C1_MODEL_PATH,
    GATE_TARGETS,
    MASTER_TRAINING_SEED,
    MAX_TRAINING_TRANSITIONS,
    M2_4_DIR,
    REPOSITORY_ROOT,
    SESSION_TRANSITION_LIMIT,
    STAGE_OPPONENTS,
    TRAINING_CHECKPOINT_PATH,
    TRAINING_RESULT_PATH,
    stage_for_transition,
)
from .gate import evaluate_gate


SESSION_STATS_DIR = TRAINING_RESULT_PATH.parent / ".training_sessions"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rounds-per-session", type=int, default=20)
    return parser.parse_args()


def _state():
    if not TRAINING_CHECKPOINT_PATH.is_file():
        return {"transitions": 0, "rounds": 0, "decisions": 0}
    checkpoint = load_torch(TRAINING_CHECKPOINT_PATH, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("variant_id") != "M2-4-C3-V2":
        raise ValueError("The training checkpoint does not belong to C3-v2")
    transitions = int(checkpoint["transition_count"])
    if not 0 <= transitions <= MAX_TRAINING_TRANSITIONS:
        raise ValueError("C3-v2 checkpoint is outside its 500k budget")
    return {
        "transitions": transitions,
        "rounds": int(checkpoint["rounds_completed"]),
        "decisions": int(checkpoint["decision_step"]),
    }


def _next_gate(transition_count):
    return next((target for target in GATE_TARGETS if target > transition_count), None)


def _session_rounds(state, target, maximum):
    remaining = target - state["transitions"]
    if state["rounds"]:
        average = state["transitions"] / float(state["rounds"])
    else:
        average = 80.0
    average = max(average, 20.0)
    estimate = int(math.ceil(1.05 * remaining / average)) + 1
    return min(maximum, max(1, estimate))


def _session_index():
    return len(list(SESSION_STATS_DIR.glob("session_*.json")))


def _session_seed(index):
    # A single master seed expands into deterministic non-repeating sessions.
    return MASTER_TRAINING_SEED * 100_000 + int(index)


def _merge_results():
    paths = sorted(SESSION_STATS_DIR.glob("session_*.json"))
    if not paths:
        raise FileNotFoundError("No C3-v2 training statistics were recorded")
    agents, rounds, round_index = {}, {}, 1
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        for agent, statistics in result.get("by_agent", {}).items():
            target = agents.setdefault(agent, {})
            for name, value in statistics.items():
                if isinstance(value, (int, float)):
                    target[name] = target.get(name, 0) + value
        for data in result.get("by_round", {}).values():
            rounds["Round {:05d}".format(round_index)] = data
            round_index += 1
    TRAINING_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = TRAINING_RESULT_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump({"by_agent": agents, "by_round": rounds}, file, indent=2)
        file.write("\n")
    os.replace(temporary, TRAINING_RESULT_PATH)
    shutil.rmtree(SESSION_STATS_DIR)


def _preflight():
    verification = verify_all()
    print("C3-v2 source C1: {}".format(verification["sources"]["c1_sha256"]))
    print("C3-v2 stage counts: {}".format(verification["stage_counts"]))


def main():
    args = _arguments()
    if args.max_rounds_per_session < 1:
        raise ValueError("max-rounds-per-session must be positive")
    _preflight()

    while True:
        state = _state()
        if state["transitions"] == SESSION_TRANSITION_LIMIT:
            if SESSION_TRANSITION_LIMIT == MAX_TRAINING_TRANSITIONS:
                final_gate = (
                    TRAINING_RESULT_PATH.parent
                    / "gates/gate-t{:07d}.json".format(state["transitions"])
                )
                final_passed = None
                if final_gate.is_file():
                    with final_gate.open("r", encoding="utf-8") as file:
                        final_passed = bool(json.load(file)["passed"])
                if final_passed is None:
                    final_passed, final_gate = evaluate_gate(state["transitions"])
                if not final_passed:
                    print(
                        "C3-v2 final gate was flagged but training remains complete: {}".format(
                            final_gate
                        )
                    )
                if SESSION_STATS_DIR.is_dir():
                    _merge_results()
                elif not TRAINING_RESULT_PATH.is_file():
                    raise FileNotFoundError("Completed C3-v2 has no training result")
                print("M2-4 C3-v2 reached exactly 500000 new transitions")
            else:
                print("C3-v2 reached the requested session transition limit")
            return

        if state["transitions"] in GATE_TARGETS:
            gate_path = (
                TRAINING_RESULT_PATH.parent
                / "gates/gate-t{:07d}.json".format(state["transitions"])
            )
            gate_passed = None
            if gate_path.is_file():
                with gate_path.open("r", encoding="utf-8") as file:
                    gate_passed = bool(json.load(file)["passed"])
            if gate_passed is None:
                gate_passed, gate_path = evaluate_gate(state["transitions"])
            if not gate_passed:
                print(
                    "C3-v2 gate was flagged; continuing to preserve the full 500k run: {}".format(
                        gate_path
                    )
                )

        stage = stage_for_transition(state["transitions"])
        if stage is None:
            raise RuntimeError("C3-v2 schedule ended before the training budget")
        stage_start, stage_target, stage_name = stage
        gate_target = _next_gate(state["transitions"])
        target = min(stage_target, gate_target, SESSION_TRANSITION_LIMIT)
        rounds = _session_rounds(state, target, args.max_rounds_per_session)
        SESSION_STATS_DIR.mkdir(parents=True, exist_ok=True)
        index = _session_index()
        result_path = SESSION_STATS_DIR / (
            "session_{:04d}_{:07d}_{:07d}.json".format(
                index, state["transitions"], target
            )
        )

        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "SEAT_D3QN_VARIANT": "m2_4_c3_v2",
                "SEAT_D3QN_RESUME": "1" if state["transitions"] else "0",
                "SEAT_D3QN_SESSION_TRANSITION_LIMIT": str(target),
                "SEAT_D3QN_CHECKPOINT_AFTER_ROUNDS": str(rounds),
                "SEAT_M2_4_CURRICULUM_BLOCK": stage_name,
                "SEAT_M2_4_OPPONENT_MODEL_PATH": str(C1_MODEL_PATH.resolve()),
                "SEAT_M2_4_NOISE_PROBABILITY": "0.10",
                "SEAT_M2_4_NOISY_SEED": str(
                    _session_seed(index) + 1_000_000
                ),
            }
        )
        environment.pop("SEAT_D3QN_MODEL_PATH", None)
        opponents = STAGE_OPPONENTS[stage_name]
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
            str(_session_seed(index)),
            "--save-stats",
            str(result_path),
        ]
        print(
            "stage={} transitions={}->{} rounds={} opponents={}".format(
                stage_name,
                state["transitions"],
                target,
                rounds,
                ",".join(opponents),
            ),
            flush=True,
        )
        subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
