"""Run the selected C1 and C2 models on one disposable benchmark."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


AGENT_NAME = "seat_d3qn"
OPPONENTS = ("peaceful_agent", "coin_collector_agent", "rule_based_agent")


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9001)
    return parser.parse_args()


def _read_complete(path, rounds):
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    if int(result["by_agent"][AGENT_NAME].get("rounds", 0)) != rounds:
        raise ValueError("Incomplete diagnostic result in {}".format(path))
    return result


def _run_branch(repository_root, output_dir, branch, rounds, seed):
    model_path = (
        repository_root
        / "agent_code"
        / "seat_d3qn"
        / "variants"
        / "m2_4"
        / "{}-model.pt".format(branch)
    )
    if not model_path.is_file():
        raise FileNotFoundError("The selected {} model is missing".format(branch.upper()))

    result_path = output_dir / "{}_seed{}_{}.json".format(branch, seed, rounds)
    if result_path.is_file():
        try:
            result = _read_complete(result_path, rounds)
            print("{} diagnostic already complete; skipping".format(branch.upper()))
            return result
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            result_path.unlink()

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "SEAT_D3QN_VARIANT": "m2_4",
            "SEAT_M2_4_BRANCH": branch,
        }
    )
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
    print("\n=== {} diagnostic ===".format(branch.upper()), flush=True)
    subprocess.run(command, cwd=repository_root, env=environment, check=True)
    return _read_complete(result_path, rounds)


def _print_scores(branch, result):
    print("\n{} scores".format(branch.upper()))
    ranking = []
    for name, statistics in result["by_agent"].items():
        rounds = int(statistics["rounds"])
        ranking.append(
            (
                statistics.get("score", 0) / rounds,
                name,
                statistics.get("kills", 0) / rounds,
                statistics.get("suicides", 0) / rounds,
            )
        )
    for score, name, kills, suicide in sorted(ranking, reverse=True):
        print(
            "{:<22} score/game={:.4f} kills/game={:.4f} suicide={:.2%}".format(
                name, score, kills, suicide
            )
        )


def main():
    args = _arguments()
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    repository_root = Path(__file__).resolve().parents[4]
    output_dir = repository_root / "results" / "m2_4" / "diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        branch: _run_branch(
            repository_root, output_dir, branch, args.rounds, args.seed
        )
        for branch in ("c1", "c2")
    }
    for branch, result in results.items():
        _print_scores(branch, result)
    print("\nDisposable results: {}".format(output_dir))


if __name__ == "__main__":
    main()
