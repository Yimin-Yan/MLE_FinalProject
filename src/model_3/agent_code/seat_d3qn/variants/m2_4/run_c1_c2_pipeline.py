"""Train, validate, and compare the two controlled M2-4 branches."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rounds-per-session", type=int, default=500)
    parser.add_argument("--validation-rounds-per-seed", type=int, default=100)
    return parser.parse_args()


def _run(module, repository_root, branch=None, extra_args=()):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if branch is None:
        environment.pop("SEAT_M2_4_BRANCH", None)
    else:
        environment["SEAT_M2_4_BRANCH"] = branch
    command = [sys.executable, "-m", module, *extra_args]
    subprocess.run(command, cwd=repository_root, env=environment, check=True)


def main():
    args = _arguments()
    if args.max_rounds_per_session <= 0:
        raise ValueError("max-rounds-per-session must be positive")
    if args.validation_rounds_per_seed <= 0:
        raise ValueError("validation-rounds-per-seed must be positive")

    repository_root = Path(__file__).resolve().parents[4]
    training_args = ("--max-rounds-per-session", str(args.max_rounds_per_session))
    validation_args = (
        "--rounds-per-seed",
        str(args.validation_rounds_per_seed),
    )

    print("\n=== M2-4 C1 training ===", flush=True)
    _run(
        "agent_code.seat_d3qn.variants.m2_4.run_training",
        repository_root,
        branch="c1",
        extra_args=training_args,
    )
    print("\n=== M2-4 C1 validation ===", flush=True)
    _run(
        "agent_code.seat_d3qn.variants.m2_4.select_best",
        repository_root,
        branch="c1",
        extra_args=validation_args,
    )

    print("\n=== M2-4 C2 training ===", flush=True)
    _run(
        "agent_code.seat_d3qn.variants.m2_4.run_training",
        repository_root,
        branch="c2",
        extra_args=training_args,
    )
    print("\n=== M2-4 C2 validation ===", flush=True)
    _run(
        "agent_code.seat_d3qn.variants.m2_4.select_best",
        repository_root,
        branch="c2",
        extra_args=validation_args,
    )

    print("\n=== M2-4 C1/C2 comparison ===", flush=True)
    _run(
        "agent_code.seat_d3qn.variants.m2_4.compare_branches",
        repository_root,
    )


if __name__ == "__main__":
    main()
