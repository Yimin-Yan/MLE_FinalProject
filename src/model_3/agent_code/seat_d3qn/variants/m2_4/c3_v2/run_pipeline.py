"""Run C3-v2 training and validation, with an optional formal test."""

import argparse
import os
import subprocess
import sys

from .config import REPOSITORY_ROOT


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rounds-per-session", type=int, default=20)
    parser.add_argument("--validation-rounds-per-seed", type=int, default=100)
    parser.add_argument("--include-test", action="store_true")
    return parser.parse_args()


def _run(module, *arguments):
    command = [sys.executable, "-m", module, *map(str, arguments)]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        command, cwd=REPOSITORY_ROOT, env=environment, check=True
    )


def main():
    args = _arguments()
    _run(
        "agent_code.seat_d3qn.variants.m2_4.c3_v2.run_training",
        "--max-rounds-per-session",
        args.max_rounds_per_session,
    )
    _run(
        "agent_code.seat_d3qn.variants.m2_4.c3_v2.select_best",
        "--rounds-per-seed",
        args.validation_rounds_per_seed,
    )
    if args.include_test:
        _run("agent_code.seat_d3qn.variants.m2_4.c3_v2.run_test")


if __name__ == "__main__":
    main()
