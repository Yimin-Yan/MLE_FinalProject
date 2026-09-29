"""Compare C1 and C2 on disposable seeds with paired bootstrap intervals."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


AGENT_NAME = "seat_d3qn"
RULE_NAME = "rule_based_agent"
OPPONENTS = ("peaceful_agent", "coin_collector_agent", RULE_NAME)
DEFAULT_SEEDS = tuple(range(9201, 9211))
FINAL_TEST_SEEDS = frozenset(range(10001, 10031))
TRAINING_SEEDS = frozenset(range(42, 67))
VALIDATION_SEEDS = frozenset((1101, 2202, 3303, 4404, 5505, 6606))


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds-per-seed", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def _read_complete(path, rounds):
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    if int(result["by_agent"][AGENT_NAME].get("rounds", 0)) != rounds:
        raise ValueError("Incomplete diagnostic result in {}".format(path))
    return result


def _run_seed(repository_root, output_dir, branch, seed, rounds):
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
            print("{} seed {} already complete; skipping".format(branch.upper(), seed))
            return result_path, result
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
    print("{} seed {} ({} rounds)".format(branch.upper(), seed, rounds), flush=True)
    subprocess.run(command, cwd=repository_root, env=environment, check=True)
    return result_path, _read_complete(result_path, rounds)


def _rates(result, agent):
    statistics = result["by_agent"][agent]
    rounds = float(statistics["rounds"])
    return {
        "score": statistics.get("score", 0) / rounds,
        "kills": statistics.get("kills", 0) / rounds,
        "suicide": statistics.get("suicides", 0) / rounds,
    }


def _interval(values, bootstrap_samples, rng):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "ci95": [float(lower), float(upper)],
    }


def _paired_difference(left, right):
    return np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)


def _show(name, estimate, percentage=False):
    scale = 100.0 if percentage else 1.0
    suffix = "%" if percentage else ""
    print(
        "{:<29} {:>8.4f}{}  95% CI [{:.4f}{}, {:.4f}{}]".format(
            name,
            estimate["mean"] * scale,
            suffix,
            estimate["ci95"][0] * scale,
            suffix,
            estimate["ci95"][1] * scale,
            suffix,
        )
    )


def main():
    args = _arguments()
    if args.rounds_per_seed <= 0:
        raise ValueError("rounds-per-seed must be positive")
    if args.bootstrap_samples < 1000:
        raise ValueError("bootstrap-samples must be at least 1000")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Diagnostic seeds must be non-empty and unique")
    forbidden = TRAINING_SEEDS | VALIDATION_SEEDS | FINAL_TEST_SEEDS
    overlap = sorted(set(args.seeds) & forbidden)
    if overlap:
        raise ValueError("Diagnostic seeds overlap a reserved split: {}".format(overlap))

    repository_root = Path(__file__).resolve().parents[4]
    output_dir = repository_root / "results" / "m2_4" / "diagnostic" / "multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"c1": [], "c2": []}
    result_files = {"c1": [], "c2": []}
    for branch in ("c1", "c2"):
        print("\n=== {} multi-seed diagnostic ===".format(branch.upper()), flush=True)
        for seed in args.seeds:
            path, result = _run_seed(
                repository_root,
                output_dir,
                branch,
                seed,
                args.rounds_per_seed,
            )
            result_files[branch].append(str(path.relative_to(repository_root)))
            results[branch].append(result)

    branch_rates = {
        branch: {
            "seat": [_rates(result, AGENT_NAME) for result in branch_results],
            "rule": [_rates(result, RULE_NAME) for result in branch_results],
        }
        for branch, branch_results in results.items()
    }
    rng = np.random.default_rng(20260918)

    estimates = {}
    for branch in ("c1", "c2"):
        seat = branch_rates[branch]["seat"]
        rule = branch_rates[branch]["rule"]
        estimates[branch] = {
            "seat_score": _interval(
                [row["score"] for row in seat], args.bootstrap_samples, rng
            ),
            "rule_score": _interval(
                [row["score"] for row in rule], args.bootstrap_samples, rng
            ),
            "seat_minus_rule": _interval(
                _paired_difference(
                    [row["score"] for row in seat],
                    [row["score"] for row in rule],
                ),
                args.bootstrap_samples,
                rng,
            ),
            "seat_kills": _interval(
                [row["kills"] for row in seat], args.bootstrap_samples, rng
            ),
            "seat_suicide": _interval(
                [row["suicide"] for row in seat], args.bootstrap_samples, rng
            ),
        }

    c1_seat = branch_rates["c1"]["seat"]
    c2_seat = branch_rates["c2"]["seat"]
    estimates["c2_minus_c1"] = {
        metric: _interval(
            _paired_difference(
                [row[metric] for row in c2_seat],
                [row[metric] for row in c1_seat],
            ),
            args.bootstrap_samples,
            rng,
        )
        for metric in ("score", "kills", "suicide")
    }

    print("\n=== Paired bootstrap results ===")
    for branch in ("c1", "c2"):
        print("\n{}".format(branch.upper()))
        _show("seat score/game", estimates[branch]["seat_score"])
        _show("rule score/game", estimates[branch]["rule_score"])
        _show("seat - rule score/game", estimates[branch]["seat_minus_rule"])
        _show("seat kills/game", estimates[branch]["seat_kills"])
        _show("seat suicide", estimates[branch]["seat_suicide"], percentage=True)

    print("\nC2 - C1")
    _show("score/game difference", estimates["c2_minus_c1"]["score"])
    _show("kills/game difference", estimates["c2_minus_c1"]["kills"])
    _show(
        "suicide-rate difference",
        estimates["c2_minus_c1"]["suicide"],
        percentage=True,
    )
    score_ci = estimates["c2_minus_c1"]["score"]["ci95"]
    if score_ci[0] > 0.0:
        decision = "C2 has positive diagnostic evidence over C1"
    elif score_ci[1] < 0.0:
        decision = "C1 has positive diagnostic evidence over C2"
    else:
        decision = "C1 and C2 are not distinguishable on this diagnostic"
    print("Decision: {}".format(decision))

    summary = {
        "protocol": {
            "seeds": list(args.seeds),
            "rounds_per_seed": args.rounds_per_seed,
            "rounds_per_branch": len(args.seeds) * args.rounds_per_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "seed",
            "final_test_seeds_reserved": sorted(FINAL_TEST_SEEDS),
        },
        "estimates": estimates,
        "decision": decision,
        "result_files": result_files,
    }
    summary_path = output_dir / "diagnostic-ci-summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, summary_path)
    print("Summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
