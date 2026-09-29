"""Report the independently selected C1 and C2 checkpoints."""

import json
from pathlib import Path

from .config import REPOSITORY_ROOT


def _load(branch):
    path = (
        REPOSITORY_ROOT
        / "results"
        / "m2_4"
        / branch
        / "validation"
        / "selection-summary.json"
    )
    if not path.is_file():
        raise FileNotFoundError("Missing {} selection summary: {}".format(branch, path))
    with path.open("r", encoding="utf-8") as file:
        return path, json.load(file)


def main():
    rows = []
    for branch in ("c1", "c2"):
        path, summary = _load(branch)
        best = summary["best_snapshot"]
        rows.append(
            {
                "branch": branch,
                "score": float(best["mean_official_score"]),
                "kills": float(best["kills_per_round"]),
                "suicide": float(best["suicide_rate"]),
                "transition": int(best["transition_count"]),
                "summary": str(path.relative_to(REPOSITORY_ROOT)),
            }
        )

    for row in rows:
        print(
            "{} score={:.4f} kills/game={:.4f} suicide={:.2%} checkpoint=t{:07d}".format(
                row["branch"].upper(),
                row["score"],
                row["kills"],
                row["suicide"],
                row["transition"],
            )
        )
    print("C3 source: selected C1 checkpoint")


if __name__ == "__main__":
    main()
