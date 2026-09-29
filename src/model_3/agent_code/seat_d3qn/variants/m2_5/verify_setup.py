"""Print the fixed M2-5 inheritance and collection plan."""

import argparse
import json

from .checks import verify_preflight
from .config import COLLECTION_BLOCKS, experiment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", action="store_true")
    args = parser.parse_args()
    report = verify_preflight(require_risk=args.trained)
    report["collection_blocks"] = [
        {"index": index, "domain": domain, "opponents": list(opponents)}
        for index, (domain, opponents) in enumerate(COLLECTION_BLOCKS)
    ]
    report["algorithm_id"] = experiment_config()["algorithm_id"]
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
