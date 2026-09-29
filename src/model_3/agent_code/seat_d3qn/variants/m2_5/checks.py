"""Fast checks before collection or evaluation."""

from collections import Counter

from .config import (
    BLOCK_TRANSITIONS,
    CHECKPOINT_DIR,
    COLLECTION_BLOCKS,
    MAX_COLLECTION_TRANSITIONS,
    REPOSITORY_ROOT,
    RISK_CHECKPOINT_PATH,
    SOURCE_SHA256,
    checkpoint_metadata,
)
from .source import load_torch, sha256, verify_source_files


REQUIRED_OPPONENTS = (
    "peaceful_agent",
    "coin_collector_agent",
    "rule_based_agent",
    "m2_4_noisy_agent",
)


def verify_preflight(require_risk=False):
    source = verify_source_files()
    if sha256(source_path()) != SOURCE_SHA256:
        raise ValueError("The M2-5 frozen source checksum is wrong")
    for name in REQUIRED_OPPONENTS:
        if not (REPOSITORY_ROOT / "agent_code" / name).is_dir():
            raise FileNotFoundError("M2-5 opponent is missing: " + name)

    counts = Counter(domain for domain, _ in COLLECTION_BLOCKS)
    if counts != Counter({"official": 4, "pressure": 3, "stochastic": 3}):
        raise ValueError("M2-5 collection must use a 4/3/3 block split")
    if len(COLLECTION_BLOCKS) * BLOCK_TRANSITIONS != MAX_COLLECTION_TRANSITIONS:
        raise ValueError("M2-5 collection budget is not exactly 250000")

    output = {
        "source_transition_count": int(source["transition_count"]),
        "source_sha256": SOURCE_SHA256,
        "training_budget": MAX_COLLECTION_TRANSITIONS,
        "domain_blocks": dict(counts),
    }
    if require_risk:
        if not RISK_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError("Train the M2-5 risk model before validation")
        risk = load_torch(RISK_CHECKPOINT_PATH, "cpu", weights_only=False)
        expected = checkpoint_metadata()
        metadata = risk.get("metadata", {})
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError("M2-5 risk checkpoint mismatch for {}".format(key))
        if int(risk.get("collection_transition_count", -1)) != MAX_COLLECTION_TRANSITIONS:
            raise ValueError("M2-5 risk model was not fitted on 250000 transitions")
        output["risk_checkpoint"] = str(RISK_CHECKPOINT_PATH.resolve())
        output["risk_threshold"] = float(risk["risk_threshold"])
    return output


def source_path():
    from .config import SOURCE_MODEL_PATH

    return SOURCE_MODEL_PATH
