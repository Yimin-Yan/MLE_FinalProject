"""Fixed settings for the final M2-5 experiment."""

from pathlib import Path
import hashlib
import json
import os


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}

VARIANT_ID = "M2-5"
ALGORITHM_ID = "frozen_c3_wasteful_death_recovery_gate"
CHECKPOINT_FORMAT_VERSION = 1
FEATURE_SCHEMA_VERSION = "2.0"
RISK_SCHEMA_VERSION = "1.0"
FEATURE_DIM = 64
SOURCE_TRANSITION_COUNT = 225_000
SOURCE_SHA256 = "96e44f8cf7644524db837a05f37289dae710128cc4c2ba0c145edb78df7ca163"

# A bomb needs four ticks to explode and the flame stays for two ticks.
LABEL_HORIZON = 6
MAX_COLLECTION_TRANSITIONS = 250_000
BLOCK_TRANSITIONS = 25_000
COLLECTION_ROUNDS_PER_SESSION = 20
MASTER_SEED = 55

# Exploration is collection-only. The submitted policy never uses it.
HAZARD_EXPLORATION_PROBABILITY = 0.10
EXPLORATION_COOLDOWN = LABEL_HORIZON

RISK_HIDDEN_DIMS = (128, 64)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1024
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 8
GRAD_CLIP_NORM = 5.0
POS_WEIGHT_CAP = 100.0
MIN_TOTAL_POSITIVES = 1_000
MIN_POSITIVES_PER_RECOVERY_ACTION = 50
MIN_POSITIVES_PER_DOMAIN = 150
# AUC checks global ranking, while AP is stricter about the rare death labels.
MIN_ROC_AUC = 0.83
MIN_AVERAGE_PRECISION = 0.40

# These two values came from the conservative pilot, not from a broad sweep.
RISK_MARGIN = 0.25
Q_GAP_CAP = 0.075
RISK_THRESHOLD_CANDIDATES = (0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975)
MIN_INTERVENTION_RATE = 0.0005
MAX_INTERVENTION_RATE = 0.0020

MODEL_ENV = "SEAT_D3QN_MODEL_PATH"
DISABLE_GATE_ENV = "SEAT_M2_5_DISABLE_GATE"
MODE_ENV = "SEAT_M2_5_MODE"
SHARD_ENV = "SEAT_M2_5_SHARD_PATH"
BLOCK_INDEX_ENV = "SEAT_M2_5_BLOCK_INDEX"
DOMAIN_ID_ENV = "SEAT_M2_5_DOMAIN_ID"
COLLECTION_SEED_ENV = "SEAT_M2_5_COLLECTION_SEED"

AGENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = AGENT_DIR.parents[3]
M2_3_MODEL_PATH = AGENT_DIR.parent / "m2_3" / "my-saved-model.pt"
C1_MODEL_PATH = AGENT_DIR.parent / "m2_4" / "c1-model.pt"
STOCHASTIC_OPPONENT_MODEL_PATH = C1_MODEL_PATH
SOURCE_MODEL_PATH = (
    REPOSITORY_ROOT
    / "experiments/seat_d3qn/m2_4/c3_v2/checkpoints/snapshots/m2_4_c3_t0225000.pt"
)
EXPERIMENT_DIR = REPOSITORY_ROOT / "experiments/seat_d3qn/m2_5"
RESULTS_DIR = REPOSITORY_ROOT / "results/m2_5"
DATASET_DIR = EXPERIMENT_DIR / "dataset"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
RISK_CHECKPOINT_PATH = CHECKPOINT_DIR / "m2_5_risk_t0250000.pt"
TRAINING_SUMMARY_PATH = EXPERIMENT_DIR / "metrics/training-summary.json"
MODEL_PATH = AGENT_DIR / "my-saved-model.pt"
BEST_MODEL_PATH = AGENT_DIR / "best-model.pt"
VALIDATION_DIR = RESULTS_DIR / "validation"

# Ten interleaved blocks give exactly 40/30/30 percent at 250k transitions.
COLLECTION_BLOCKS = (
    ("official", ("peaceful_agent", "coin_collector_agent", "rule_based_agent")),
    ("pressure", ("rule_based_agent", "rule_based_agent", "coin_collector_agent")),
    ("stochastic", ("m2_4_noisy_agent", "coin_collector_agent", "rule_based_agent")),
    ("official", ("peaceful_agent", "coin_collector_agent", "rule_based_agent")),
    ("pressure", ("rule_based_agent", "rule_based_agent", "coin_collector_agent")),
    ("stochastic", ("m2_4_noisy_agent", "coin_collector_agent", "rule_based_agent")),
    ("official", ("peaceful_agent", "coin_collector_agent", "rule_based_agent")),
    ("pressure", ("rule_based_agent", "rule_based_agent", "coin_collector_agent")),
    ("stochastic", ("m2_4_noisy_agent", "coin_collector_agent", "rule_based_agent")),
    ("official", ("peaceful_agent", "coin_collector_agent", "rule_based_agent")),
)
DOMAIN_IDS = {"official": 0, "pressure": 1, "stochastic": 2}


def experiment_config():
    return {
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "source_sha256": SOURCE_SHA256,
        "source_transition_count": SOURCE_TRANSITION_COUNT,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "risk_schema_version": RISK_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "label_horizon": LABEL_HORIZON,
        "collection_transitions": MAX_COLLECTION_TRANSITIONS,
        "block_transitions": BLOCK_TRANSITIONS,
        "collection_blocks": [
            {"domain": domain, "opponents": list(opponents)}
            for domain, opponents in COLLECTION_BLOCKS
        ],
        "stochastic_opponent": {
            "source": "M2-4-C1",
            "legal_action_noise": 0.15,
        },
        "hazard_exploration_probability": HAZARD_EXPLORATION_PROBABILITY,
        "exploration_cooldown": EXPLORATION_COOLDOWN,
        "risk_hidden_dims": list(RISK_HIDDEN_DIMS),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "positive_weight_cap": POS_WEIGHT_CAP,
        "minimum_positives": {
            "total": MIN_TOTAL_POSITIVES,
            "per_non_bomb_action": MIN_POSITIVES_PER_RECOVERY_ACTION,
            "per_domain": MIN_POSITIVES_PER_DOMAIN,
        },
        "risk_margin": RISK_MARGIN,
        "q_gap_cap": Q_GAP_CAP,
        "risk_threshold_candidates": list(RISK_THRESHOLD_CANDIDATES),
        "intervention_rate_range": [MIN_INTERVENTION_RATE, MAX_INTERVENTION_RATE],
        "master_seed": MASTER_SEED,
    }


def config_hash():
    payload = json.dumps(experiment_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_metadata():
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "source_variant": "M2-4-C3-V2",
        "source_transition_count": SOURCE_TRANSITION_COUNT,
        "source_sha256": SOURCE_SHA256,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "risk_schema_version": RISK_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "config_hash": config_hash(),
    }


def evaluation_model_path():
    override = os.environ.get(MODEL_ENV)
    return Path(override).expanduser().resolve() if override else MODEL_PATH


if len(COLLECTION_BLOCKS) * BLOCK_TRANSITIONS != MAX_COLLECTION_TRANSITIONS:
    raise ValueError("The M2-5 blocks do not add up to 250000 transitions")
