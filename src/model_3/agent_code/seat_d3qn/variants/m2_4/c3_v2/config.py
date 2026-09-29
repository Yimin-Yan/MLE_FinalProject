"""Frozen configuration for the protected 500k C3 continuation."""

from pathlib import Path
import hashlib
import json
import os


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}

VARIANT_ID = "M2-4-C3-V2"
ALGORITHM_ID = "frozen_c1_residual_rehearsal_league_dqn"
FEATURE_SCHEMA_VERSION = "2.0"
FEATURE_DIM = 64
BASE_FEATURE_DIM = 42
CHECKPOINT_FORMAT_VERSION = 1
ENVIRONMENT_COMMIT = "0f55c1d207e586e6fe6db95e64302ed32281e370"

RESIDUAL_HIDDEN_DIM = 64
RESIDUAL_HEAD_DIM = 32
TRANSFER_TOLERANCE = 1e-6

GAMMA = 0.99
N_STEP = 3
LEARNING_RATE = 2.5e-5
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
BATCH_SIZE = 32
LEAGUE_BATCH_SIZE = 24
REHEARSAL_BATCH_SIZE = 8
LEAGUE_TD_WEIGHT = 0.75
REHEARSAL_TD_WEIGHT = 0.25
REPLAY_CAPACITY = 100_000
WARMUP_TRANSITIONS = 5_000
MAX_TRAINING_TRANSITIONS = 500_000
SESSION_TRANSITION_LIMIT = min(
    MAX_TRAINING_TRANSITIONS,
    int(os.environ.get("SEAT_D3QN_SESSION_TRANSITION_LIMIT", MAX_TRAINING_TRANSITIONS)),
)
REPLAY_PERIOD = 4
TARGET_UPDATE_INTERVAL = 2_000
GRAD_CLIP_NORM = 10.0

EPSILON_START = 0.05
EPSILON_END = 0.01
EPSILON_DECAY_TRANSITIONS = 100_000

LAP_ALPHA = 0.6
LAP_PRIORITY_FLOOR = 1.0
DISTILL_TEMPERATURE = 1.0
DISTILL_TARGET_RATIO = 0.10
DISTILL_LAMBDA_MIN = 0.10
DISTILL_LAMBDA_MAX = 10.0

REWARD_COIN = 1.0
REWARD_KILL = 5.0
REWARD_CRATE = 0.05
PENALTY_KILLED = -2.0
PENALTY_SUICIDE = -3.0
PENALTY_STATIC_INVALID = -0.2
PENALTY_RACE_INVALID = -0.02
FAIL_ON_STATIC_INVALID = True
USE_POTENTIAL_SHAPING = True
POTENTIAL_BETA = 0.25

MASTER_TRAINING_SEED = 42
METRIC_LOG_INTERVAL = 1_000
MODEL_SNAPSHOT_INTERVAL = 25_000
FULL_CHECKPOINT_INTERVAL = 10_000
RESUME_TRAINING = os.environ.get("SEAT_D3QN_RESUME", "0") == "1"
CHECKPOINT_AFTER_SESSION_ROUNDS = int(
    os.environ.get("SEAT_D3QN_CHECKPOINT_AFTER_ROUNDS", "0")
)

STAGE_SIZE = 50_000
GATE_INTERVAL = 25_000
STAGE_TARGETS = tuple(range(STAGE_SIZE, MAX_TRAINING_TRANSITIONS + 1, STAGE_SIZE))
STAGE_NAMES = (
    "official",
    "safety_resource",
    "rule_heavy",
    "strong_cross",
    "official",
    "noisy_robustness",
    "rule_heavy",
    "safety_resource",
    "strong_cross",
    "official",
)
GATE_TARGETS = tuple(
    range(GATE_INTERVAL, MAX_TRAINING_TRANSITIONS + 1, GATE_INTERVAL)
)
STAGE_OPPONENTS = {
    "official": ("peaceful_agent", "coin_collector_agent", "rule_based_agent"),
    "safety_resource": (
        "rule_based_agent",
        "m2_1_frozen_agent",
        "coin_collector_agent",
    ),
    "rule_heavy": (
        "rule_based_agent",
        "rule_based_agent",
        "m2_4_frozen_agent",
    ),
    "strong_cross": (
        "m2_4_frozen_agent",
        "m2_2_frozen_agent",
        "m2_3_frozen_agent",
    ),
    "noisy_robustness": (
        "rule_based_agent",
        "m2_4_noisy_agent",
        "coin_collector_agent",
    ),
}

GATE_SCORE_TOLERANCE = -0.10
GATE_KILL_TOLERANCE = -0.02
GATE_SUICIDE_TOLERANCE = 0.05
GATE_MODE = "record_only"
GATE_SEEDS = (7_201, 7_202, 7_203, 7_204, 7_205)
GATE_ROUNDS_PER_SEED = 30

if LEAGUE_BATCH_SIZE + REHEARSAL_BATCH_SIZE != BATCH_SIZE:
    raise ValueError("C3-v2 batch split must add up to 32")
if not 0 < SESSION_TRANSITION_LIMIT <= MAX_TRAINING_TRANSITIONS:
    raise ValueError("C3-v2 session target is outside the 500k budget")
if STAGE_TARGETS[-1] != MAX_TRAINING_TRANSITIONS:
    raise ValueError("C3-v2 stages must end at 500000 transitions")
if len(STAGE_TARGETS) != len(STAGE_NAMES):
    raise ValueError("Every C3-v2 stage needs one transition target")

AGENT_DIR = Path(__file__).resolve().parent
M2_4_DIR = AGENT_DIR.parent
REPOSITORY_ROOT = AGENT_DIR.parents[4]
M2_3_MODEL_PATH = M2_4_DIR.parent / "m2_3" / "my-saved-model.pt"
C1_MODEL_PATH = M2_4_DIR / "c1-model.pt"
C1_TRAINING_CHECKPOINT_PATH = (
    REPOSITORY_ROOT
    / "experiments/seat_d3qn/m2_4/c1/checkpoints/training-checkpoint.pt"
)
EXPERIMENT_DIR = REPOSITORY_ROOT / "experiments/seat_d3qn/m2_4/c3_v2"
RESULTS_DIR = REPOSITORY_ROOT / "results/m2_4/c3_v2"
MODEL_PATH = AGENT_DIR / "my-saved-model.pt"
BEST_MODEL_PATH = AGENT_DIR / "best-model.pt"
MODEL_SNAPSHOT_DIR = EXPERIMENT_DIR / "checkpoints/snapshots"
TRAINING_CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints/training-checkpoint.pt"
METRICS_PATH = EXPERIMENT_DIR / "metrics/training-metrics.jsonl"
TRAINING_RESULT_PATH = RESULTS_DIR / "m2_4_c3_training.json"
EVALUATION_MODEL_ENV = "SEAT_D3QN_MODEL_PATH"


def epsilon_for_transition(transition_count):
    progress = min(
        max(float(transition_count) / EPSILON_DECAY_TRANSITIONS, 0.0), 1.0
    )
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def stage_for_transition(transition_count):
    step = int(transition_count)
    start = 0
    for target, name in zip(STAGE_TARGETS, STAGE_NAMES):
        if step < target:
            return start, target, name
        start = target
    return None


def experiment_config():
    return {
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "gamma": GAMMA,
        "n_step": N_STEP,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "league_batch_size": LEAGUE_BATCH_SIZE,
        "rehearsal_batch_size": REHEARSAL_BATCH_SIZE,
        "league_td_weight": LEAGUE_TD_WEIGHT,
        "rehearsal_td_weight": REHEARSAL_TD_WEIGHT,
        "replay_capacity": REPLAY_CAPACITY,
        "warmup_transitions": WARMUP_TRANSITIONS,
        "max_training_transitions": MAX_TRAINING_TRANSITIONS,
        "replay_period": REPLAY_PERIOD,
        "target_update_interval": TARGET_UPDATE_INTERVAL,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay_transitions": EPSILON_DECAY_TRANSITIONS,
        "lap_alpha": LAP_ALPHA,
        "lap_priority_floor": LAP_PRIORITY_FLOOR,
        "distill_temperature": DISTILL_TEMPERATURE,
        "distill_target_ratio": DISTILL_TARGET_RATIO,
        "distill_lambda_min": DISTILL_LAMBDA_MIN,
        "distill_lambda_max": DISTILL_LAMBDA_MAX,
        "stage_targets": list(STAGE_TARGETS),
        "stage_names": list(STAGE_NAMES),
        "stage_opponents": {
            name: list(opponents) for name, opponents in STAGE_OPPONENTS.items()
        },
        "gate_targets": list(GATE_TARGETS),
        "gate_mode": GATE_MODE,
        "master_training_seed": MASTER_TRAINING_SEED,
        "source_variant": "M2-4-C1",
    }


def config_hash():
    payload = json.dumps(experiment_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_metadata(m2_3_sha256, c1_sha256):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "config_hash": config_hash(),
        "source_variant": "M2-4-C1",
        "m2_3_source_sha256": m2_3_sha256,
        "c1_source_sha256": c1_sha256,
        "environment_commit": ENVIRONMENT_COMMIT,
    }
