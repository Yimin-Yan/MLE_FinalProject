"""Configuration for the interleaved combat-curriculum agent."""

from pathlib import Path
import hashlib
import json
import os


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}

VARIANT_ID = "M2-3"
ALGORITHM_ID = "warm_started_temporal_escape_interleaved_combat_curriculum"
FEATURE_SCHEMA_VERSION = "2.0"
FEATURE_DIM = 64
BASE_FEATURE_DIM = 42

HIDDEN_DIM = 64
HEAD_HIDDEN_DIM = 32
ESCAPE_HIDDEN_DIM = 32

GAMMA = 0.99
N_STEP = 3
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
BATCH_SIZE = 32
REPLAY_CAPACITY = 100_000
WARMUP_TRANSITIONS = 5_000
MAX_TRAINING_TRANSITIONS = 250_000
SESSION_TRANSITION_LIMIT = min(
    MAX_TRAINING_TRANSITIONS,
    int(os.environ.get("SEAT_D3QN_SESSION_TRANSITION_LIMIT", MAX_TRAINING_TRANSITIONS)),
)
REPLAY_PERIOD = 4
TARGET_UPDATE_INTERVAL = 2_000
GRAD_CLIP_NORM = 10.0

EPSILON_START = 0.10
EPSILON_END = 0.02
EPSILON_DECAY_STEPS = 100_000

LAP_ALPHA = 0.6
LAP_PRIORITY_FLOOR = 1.0
ESCAPE_IS_BETA = 1.0

TEMPORAL_HORIZON = 8
BOMB_TIMER = 4
BOMB_POWER = 3
EXPLOSION_DANGER_STEPS = 2
DEAD_END_DEPTH_CAP = 8

ESCAPE_LOSS_WEIGHT = 0.05
ESCAPE_LOSS_WARMUP_UPDATES = 10_000
TRANSFER_TOLERANCE = 1e-6

USE_POTENTIAL_SHAPING = True
POTENTIAL_BETA = 0.25
CRATE_POTENTIAL_WEIGHT = 0.2

REWARD_COIN = 1.0
REWARD_KILL = 5.0
REWARD_CRATE = 0.05
PENALTY_KILLED = -2.0
PENALTY_SUICIDE = -3.0
PENALTY_STATIC_INVALID = -0.2
PENALTY_RACE_INVALID = -0.02
FAIL_ON_STATIC_INVALID = True

TRAINING_SEED = 42
MAX_GAME_STEPS = 400.0
BOMB_TIMER_SCALE = 4.0
RELATIVE_COORD_SCALE = 16.0
MANHATTAN_DISTANCE_SCALE = 32.0
BFS_DISTANCE_CAP = 64.0

METRIC_LOG_INTERVAL = 1_000
FULL_CHECKPOINT_INTERVAL = 10_000
MODEL_SNAPSHOT_INTERVAL = 25_000
RESUME_TRAINING = os.environ.get("SEAT_D3QN_RESUME", "0") == "1"
CHECKPOINT_AFTER_SESSION_ROUNDS = int(
    os.environ.get("SEAT_D3QN_CHECKPOINT_AFTER_ROUNDS", "0")
)
if not 0 < SESSION_TRANSITION_LIMIT <= MAX_TRAINING_TRANSITIONS:
    raise ValueError("SEAT_D3QN_SESSION_TRANSITION_LIMIT is outside the M2-3 budget")
if CHECKPOINT_AFTER_SESSION_ROUNDS < 0:
    raise ValueError("SEAT_D3QN_CHECKPOINT_AFTER_ROUNDS cannot be negative")

ENVIRONMENT_COMMIT = "0f55c1d207e586e6fe6db95e64302ed32281e370"
CHECKPOINT_FORMAT_VERSION = 3

AGENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = AGENT_DIR.parents[3]
M2_2_MODEL_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "seat_d3qn"
    / "m2_2"
    / "checkpoints"
    / "snapshots"
    / "m2_2_t0250000.pt"
)
EXPERIMENT_DIR = REPOSITORY_ROOT / "experiments" / "seat_d3qn" / "m2_3"
MODEL_PATH = AGENT_DIR / "my-saved-model.pt"
BEST_MODEL_PATH = AGENT_DIR / "best-model.pt"
MODEL_SNAPSHOT_DIR = EXPERIMENT_DIR / "checkpoints" / "snapshots"
TRAINING_CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "training-checkpoint.pt"
METRICS_PATH = EXPERIMENT_DIR / "metrics" / "training-metrics.jsonl"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "m2_3"
TRAINING_RESULT_PATH = RESULTS_DIR / "m2_3_training.json"
EVALUATION_MODEL_ENV = "SEAT_D3QN_MODEL_PATH"

# Each block contributes 10k new transitions. Interleaving the three opponent
# sets avoids long stretches where the policy only sees one style of play.
CURRICULUM_TARGETS = tuple(range(10_000, 250_001, 10_000))
CURRICULUM_STAGES = (
    "official",
    "weak", "official", "weak", "rule_heavy",
    "official", "rule_heavy", "official", "rule_heavy", "weak",
    "official", "rule_heavy", "official", "rule_heavy", "weak",
    "official", "rule_heavy", "official", "rule_heavy", "weak",
    "official", "rule_heavy", "official", "official", "official",
)


def experiment_config():
    """Return settings that must match when a checkpoint is resumed."""
    return {
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "base_feature_dim": BASE_FEATURE_DIM,
        "actions": list(ACTIONS),
        "hidden_dim": HIDDEN_DIM,
        "head_hidden_dim": HEAD_HIDDEN_DIM,
        "escape_hidden_dim": ESCAPE_HIDDEN_DIM,
        "gamma": GAMMA,
        "n_step": N_STEP,
        "learning_rate": LEARNING_RATE,
        "adam_betas": list(ADAM_BETAS),
        "adam_eps": ADAM_EPS,
        "batch_size": BATCH_SIZE,
        "replay_capacity": REPLAY_CAPACITY,
        "warmup_transitions": WARMUP_TRANSITIONS,
        "max_training_transitions": MAX_TRAINING_TRANSITIONS,
        "replay_period": REPLAY_PERIOD,
        "target_update_interval": TARGET_UPDATE_INTERVAL,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay_steps": EPSILON_DECAY_STEPS,
        "lap_alpha": LAP_ALPHA,
        "lap_priority_floor": LAP_PRIORITY_FLOOR,
        "escape_is_beta": ESCAPE_IS_BETA,
        "temporal_horizon": TEMPORAL_HORIZON,
        "bomb_timer": BOMB_TIMER,
        "bomb_power": BOMB_POWER,
        "explosion_danger_steps": EXPLOSION_DANGER_STEPS,
        "dead_end_depth_cap": DEAD_END_DEPTH_CAP,
        "escape_loss_weight": ESCAPE_LOSS_WEIGHT,
        "escape_loss_warmup_updates": ESCAPE_LOSS_WARMUP_UPDATES,
        "use_potential_shaping": USE_POTENTIAL_SHAPING,
        "potential_beta": POTENTIAL_BETA,
        "crate_potential_weight": CRATE_POTENTIAL_WEIGHT,
        "reward_coin": REWARD_COIN,
        "reward_kill": REWARD_KILL,
        "reward_crate": REWARD_CRATE,
        "penalty_killed": PENALTY_KILLED,
        "penalty_suicide": PENALTY_SUICIDE,
        "penalty_static_invalid": PENALTY_STATIC_INVALID,
        "penalty_race_invalid": PENALTY_RACE_INVALID,
        "fail_on_static_invalid": FAIL_ON_STATIC_INVALID,
        "training_seed": TRAINING_SEED,
        "source_variant": "M2-2",
        "source_transition_count": 250_000,
        "curriculum_targets": list(CURRICULUM_TARGETS),
        "curriculum_stages": list(CURRICULUM_STAGES),
    }


def config_hash():
    payload = json.dumps(experiment_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_metadata():
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "config_hash": config_hash(),
        "environment_commit": ENVIRONMENT_COMMIT,
        "training_seed": TRAINING_SEED,
        "source_variant": "M2-2",
        "source_transition_count": 250_000,
        "curriculum_transition_count": 250_000,
    }
