"""Configuration shared by the M2-0 agent modules."""

from pathlib import Path
import hashlib
import json


# The order is part of the saved-model format. Do not rearrange it later.
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}

VARIANT_ID = "M2-0"
FEATURE_SCHEMA_VERSION = "1.4"
FEATURE_DIM = 64

HIDDEN_DIM = 64
HEAD_HIDDEN_DIM = 32

GAMMA = 0.99
LEARNING_RATE = 3e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
BATCH_SIZE = 128
REPLAY_CAPACITY = 100_000
WARMUP_TRANSITIONS = 5_000
MAX_TRAINING_TRANSITIONS = 500_000
TARGET_UPDATE_INTERVAL = 2_000
GRAD_CLIP_NORM = 10.0

RANDOM_ACTION_STEPS = 5_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 200_000

USE_POTENTIAL_SHAPING = True
POTENTIAL_BETA = 0.25
CRATE_POTENTIAL_WEIGHT = 0.2

REWARD_COIN = 1.0
REWARD_KILL = 5.0
# Crates help uncover coins, but they are not part of the official score.  A
# small bonus keeps that useful intermediate behaviour without dominating the
# actual objective.
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
FULL_CHECKPOINT_INTERVAL = 50_000
MODEL_SNAPSHOT_INTERVAL = 50_000
RESUME_TRAINING = False

N_STEP = 1
USE_TEMPORAL_FEATURES = False
USE_ESCAPE_MODULE = False
USE_ATTACK_FEATURES = False

# This is the official framework revision used while designing the feature contract.
ENVIRONMENT_COMMIT = "0f55c1d207e586e6fe6db95e64302ed32281e370"
CHECKPOINT_FORMAT_VERSION = 1

AGENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = AGENT_DIR.parents[3]
EXPERIMENT_DIR = REPOSITORY_ROOT / "experiments" / "seat_d3qn" / "m2_0"
MODEL_PATH = AGENT_DIR / "my-saved-model.pt"
BEST_MODEL_PATH = EXPERIMENT_DIR / "checkpoints" / "best-model.pt"
MODEL_SNAPSHOT_DIR = EXPERIMENT_DIR / "checkpoints" / "snapshots"
TRAINING_CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "training-checkpoint.pt"
METRICS_PATH = EXPERIMENT_DIR / "metrics" / "training-metrics.jsonl"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "m2_0"
EVALUATION_MODEL_ENV = "SEAT_D3QN_MODEL_PATH"


def experiment_config():
    """Return the settings that affect features, rewards, or learned weights."""
    return {
        "variant_id": VARIANT_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "relative_coord_scale": RELATIVE_COORD_SCALE,
        "manhattan_distance_scale": MANHATTAN_DISTANCE_SCALE,
        "bfs_distance_cap": BFS_DISTANCE_CAP,
        "max_game_steps": MAX_GAME_STEPS,
        "bomb_timer_scale": BOMB_TIMER_SCALE,
        "hidden_dim": HIDDEN_DIM,
        "head_hidden_dim": HEAD_HIDDEN_DIM,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "adam_betas": list(ADAM_BETAS),
        "adam_eps": ADAM_EPS,
        "batch_size": BATCH_SIZE,
        "replay_capacity": REPLAY_CAPACITY,
        "warmup_transitions": WARMUP_TRANSITIONS,
        "max_training_transitions": MAX_TRAINING_TRANSITIONS,
        "target_update_interval": TARGET_UPDATE_INTERVAL,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "random_action_steps": RANDOM_ACTION_STEPS,
        "epsilon_decay_steps": EPSILON_DECAY_STEPS,
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
        "n_step": N_STEP,
        "use_temporal_features": USE_TEMPORAL_FEATURES,
        "use_escape_module": USE_ESCAPE_MODULE,
        "use_attack_features": USE_ATTACK_FEATURES,
    }


def config_hash():
    """Make incompatible checkpoints fail early instead of failing silently."""
    payload = json.dumps(experiment_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_metadata():
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "variant_id": VARIANT_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "config_hash": config_hash(),
        "environment_commit": ENVIRONMENT_COMMIT,
        "training_seed": TRAINING_SEED,
    }
