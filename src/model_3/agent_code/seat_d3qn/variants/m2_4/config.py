"""Configuration shared by the three M2-4 experiment branches."""

from pathlib import Path
import hashlib
import json
import os


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}

BRANCH = os.environ.get("SEAT_M2_4_BRANCH", "c2").strip().lower()
if BRANCH not in {"c1", "c2", "c3"}:
    raise ValueError("SEAT_M2_4_BRANCH must be 'c1', 'c2', or 'c3'")

VARIANT_ID = "M2-4-{}".format(BRANCH.upper())
if BRANCH == "c3":
    ALGORITHM_ID = "c1_warm_started_diverse_league_dqn"
else:
    ALGORITHM_ID = "frozen_m2_3_residual_q_selective_rule_teacher"
FEATURE_SCHEMA_VERSION = "2.0"
FEATURE_DIM = 64
BASE_FEATURE_DIM = 42

RESIDUAL_HIDDEN_DIM = 64
RESIDUAL_HEAD_DIM = 32
TRANSFER_TOLERANCE = 1e-6

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

ANCHOR_WEIGHT = 1e-3
C3_KEEP_WEIGHT = 1e-3
C3_SCORE_EXCEPTION = 0.30
C3_SUICIDE_TOLERANCE = 0.05
C3_SCORE_TIE_WINDOW = 0.10
TEACHER_ENABLED = BRANCH == "c2"
TEACHER_QUERY_COUNT = 5
TEACHER_MIN_VOTES = 4
TEACHER_CONFIDENCE_THRESHOLD = 0.8
TEACHER_MARGIN = 0.4
TEACHER_BATCH_SIZE = 8
TEACHER_BUFFER_CAPACITY = 50_000
TEACHER_SEED = 24_042
TEACHER_ENEMY_DISTANCE = 4
BOMB_POWER = 3

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

TRAINING_SEED = 42
METRIC_LOG_INTERVAL = 1_000
FULL_CHECKPOINT_INTERVAL = 10_000
MODEL_SNAPSHOT_INTERVAL = 25_000
RESUME_TRAINING = os.environ.get("SEAT_D3QN_RESUME", "0") == "1"
CHECKPOINT_AFTER_SESSION_ROUNDS = int(
    os.environ.get("SEAT_D3QN_CHECKPOINT_AFTER_ROUNDS", "0")
)
if not 0 < SESSION_TRANSITION_LIMIT <= MAX_TRAINING_TRANSITIONS:
    raise ValueError("SEAT_D3QN_SESSION_TRANSITION_LIMIT is outside the M2-4 budget")
if CHECKPOINT_AFTER_SESSION_ROUNDS < 0:
    raise ValueError("SEAT_D3QN_CHECKPOINT_AFTER_ROUNDS cannot be negative")

CHECKPOINT_FORMAT_VERSION = 1
ENVIRONMENT_COMMIT = "0f55c1d207e586e6fe6db95e64302ed32281e370"

AGENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = AGENT_DIR.parents[3]
M2_3_MODEL_PATH = AGENT_DIR.parent / "m2_3" / "my-saved-model.pt"
C1_MODEL_PATH = AGENT_DIR / "c1-model.pt"
C2_MODEL_PATH = AGENT_DIR / "c2-model.pt"
PROBE_ROOT = os.environ.get("SEAT_M2_4_PROBE_ROOT")
if PROBE_ROOT:
    probe_root = Path(PROBE_ROOT).expanduser().resolve()
    EXPERIMENT_DIR = probe_root / "experiment"
    RESULTS_DIR = probe_root / "results"
    MODEL_PATH = probe_root / "{}-model.pt".format(BRANCH)
    BEST_MODEL_PATH = probe_root / "{}-best-model.pt".format(BRANCH)
else:
    EXPERIMENT_DIR = REPOSITORY_ROOT / "experiments" / "seat_d3qn" / "m2_4" / BRANCH
    RESULTS_DIR = REPOSITORY_ROOT / "results" / "m2_4" / BRANCH
    MODEL_PATH = AGENT_DIR / "{}-model.pt".format(BRANCH)
    BEST_MODEL_PATH = AGENT_DIR / "{}-best-model.pt".format(BRANCH)
MODEL_SNAPSHOT_DIR = EXPERIMENT_DIR / "checkpoints" / "snapshots"
TRAINING_CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "training-checkpoint.pt"
METRICS_PATH = EXPERIMENT_DIR / "metrics" / "training-metrics.jsonl"
TRAINING_RESULT_PATH = RESULTS_DIR / "m2_4_{}_training.json".format(BRANCH)
EVALUATION_MODEL_ENV = "SEAT_D3QN_MODEL_PATH"

# C1 and C2 deliberately use the same opponent schedule. That keeps the
# teacher loss as the only experimental difference between the two branches.
CURRICULUM_TARGETS = tuple(range(10_000, 250_001, 10_000))
CURRICULUM_STAGES = (
    "official",
    "weak", "official", "weak", "rule_heavy",
    "official", "rule_heavy", "official", "rule_heavy", "weak",
    "official", "rule_heavy", "official", "rule_heavy", "weak",
    "official", "rule_heavy", "official", "rule_heavy", "weak",
    "official", "rule_heavy", "official", "official", "official",
)

# Twenty equal blocks reproduce the frozen C3 league percentages exactly.
C3_LEAGUE_TARGETS = tuple(range(12_500, 250_001, 12_500))
C3_LEAGUE_STAGES = (
    "hard_rule",
    "self_history",
    "attack",
    "non_rule_strong",
    "hard_rule",
    "conservative",
    "self_history",
    "resource",
    "attack",
    "noisy",
    "hard_rule",
    "non_rule_strong",
    "self_history",
    "conservative",
    "attack",
    "resource",
    "hard_rule",
    "noisy",
    "non_rule_strong",
    "mixed",
)
C3_LEAGUE_OPPONENTS = {
    "hard_rule": ("rule_based_agent", "rule_based_agent", "m2_4_frozen_agent"),
    "self_history": (
        "rule_based_agent",
        "m2_4_frozen_agent",
        "m2_4_historical_agent",
    ),
    "attack": ("rule_based_agent", "m2_2_frozen_agent", "m2_4_frozen_agent"),
    "non_rule_strong": (
        "m2_4_frozen_agent",
        "m2_2_frozen_agent",
        "m2_3_frozen_agent",
    ),
    "conservative": (
        "rule_based_agent",
        "m2_1_frozen_agent",
        "m2_4_historical_agent",
    ),
    "resource": (
        "m2_4_frozen_agent",
        "coin_collector_agent",
        "m2_3_frozen_agent",
    ),
    "noisy": ("m2_4_frozen_agent", "m2_4_noisy_agent", "m2_3_frozen_agent"),
    "mixed": (
        "m2_4_historical_agent",
        "m2_4_noisy_agent",
        "coin_collector_agent",
    ),
}


def teacher_weight(transition_count):
    """Return the frozen C2 teacher schedule for a transition count."""
    if not TEACHER_ENABLED:
        return 0.0
    step = float(max(0, transition_count))
    if step < 50_000:
        return 0.10 - 0.05 * (step / 50_000.0)
    if step < 125_000:
        return 0.05 - 0.03 * ((step - 50_000.0) / 75_000.0)
    if step < 200_000:
        return 0.02 * (1.0 - (step - 125_000.0) / 75_000.0)
    return 0.0


def learning_rate(transition_count):
    """Keep C1/C2 fixed and lower the rate during the C3 continuation."""
    if BRANCH != "c3":
        return LEARNING_RATE
    if int(transition_count) < 150_000:
        return 5e-5
    return 2.5e-5


def experiment_config():
    config = {
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "branch": BRANCH,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "residual_hidden_dim": RESIDUAL_HIDDEN_DIM,
        "residual_head_dim": RESIDUAL_HEAD_DIM,
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
        "anchor_weight": ANCHOR_WEIGHT,
        "teacher_enabled": TEACHER_ENABLED,
        "teacher_query_count": TEACHER_QUERY_COUNT,
        "teacher_min_votes": TEACHER_MIN_VOTES,
        "teacher_confidence_threshold": TEACHER_CONFIDENCE_THRESHOLD,
        "teacher_margin": TEACHER_MARGIN,
        "teacher_batch_size": TEACHER_BATCH_SIZE,
        "teacher_buffer_capacity": TEACHER_BUFFER_CAPACITY,
        "teacher_seed": TEACHER_SEED,
        "teacher_enemy_distance": TEACHER_ENEMY_DISTANCE,
        "bomb_power": BOMB_POWER,
        "reward_coin": REWARD_COIN,
        "reward_kill": REWARD_KILL,
        "reward_crate": REWARD_CRATE,
        "penalty_killed": PENALTY_KILLED,
        "penalty_suicide": PENALTY_SUICIDE,
        "penalty_static_invalid": PENALTY_STATIC_INVALID,
        "penalty_race_invalid": PENALTY_RACE_INVALID,
        "potential_beta": POTENTIAL_BETA,
        "training_seed": TRAINING_SEED,
        "source_variant": "M2-3",
        "curriculum_targets": list(CURRICULUM_TARGETS),
        "curriculum_stages": list(CURRICULUM_STAGES),
    }
    if BRANCH == "c3":
        config.update(
            {
                "source_branch": "c1",
                "teacher_enabled": False,
                "c1_keep_weight": C3_KEEP_WEIGHT,
                "c3_learning_rate_first": 5e-5,
                "c3_learning_rate_final": 2.5e-5,
                "c3_learning_rate_switch": 150_000,
                "c3_score_exception": C3_SCORE_EXCEPTION,
                "c3_suicide_tolerance": C3_SUICIDE_TOLERANCE,
                "c3_score_tie_window": C3_SCORE_TIE_WINDOW,
                "curriculum_targets": list(C3_LEAGUE_TARGETS),
                "curriculum_stages": list(C3_LEAGUE_STAGES),
                "league_opponents": {
                    name: list(opponents)
                    for name, opponents in C3_LEAGUE_OPPONENTS.items()
                },
            }
        )
    return config


def config_hash():
    payload = json.dumps(experiment_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_metadata(source_sha256, c1_source_sha256=None):
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "branch": BRANCH,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "config_hash": config_hash(),
        "source_variant": "M2-3",
        "source_model_sha256": source_sha256,
        "environment_commit": ENVIRONMENT_COMMIT,
    }
    if BRANCH == "c3":
        if not c1_source_sha256:
            raise ValueError("C3 metadata needs the selected C1 model hash")
        metadata["c1_source_sha256"] = c1_source_sha256
    return metadata
