"""Frozen configuration for the M2 Double-DQN agent."""

import os
from pathlib import Path


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
MOVE_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}

BOARD_CHANNELS = 10
BASE_AUX_FEATURES = 32
ACTION_FEATURES_PER_ACTION = 7
AUX_FEATURES = BASE_AUX_FEATURES + len(ACTIONS) * ACTION_FEATURES_PER_ACTION
BOARD_SIZE = 17
GAMMA = 0.95
N_STEP = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
REPLAY_COMPONENT_CAPACITY = 50_000
LEARNING_STARTS = 5_000
TRAIN_EVERY = 4
TARGET_UPDATE_INTERVAL = 2_000
CHECKPOINT_INTERVAL = 25_000
EPSILON_START = 0.20
EPSILON_END = 0.05
RANDOM_SEED = 2026
GRADIENT_CLIP = 10.0
BOMB_POWER = 3
BOMB_TIMER = 4

STAGE_LIMITS = {
    "task1": 100_000,
    "task2": 200_000,
    "task3a": 300_000,
    "task3b": 300_000,
    "task4": 500_000,
}

DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name("model.pt")
MODEL_PATH_ENV = "M2_MODEL_PATH"
REPLAY_PATH_ENV = "M2_REPLAY_PATH"
STAGE_ENV = "M2_TRAINING_STAGE"
RUN_ID_ENV = "M2_RUN_ID"
DISABLE_UPDATES_ENV = "M2_DISABLE_UPDATES"
DEVICE_ENV = "M2_DEVICE"
SAVE_REPLAY_ON_ROUND_END_ENV = "M2_SAVE_REPLAY_ON_ROUND_END"
CHECKPOINT_DIR_ENV = "M2_CHECKPOINT_DIR"
DEMO_PATHS_ENV = "M2_DEMO_PATHS"


def configured_model_path() -> Path:
    raw = os.environ.get(MODEL_PATH_ENV, "").strip()
    return DEFAULT_MODEL_PATH if not raw else Path(raw).resolve()


def configured_replay_path() -> Path | None:
    raw = os.environ.get(REPLAY_PATH_ENV, "").strip()
    return None if not raw else Path(raw).resolve()


def configured_checkpoint_dir() -> Path | None:
    raw = os.environ.get(CHECKPOINT_DIR_ENV, "").strip()
    return None if not raw else Path(raw).resolve()


def configured_demo_paths() -> tuple[Path, ...]:
    raw = os.environ.get(DEMO_PATHS_ENV, "").strip()
    if not raw:
        return ()
    paths = tuple(Path(item).resolve() for item in raw.split(os.pathsep) if item)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("One or more M2 demonstration files do not exist")
    return paths


def configured_stage() -> str:
    stage = os.environ.get(STAGE_ENV, "task4").strip().lower()
    if stage not in STAGE_LIMITS:
        raise ValueError(f"{STAGE_ENV} must be one of {tuple(STAGE_LIMITS)}")
    return stage


def configured_run_id(require_explicit: bool = False) -> str:
    value = os.environ.get(RUN_ID_ENV, "").strip()
    if require_explicit and not value:
        raise ValueError(f"{RUN_ID_ENV} must be set for training")
    return value or "inference"


def configured_device() -> str:
    value = os.environ.get(DEVICE_ENV, "auto").strip().lower()
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"{DEVICE_ENV} must be auto, cpu, or cuda")
    return value


def updates_disabled() -> bool:
    value = os.environ.get(DISABLE_UPDATES_ENV, "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"{DISABLE_UPDATES_ENV} must be boolean")
    return value in {"1", "true"}


def save_replay_on_round_end() -> bool:
    value = os.environ.get(SAVE_REPLAY_ON_ROUND_END_ENV, "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"{SAVE_REPLAY_ON_ROUND_END_ENV} must be boolean")
    return value in {"1", "true"}


def epsilon_for_stage(stage_steps: int, stage: str) -> float:
    fraction = min(max(stage_steps, 0) / STAGE_LIMITS[stage], 1.0)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)
