"""Action callbacks for the matched M2-2 control run."""

import os
from pathlib import Path

import numpy as np
import torch

from ..m2_1.features import legal_action_mask, state_to_features
from ..m2_1.model import DuelingDQN, mask_q_values
from .config import (
    ACTIONS,
    ALGORITHM_ID,
    CHECKPOINT_FORMAT_VERSION,
    EPSILON_DECAY_STEPS,
    EPSILON_END,
    EPSILON_START,
    EVALUATION_MODEL_ENV,
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    MODEL_PATH,
    TRAINING_SEED,
    VARIANT_ID,
    config_hash,
)


def _load_torch_file(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _check_model_metadata(metadata):
    required = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "config_hash": config_hash(),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                "Control model mismatch for {}: expected {!r}, found {!r}".format(
                    key,
                    expected,
                    metadata.get(key),
                )
            )


def _evaluation_model_path():
    override = os.environ.get(EVALUATION_MODEL_ENV)
    if not override:
        return MODEL_PATH
    return Path(override).expanduser().resolve()


def setup(self):
    self.device = torch.device("cpu")
    self.rng = np.random.default_rng(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    self.online_net = DuelingDQN(input_dim=FEATURE_DIM, action_dim=len(ACTIONS))
    self.online_net.to(self.device)
    self.online_net.eval()
    self.decision_step = 0

    if self.train:
        self.logger.info("Control run will start from the frozen M2-1 checkpoint")
        return

    model_path = _evaluation_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            "No trained M2-2 control model found at {}".format(model_path)
        )
    checkpoint = _load_torch_file(model_path, self.device)
    if "model_state_dict" not in checkpoint or "metadata" not in checkpoint:
        raise ValueError("The saved control model has an unknown format")
    _check_model_metadata(checkpoint["metadata"])
    self.online_net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    self.online_net.eval()
    self.logger.info("Loaded M2-2 control weights from %s", model_path)


def epsilon_for_step(decision_step):
    progress = min(max(decision_step / float(EPSILON_DECAY_STEPS), 0.0), 1.0)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def act(self, game_state: dict) -> str:
    if game_state is None:
        return "WAIT"

    features = state_to_features(game_state)
    action_mask = legal_action_mask(game_state)
    legal_indices = np.flatnonzero(action_mask)
    if legal_indices.size == 0:
        raise RuntimeError("The observed state has no legal action, including WAIT")

    use_random_action = self.train and (
        self.rng.random() < epsilon_for_step(self.decision_step)
    )
    if use_random_action:
        action_index = int(self.rng.choice(legal_indices))
    else:
        state_tensor = torch.from_numpy(features).to(self.device).unsqueeze(0)
        mask_tensor = torch.from_numpy(action_mask).to(self.device).unsqueeze(0)
        self.online_net.eval()
        with torch.inference_mode():
            q_values = self.online_net(state_tensor)
            q_values = mask_q_values(q_values, mask_tensor)
            action_index = int(q_values.argmax(dim=1).item())

    if self.train:
        self.decision_step += 1
    return ACTIONS[action_index]
