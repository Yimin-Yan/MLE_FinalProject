"""Callbacks used for both training actions and tournament inference."""

import os
from pathlib import Path

import numpy as np
import torch

from .config import (
    ACTIONS,
    CHECKPOINT_FORMAT_VERSION,
    EPSILON_DECAY_STEPS,
    EPSILON_END,
    EPSILON_START,
    EVALUATION_MODEL_ENV,
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    MODEL_PATH,
    RANDOM_ACTION_STEPS,
    TRAINING_SEED,
    VARIANT_ID,
    config_hash,
)
from .features import legal_action_mask, state_to_features
from .model import DuelingDQN, mask_q_values


def _load_torch_file(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older course Docker images do not expose the weights_only argument.
        return torch.load(path, map_location=map_location)


def _check_model_metadata(metadata):
    required = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "variant_id": VARIANT_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "config_hash": config_hash(),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                "Saved model is incompatible for {}: expected {!r}, found {!r}".format(
                    key, expected, metadata.get(key)
                )
            )


def _evaluation_model_path():
    """Return a validation snapshot when one is explicitly requested."""
    override = os.environ.get(EVALUATION_MODEL_ENV)
    if not override:
        return MODEL_PATH
    return Path(override).expanduser().resolve()


def setup(self):
    """Create the network and load weights when the agent is evaluated."""
    self.device = torch.device("cpu")
    self.rng = np.random.default_rng(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)

    self.online_net = DuelingDQN(input_dim=FEATURE_DIM, action_dim=len(ACTIONS))
    self.online_net.to(self.device)
    self.online_net.eval()
    self.decision_step = 0

    if self.train:
        self.logger.info("Preparing M2-0 network for the training callbacks")
        return

    model_path = _evaluation_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            "No trained model found at {}. Evaluation must not use random weights.".format(
                model_path
            )
        )

    checkpoint = _load_torch_file(model_path, self.device)
    if "model_state_dict" not in checkpoint or "metadata" not in checkpoint:
        raise ValueError("The saved M2-0 model has an unknown format")
    _check_model_metadata(checkpoint["metadata"])
    self.online_net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    self.online_net.eval()
    self.logger.info("Loaded trained M2-0 weights from %s", model_path)


def epsilon_for_step(decision_step):
    if decision_step < RANDOM_ACTION_STEPS:
        return EPSILON_START
    progress = (decision_step - RANDOM_ACTION_STEPS) / float(EPSILON_DECAY_STEPS)
    progress = min(max(progress, 0.0), 1.0)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def act(self, game_state: dict) -> str:
    """Choose uniformly among legal actions for exploration, otherwise greedily."""
    if game_state is None:
        return "WAIT"

    features = state_to_features(game_state)
    action_mask = legal_action_mask(game_state)
    legal_indices = np.flatnonzero(action_mask)
    if legal_indices.size == 0:
        raise RuntimeError("The observed state has no legal action, including WAIT")

    use_random_action = False
    if self.train:
        epsilon = epsilon_for_step(self.decision_step)
        use_random_action = self.rng.random() < epsilon

    if use_random_action:
        action_index = int(self.rng.choice(legal_indices))
    else:
        state_tensor = torch.from_numpy(features).to(self.device).unsqueeze(0)
        mask_tensor = torch.from_numpy(action_mask).to(self.device).unsqueeze(0)
        self.online_net.eval()
        with torch.inference_mode():
            q_values = self.online_net(state_tensor)
            masked_values = mask_q_values(q_values, mask_tensor)
            action_index = int(masked_values.argmax(dim=1).item())

    if self.train:
        self.decision_step += 1
    return ACTIONS[action_index]
