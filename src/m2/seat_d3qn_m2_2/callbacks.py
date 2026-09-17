"""Action callbacks for M2-2 training and evaluation."""

import os
from pathlib import Path

import numpy as np
import torch

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
from .features import state_to_features
from .model import TemporalDuelingDQN, mask_q_values
from .planner import KnownHazardPlanner


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
                "Saved model is incompatible for {}: expected {!r}, found {!r}".format(
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
    """Prepare the network, planner, and deterministic random generator."""
    self.device = torch.device("cpu")
    self.rng = np.random.default_rng(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    self.temporal_planner = KnownHazardPlanner()
    self.shield_window = _fresh_shield_window()

    self.online_net = TemporalDuelingDQN(
        input_dim=FEATURE_DIM,
        action_dim=len(ACTIONS),
    ).to(self.device)
    self.online_net.eval()
    self.decision_step = 0

    if self.train:
        self.logger.info("M2-2 will start from the frozen M2-1 checkpoint")
        return

    model_path = _evaluation_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            "No trained M2-2 model found at {}. Evaluation needs saved weights.".format(
                model_path
            )
        )
    checkpoint = _load_torch_file(model_path, self.device)
    if "model_state_dict" not in checkpoint or "metadata" not in checkpoint:
        raise ValueError("The saved M2-2 model has an unknown format")
    _check_model_metadata(checkpoint["metadata"])
    self.online_net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    self.online_net.eval()
    self.logger.info("Loaded trained M2-2 weights from %s", model_path)


def _fresh_shield_window():
    return {
        "random_decisions": 0,
        "legal_actions": 0,
        "pruned_actions": 0,
        "bomb_candidates": 0,
        "bomb_pruned": 0,
        "no_safe_fallbacks": 0,
    }


def epsilon_for_step(decision_step):
    progress = min(max(decision_step / float(EPSILON_DECAY_STEPS), 0.0), 1.0)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def _greedy_action(self, features, legal_mask):
    state_tensor = torch.from_numpy(features).to(self.device).unsqueeze(0)
    mask_tensor = torch.from_numpy(legal_mask).to(self.device).unsqueeze(0)
    self.online_net.eval()
    with torch.inference_mode():
        q_values = self.online_net(state_tensor)
        q_values = mask_q_values(q_values, mask_tensor)
    return int(q_values.argmax(dim=1).item())


def act(self, game_state: dict) -> str:
    """Choose an action; the safety filter only touches random exploration."""
    if game_state is None:
        return "WAIT"

    analysis = self.temporal_planner.analyze(
        game_state,
        need_escape_labels=self.train,
    )
    features = state_to_features(game_state, analysis)
    legal_mask = analysis.legal_mask
    legal_indices = np.flatnonzero(legal_mask)
    if legal_indices.size == 0:
        raise RuntimeError("The observed state has no legal action, including WAIT")

    use_random_action = self.train and (
        self.rng.random() < epsilon_for_step(self.decision_step)
    )

    if use_random_action:
        safe_mask = legal_mask & analysis.escape_feasible
        safe_indices = np.flatnonzero(safe_mask)
        window = self.shield_window
        window["random_decisions"] += 1
        window["legal_actions"] += int(legal_mask.sum())
        window["pruned_actions"] += int((legal_mask & ~safe_mask).sum())
        if legal_mask[5]:
            window["bomb_candidates"] += 1
            window["bomb_pruned"] += int(not safe_mask[5])

        if safe_indices.size > 0:
            action_index = int(self.rng.choice(safe_indices))
        else:
            # If everything looks bad, trust the learned legal Q ranking.
            window["no_safe_fallbacks"] += 1
            action_index = _greedy_action(self, features, legal_mask)
    else:
        action_index = _greedy_action(self, features, legal_mask)

    if self.train:
        self.decision_step += 1
    return ACTIONS[action_index]
