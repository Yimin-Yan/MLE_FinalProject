"""Action callbacks for the protected C1 continuation."""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from ...m2_3.features import state_to_features
from ...m2_3.model import TemporalDuelingDQN
from ...m2_3.planner import KnownHazardPlanner
from .config import (
    ACTIONS,
    CHECKPOINT_FORMAT_VERSION,
    EVALUATION_MODEL_ENV,
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    MASTER_TRAINING_SEED,
    M2_3_MODEL_PATH,
    C1_MODEL_PATH,
    MODEL_PATH,
    VARIANT_ID,
    ALGORITHM_ID,
    config_hash,
    epsilon_for_transition,
)
from .model import C3ResidualAdapter, mask_q_values


def _load_torch_file(path, map_location, weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_m2_3(checkpoint):
    metadata = checkpoint.get("metadata", {})
    required = {
        "variant_id": "M2-3",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError("M2-3 source mismatch for {}".format(key))


def _check_c1(checkpoint, m2_3_sha256):
    metadata = checkpoint.get("metadata", {})
    if metadata.get("variant_id") != "M2-4-C1":
        raise ValueError("C3-v2 source is not the selected M2-4 C1 model")
    if metadata.get("source_model_sha256") != m2_3_sha256:
        raise ValueError("C1 and C3-v2 do not share the same M2-3 source")
    if int(checkpoint.get("transition_count", -1)) != 250_000:
        raise ValueError("C3-v2 must start from C1 t0250000")


def _check_adapter(checkpoint, self):
    metadata = checkpoint.get("metadata", {})
    required = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "config_hash": config_hash(),
        "m2_3_source_sha256": self.m2_3_source_sha256,
        "c1_source_sha256": self.c1_source_sha256,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError("Saved C3-v2 model mismatch for {}".format(key))


def _evaluation_model_path():
    override = os.environ.get(EVALUATION_MODEL_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return MODEL_PATH


def _fresh_shield_window():
    return {
        "random_decisions": 0,
        "legal_actions": 0,
        "pruned_actions": 0,
        "bomb_candidates": 0,
        "bomb_pruned": 0,
        "no_safe_fallbacks": 0,
    }


def setup(self):
    """Load and freeze C1, then attach a fresh zero-output C3 adapter."""
    self.device = torch.device("cpu")
    self.rng = np.random.default_rng(MASTER_TRAINING_SEED)
    torch.manual_seed(MASTER_TRAINING_SEED)
    self.temporal_planner = KnownHazardPlanner()
    self.shield_window = _fresh_shield_window()
    self.decision_step = 0
    self.source_transition_count = 0

    if not M2_3_MODEL_PATH.is_file():
        raise FileNotFoundError("C3-v2 needs M2-3 at {}".format(M2_3_MODEL_PATH))
    if not C1_MODEL_PATH.is_file():
        raise FileNotFoundError("C3-v2 needs selected C1 at {}".format(C1_MODEL_PATH))

    m2_3 = _load_torch_file(M2_3_MODEL_PATH, self.device)
    _check_m2_3(m2_3)
    self.m2_3_source_sha256 = _sha256(M2_3_MODEL_PATH)
    self.base_net = TemporalDuelingDQN(FEATURE_DIM, len(ACTIONS)).to(self.device)
    self.base_net.load_state_dict(m2_3["model_state_dict"], strict=True)
    self.base_net.eval()
    for parameter in self.base_net.parameters():
        parameter.requires_grad_(False)

    c1 = _load_torch_file(C1_MODEL_PATH, self.device)
    _check_c1(c1, self.m2_3_source_sha256)
    self.c1_source_sha256 = _sha256(C1_MODEL_PATH)
    self.c1_residual_net = C3ResidualAdapter(FEATURE_DIM, len(ACTIONS)).to(self.device)
    self.c1_residual_net.load_state_dict(c1["residual_state_dict"], strict=True)
    self.c1_residual_net.eval()
    for parameter in self.c1_residual_net.parameters():
        parameter.requires_grad_(False)

    self.adapter_net = C3ResidualAdapter(FEATURE_DIM, len(ACTIONS)).to(self.device)
    self.adapter_net.eval()
    with torch.no_grad():
        probe = torch.linspace(-1.0, 1.0, steps=4 * FEATURE_DIM).reshape(4, FEATURE_DIM)
        self.transfer_max_q_error = float(self.adapter_net(probe).abs().max().item())

    if self.train:
        self.logger.info("C3-v2 starts from frozen C1 with a zero-output adapter")
        return

    model_path = _evaluation_model_path()
    if not model_path.is_file():
        raise FileNotFoundError("No trained C3-v2 model found at {}".format(model_path))
    checkpoint = _load_torch_file(model_path, self.device)
    _check_adapter(checkpoint, self)
    self.adapter_net.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    self.adapter_net.eval()
    self.logger.info("Loaded C3-v2 weights from %s", model_path)


def _network_values(self, features):
    states = torch.from_numpy(features).to(self.device).unsqueeze(0)
    with torch.inference_mode():
        frozen = self.base_net(states) + self.c1_residual_net(states)
        total = frozen + self.adapter_net(states)
    return total


def _greedy_index(values, legal_mask):
    mask = torch.from_numpy(legal_mask).to(values.device).unsqueeze(0)
    return int(mask_q_values(values, mask).argmax(dim=1).item())


def act(self, game_state):
    """Use C1 plus the learned adapter; exploration is tied to saved data."""
    if game_state is None:
        return "WAIT"
    analysis = self.temporal_planner.analyze(game_state, need_escape_labels=self.train)
    features = state_to_features(game_state, analysis)
    legal_mask = analysis.legal_mask
    legal_indices = np.flatnonzero(legal_mask)
    if legal_indices.size == 0:
        raise RuntimeError("The observed state has no legal action, including WAIT")

    values = _network_values(self, features)
    epsilon = epsilon_for_transition(self.source_transition_count)
    random_action = self.train and self.rng.random() < epsilon
    if random_action:
        safe_mask = legal_mask & analysis.escape_feasible
        safe_indices = np.flatnonzero(safe_mask)
        window = self.shield_window
        window["random_decisions"] += 1
        window["legal_actions"] += int(legal_mask.sum())
        window["pruned_actions"] += int((legal_mask & ~safe_mask).sum())
        if legal_mask[5]:
            window["bomb_candidates"] += 1
            window["bomb_pruned"] += int(not safe_mask[5])
        if safe_indices.size:
            action_index = int(self.rng.choice(safe_indices))
        else:
            window["no_safe_fallbacks"] += 1
            action_index = _greedy_index(values, legal_mask)
    else:
        action_index = _greedy_index(values, legal_mask)

    if self.train:
        self.decision_step += 1
    return ACTIONS[action_index]
