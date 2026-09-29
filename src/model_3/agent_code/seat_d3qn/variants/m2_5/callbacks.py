"""Actions for data collection and for the final recovery-gated policy."""

import os

import numpy as np
import torch

from ..m2_3.features import state_to_features
from ..m2_3.planner import KnownHazardPlanner
from .config import (
    ACTIONS,
    COLLECTION_SEED_ENV,
    DISABLE_GATE_ENV,
    EXPLORATION_COOLDOWN,
    HAZARD_EXPLORATION_PROBABILITY,
    MODE_ENV,
    SOURCE_SHA256,
    checkpoint_metadata,
    evaluation_model_path,
)
from .model import WastefulDeathRisk
from .policy import greedy_legal_action, select_recovery_action
from .source import FrozenC3Policy, load_torch


def _state_key(game_state):
    if game_state is None:
        return None
    return int(game_state["round"]), int(game_state["step"])


def _check_risk_checkpoint(checkpoint):
    metadata = checkpoint.get("metadata", {})
    expected = checkpoint_metadata()
    for key in (
        "format_version",
        "variant_id",
        "algorithm_id",
        "source_sha256",
        "feature_schema_version",
        "risk_schema_version",
        "feature_dim",
        "actions",
        "config_hash",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError("Saved M2-5 model mismatch for {}".format(key))
    if metadata.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("M2-5 risk model belongs to another source policy")


def setup(self):
    self.device = torch.device("cpu")
    self.temporal_planner = KnownHazardPlanner()
    self.source_net = FrozenC3Policy(self.device)
    self.last_observation = None
    self.risk_net = None

    if self.train:
        if os.environ.get(MODE_ENV) != "collect":
            raise ValueError("Training M2-5 is only allowed in collection mode")
        seed = int(os.environ.get(COLLECTION_SEED_ENV, "0"))
        self.collection_rng = np.random.default_rng(seed)
        self.collection_cooldown = 0
        return

    model_path = evaluation_model_path()
    if not model_path.is_file():
        raise FileNotFoundError("No trained M2-5 risk model found at {}".format(model_path))
    checkpoint = load_torch(model_path, self.device, weights_only=False)
    _check_risk_checkpoint(checkpoint)
    self.risk_net = WastefulDeathRisk().to(self.device)
    self.risk_net.load_state_dict(checkpoint["risk_state_dict"], strict=True)
    self.risk_net.eval()
    self.risk_threshold = float(checkpoint["risk_threshold"])
    self.risk_margin = float(checkpoint["risk_margin"])
    self.q_gap_cap = float(checkpoint["q_gap_cap"])
    self.disable_gate = os.environ.get(DISABLE_GATE_ENV, "0") == "1"


def _source_values(self, features):
    states = torch.from_numpy(features).to(self.device).unsqueeze(0)
    with torch.inference_mode():
        return self.source_net(states).squeeze(0).cpu().numpy()


def _collection_action(self, base_index, analysis):
    explored = False
    chosen = base_index
    if self.collection_cooldown > 0:
        self.collection_cooldown -= 1
        return chosen, explored

    can_probe = bool(analysis.known_hazard_present) and base_index != 5
    if can_probe and self.collection_rng.random() < HAZARD_EXPLORATION_PROBABILITY:
        candidates = np.flatnonzero(analysis.legal_mask)
        candidates = candidates[(candidates != 5) & (candidates != base_index)]
        if candidates.size:
            chosen = int(self.collection_rng.choice(candidates))
            explored = True
            self.collection_cooldown = EXPLORATION_COOLDOWN
    return chosen, explored


def act(self, game_state):
    if game_state is None:
        return "WAIT"
    analysis = self.temporal_planner.analyze(game_state, need_escape_labels=False)
    features = state_to_features(game_state, analysis)
    q_values = _source_values(self, features)
    base_index = greedy_legal_action(q_values, analysis.legal_mask)

    explored = False
    changed = False
    chosen = base_index
    if self.train:
        chosen, explored = _collection_action(self, base_index, analysis)
    elif not self.disable_gate:
        states = torch.from_numpy(features).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            risks = torch.sigmoid(self.risk_net(states)).squeeze(0).cpu().numpy()
        chosen, changed = select_recovery_action(
            q_values=q_values,
            risk_scores=risks,
            legal_mask=analysis.legal_mask,
            known_hazard=analysis.known_hazard_present,
            risk_threshold=self.risk_threshold,
            risk_margin=self.risk_margin,
            q_gap_cap=self.q_gap_cap,
        )

    action = ACTIONS[chosen]
    self.last_observation = {
        "key": _state_key(game_state),
        "features": features.copy(),
        "action": action,
        "explored": bool(explored),
        "gate_changed": bool(changed),
    }
    return action

