"""Greedy evaluation wrapper for a frozen M2-4 residual checkpoint."""

import hashlib
import os
from pathlib import Path

import torch

from agent_code.seat_d3qn.variants.m2_3.features import state_to_features
from agent_code.seat_d3qn.variants.m2_3.model import TemporalDuelingDQN
from agent_code.seat_d3qn.variants.m2_3.planner import KnownHazardPlanner
from agent_code.seat_d3qn.variants.m2_4.config import ACTIONS, FEATURE_DIM, M2_3_MODEL_PATH
from agent_code.seat_d3qn.variants.m2_4.model import ResidualQNetwork, mask_q_values


MODEL_ENV = "SEAT_M2_4_OPPONENT_MODEL_PATH"


def _load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_with_model_env(self, model_env):
    self.device = torch.device("cpu")
    self.temporal_planner = KnownHazardPlanner()
    if self.train:
        raise RuntimeError("m2_4_frozen_agent is an evaluation-only opponent")

    override = os.environ.get(model_env)
    if not override:
        raise ValueError("{} must name the frozen opponent checkpoint".format(model_env))
    residual_path = Path(override).expanduser().resolve()
    if not residual_path.is_file() or not M2_3_MODEL_PATH.is_file():
        raise FileNotFoundError("Frozen self-play checkpoint or M2-3 base is missing")

    base_checkpoint = _load(M2_3_MODEL_PATH, self.device)
    residual_checkpoint = _load(residual_path, self.device)
    metadata = residual_checkpoint.get("metadata", {})
    if metadata.get("source_model_sha256") != _sha256(M2_3_MODEL_PATH):
        raise ValueError("Frozen opponent was built on a different M2-3 model")
    if metadata.get("feature_dim") != FEATURE_DIM:
        raise ValueError("Frozen opponent has the wrong feature dimension")

    self.base_net = TemporalDuelingDQN(FEATURE_DIM, len(ACTIONS)).to(self.device)
    self.base_net.load_state_dict(base_checkpoint["model_state_dict"], strict=True)
    self.base_net.eval()
    self.residual_net = ResidualQNetwork(FEATURE_DIM, len(ACTIONS)).to(self.device)
    self.residual_net.load_state_dict(
        residual_checkpoint["residual_state_dict"], strict=True
    )
    self.residual_net.eval()


def setup(self):
    setup_with_model_env(self, MODEL_ENV)


def greedy_action(self, game_state):
    if game_state is None:
        return "WAIT"
    analysis = self.temporal_planner.analyze(game_state, need_escape_labels=False)
    features = state_to_features(game_state, analysis)
    states = torch.from_numpy(features).to(self.device).unsqueeze(0)
    masks = torch.from_numpy(analysis.legal_mask).to(self.device).unsqueeze(0)
    with torch.inference_mode():
        q_values = self.base_net(states) + self.residual_net(states)
        q_values = mask_q_values(q_values, masks)
    return ACTIONS[int(q_values.argmax(dim=1).item())]


def act(self, game_state):
    return greedy_action(self, game_state)
