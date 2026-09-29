"""Mix frozen C1 decisions with a small amount of legal random play."""

import os

import numpy as np

from agent_code.m2_4_frozen_agent.callbacks import (
    MODEL_ENV,
    greedy_action,
    setup_with_model_env,
)
from agent_code.seat_d3qn.variants.m2_4.config import ACTIONS


def setup(self):
    setup_with_model_env(self, MODEL_ENV)
    self.noisy_rng = np.random.default_rng(
        int(os.environ.get("SEAT_M2_4_NOISY_SEED", "9042"))
    )
    self.noise_probability = float(
        os.environ.get("SEAT_M2_4_NOISE_PROBABILITY", "0.10")
    )
    if not 0.0 <= self.noise_probability <= 1.0:
        raise ValueError("SEAT_M2_4_NOISE_PROBABILITY must be in [0, 1]")


def act(self, game_state):
    if game_state is None:
        return "WAIT"
    if self.noisy_rng.random() >= self.noise_probability:
        return greedy_action(self, game_state)

    analysis = self.temporal_planner.analyze(
        game_state,
        need_escape_labels=False,
    )
    legal = np.flatnonzero(analysis.legal_mask)
    if legal.size == 0:
        raise RuntimeError("The noisy opponent found no legal action")
    return ACTIONS[int(self.noisy_rng.choice(legal))]
