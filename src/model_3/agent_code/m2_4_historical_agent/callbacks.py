"""Load one older C1 checkpoint as a deterministic league opponent."""

from agent_code.m2_4_frozen_agent.callbacks import (
    greedy_action,
    setup_with_model_env,
)


MODEL_ENV = "SEAT_M2_4_HISTORICAL_MODEL_PATH"


def setup(self):
    setup_with_model_env(self, MODEL_ENV)


def act(self, game_state):
    return greedy_action(self, game_state)
