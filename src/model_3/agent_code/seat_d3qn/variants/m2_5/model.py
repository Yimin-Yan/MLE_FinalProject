"""The small action-conditional risk model used by M2-5."""

import torch
from torch import nn

from .config import ACTIONS, FEATURE_DIM, RISK_HIDDEN_DIMS


class WastefulDeathRisk(nn.Module):
    def __init__(self):
        super().__init__()
        first, second = RISK_HIDDEN_DIMS
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, len(ACTIONS)),
        )

    def forward(self, states):
        return self.network(states)


def chosen_action_logits(model, states, actions):
    return model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

