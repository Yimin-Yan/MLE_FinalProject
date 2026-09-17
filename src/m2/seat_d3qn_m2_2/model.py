"""Dueling Q-network with a small auxiliary escape head."""

import torch
from torch import nn

from .config import (
    ESCAPE_HIDDEN_DIM,
    FEATURE_DIM,
    HEAD_HIDDEN_DIM,
    HIDDEN_DIM,
)


class TemporalDuelingDQN(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, action_dim=6):
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN_DIM, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN_DIM, action_dim),
        )
        self.escape_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, ESCAPE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(ESCAPE_HIDDEN_DIM, action_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        hidden_layers = (
            self.feature_layers[0],
            self.feature_layers[2],
            self.value_stream[0],
            self.advantage_stream[0],
            self.escape_head[0],
        )
        for layer in hidden_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        output_layers = (
            self.value_stream[2],
            self.advantage_stream[2],
            self.escape_head[2],
        )
        for layer in output_layers:
            nn.init.uniform_(layer.weight, -1e-3, 1e-3)
            nn.init.zeros_(layer.bias)

    def shared_features(self, states):
        return self.feature_layers(states)

    def q_values_from_shared(self, shared):
        values = self.value_stream(shared)
        advantages = self.advantage_stream(shared)
        return values + advantages - advantages.mean(dim=-1, keepdim=True)

    def forward(self, states):
        shared = self.shared_features(states)
        return self.q_values_from_shared(shared)

    def forward_with_escape(self, states):
        shared = self.shared_features(states)
        return self.q_values_from_shared(shared), self.escape_head(shared)


def mask_q_values(q_values, legal_masks):
    """Mask illegal actions right before an argmax."""
    if legal_masks.dtype != torch.bool:
        legal_masks = legal_masks.to(dtype=torch.bool)
    if q_values.shape != legal_masks.shape:
        raise ValueError("Q-values and legal masks must have the same shape")
    if not torch.all(legal_masks.any(dim=-1)):
        raise ValueError("Every non-terminal state must have a legal action")
    return q_values.masked_fill(~legal_masks, -torch.inf)
