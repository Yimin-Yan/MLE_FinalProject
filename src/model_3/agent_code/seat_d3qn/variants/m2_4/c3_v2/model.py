"""Frozen C1 value function plus a new zero-output residual adapter."""

import torch
from torch import nn

from .config import FEATURE_DIM, RESIDUAL_HEAD_DIM, RESIDUAL_HIDDEN_DIM


class C3ResidualAdapter(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, action_dim=6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, RESIDUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(RESIDUAL_HIDDEN_DIM, RESIDUAL_HEAD_DIM),
            nn.ReLU(),
            nn.Linear(RESIDUAL_HEAD_DIM, action_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.layers[0], self.layers[2]):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.layers[4].weight)
        nn.init.zeros_(self.layers[4].bias)

    def forward(self, states):
        return self.layers(states)


def frozen_c1_values(base_network, c1_residual_network, states):
    with torch.no_grad():
        return base_network(states) + c1_residual_network(states)


def mask_q_values(q_values, legal_masks):
    if legal_masks.dtype != torch.bool:
        legal_masks = legal_masks.to(dtype=torch.bool)
    if q_values.shape != legal_masks.shape:
        raise ValueError("Q-values and legal masks must have the same shape")
    if not torch.all(legal_masks.any(dim=-1)):
        raise ValueError("Every non-terminal state must have a legal action")
    return q_values.masked_fill(~legal_masks, -torch.inf)
