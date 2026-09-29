"""Dueling Q-network used by the M2-1 agent."""

import torch
from torch import nn

from .config import FEATURE_DIM, HEAD_HIDDEN_DIM, HIDDEN_DIM


class DuelingDQN(nn.Module):
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
        self.reset_parameters()

    def reset_parameters(self):
        hidden_layers = (
            self.feature_layers[0],
            self.feature_layers[2],
            self.value_stream[0],
            self.advantage_stream[0],
        )
        for layer in hidden_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        # Keeping the first Q estimates small makes early targets less jumpy.
        for output_layer in (self.value_stream[2], self.advantage_stream[2]):
            nn.init.uniform_(output_layer.weight, -1e-3, 1e-3)
            nn.init.zeros_(output_layer.bias)

    def forward(self, states):
        shared = self.feature_layers(states)
        values = self.value_stream(shared)
        advantages = self.advantage_stream(shared)
        return values + advantages - advantages.mean(dim=-1, keepdim=True)


def mask_q_values(q_values, legal_masks):
    """Exclude illegal actions immediately before an argmax operation."""
    if legal_masks.dtype != torch.bool:
        legal_masks = legal_masks.to(dtype=torch.bool)
    if q_values.shape != legal_masks.shape:
        raise ValueError("Q-values and legal masks must have the same shape")
    if not torch.all(legal_masks.any(dim=-1)):
        raise ValueError("Every non-terminal state must have a legal action")
    return q_values.masked_fill(~legal_masks, -torch.inf)
