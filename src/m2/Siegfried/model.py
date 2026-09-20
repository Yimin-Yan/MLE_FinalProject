"""Dueling convolutional Q network and checkpoint helpers."""

from pathlib import Path
import os

import torch
from torch import nn

from .config import (
    ACTION_FEATURES_PER_ACTION,
    ACTIONS,
    AUX_FEATURES,
    BASE_AUX_FEATURES,
    BOARD_CHANNELS,
    BOARD_SIZE,
)


MODEL_FORMAT_VERSION = 1


class DuelingQNetwork(nn.Module):
    def __init__(self, n_actions: int = 6):
        super().__init__()
        if n_actions != len(ACTIONS):
            raise ValueError(f"Expected {len(ACTIONS)} actions, got {n_actions}")
        self.encoder = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            encoded = self.encoder(
                torch.zeros(1, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            ).numel()
        self.trunk = nn.Sequential(
            nn.Linear(encoded + BASE_AUX_FEATURES, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions)
        )
        self.tactical_residual = nn.Linear(ACTION_FEATURES_PER_ACTION, 1, bias=False)
        nn.init.zeros_(self.tactical_residual.weight)

    def forward(self, board: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        if aux.shape[1] != AUX_FEATURES:
            raise ValueError(f"Expected {AUX_FEATURES} auxiliary values, got {aux.shape[1]}")
        encoded = self.encoder(board).flatten(1)
        base_aux = aux[:, :BASE_AUX_FEATURES]
        tactical = aux[:, BASE_AUX_FEATURES:].reshape(
            aux.shape[0], len(ACTIONS), ACTION_FEATURES_PER_ACTION
        )
        hidden = self.trunk(torch.cat((encoded, base_aux), dim=1))
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        base_q = value + advantage - advantage.mean(dim=1, keepdim=True)
        residual = self.tactical_residual(tactical).squeeze(-1)
        return base_q + residual


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": MODEL_FORMAT_VERSION, **payload}, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    if int(payload.get("format_version", -1)) != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported M2 checkpoint format")
    return payload


def load_policy_state(model: nn.Module, policy_state: dict) -> bool:
    """Load V15 weights, or migrate the frozen 32-aux predecessor once.

    Every predecessor tensor is copied exactly.  The missing action-aligned
    residual is zero initialized, so migration adds no tactical bias and
    preserves the predecessor Q values exactly.
    Returns whether migration was required.
    """
    current = model.state_dict()
    migrated = False
    for name, value in policy_state.items():
        if name not in current:
            raise RuntimeError(f"Unexpected checkpoint tensor {name}")
        target = current[name]
        value = value.to(device=target.device, dtype=target.dtype)
        if value.shape == target.shape:
            current[name] = value
            continue
        raise RuntimeError(
            f"Incompatible checkpoint tensor {name}: {tuple(value.shape)} -> {tuple(target.shape)}"
        )
    missing = set(current) - set(policy_state)
    allowed_missing = {"tactical_residual.weight"}
    if missing == allowed_missing:
        current["tactical_residual.weight"] = torch.zeros_like(
            current["tactical_residual.weight"]
        )
        migrated = True
    elif missing:
        raise RuntimeError(f"Missing checkpoint tensors: {sorted(missing)}")
    model.load_state_dict(current, strict=True)
    return migrated
