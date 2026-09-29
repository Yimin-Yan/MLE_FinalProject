"""Compact replay storage for accepted rule-teacher action labels."""

import numpy as np


class TeacherReplayBuffer:
    def __init__(self, capacity, feature_dim):
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.states = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, 6), dtype=np.bool_)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.confidences = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, state, legal_mask, action, confidence):
        state = np.asarray(state, dtype=np.float32)
        legal_mask = np.asarray(legal_mask, dtype=np.bool_)
        action = int(action)
        confidence = float(confidence)
        if state.shape != (self.feature_dim,) or not np.isfinite(state).all():
            raise ValueError("Invalid state written to teacher replay")
        if legal_mask.shape != (6,) or not legal_mask.any():
            raise ValueError("Invalid legal mask written to teacher replay")
        if not 0 <= action < 6 or not legal_mask[action]:
            raise ValueError("Teacher action is not legal in its stored state")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Teacher confidence is outside [0, 1]")

        index = self.position
        self.states[index] = state
        self.legal_masks[index] = legal_mask
        self.actions[index] = action
        self.confidences[index] = confidence
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        if self.size < int(batch_size):
            raise ValueError("Not enough teacher labels to sample this batch")
        indices = rng.choice(self.size, size=int(batch_size), replace=False)
        return {
            "states": self.states[indices],
            "legal_masks": self.legal_masks[indices],
            "actions": self.actions[indices],
            "confidences": self.confidences[indices],
        }

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "position": self.position,
            "size": self.size,
            "states": self.states[: self.size].copy(),
            "legal_masks": self.legal_masks[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "confidences": self.confidences[: self.size].copy(),
        }

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.capacity:
            raise ValueError("Teacher replay capacity does not match")
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("Teacher replay feature dimension does not match")
        size = int(state["size"])
        if not 0 <= size <= self.capacity:
            raise ValueError("Teacher replay size is invalid")
        self.states[:size] = state["states"]
        self.legal_masks[:size] = state["legal_masks"]
        self.actions[:size] = state["actions"]
        self.confidences[:size] = state["confidences"]
        self.size = size
        self.position = int(state["position"])

