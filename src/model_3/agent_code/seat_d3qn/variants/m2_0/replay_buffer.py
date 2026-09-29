"""Preallocated uniform replay buffer for the one-step baseline."""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, feature_dim):
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.states = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.next_masks = np.zeros((capacity, 6), dtype=np.bool_)
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done, next_mask):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.feature_dim,) or not np.isfinite(state).all():
            raise ValueError("Invalid state written to replay buffer")
        if not 0 <= int(action) < 6:
            raise ValueError("Replay action index is outside the six-action space")
        if not np.isfinite(float(reward)):
            raise ValueError("Non-finite reward written to replay buffer")

        if done:
            stored_next_state = np.zeros(self.feature_dim, dtype=np.float32)
            stored_next_mask = np.zeros(6, dtype=np.bool_)
        else:
            stored_next_state = np.asarray(next_state, dtype=np.float32)
            stored_next_mask = np.asarray(next_mask, dtype=np.bool_)
            if stored_next_state.shape != (self.feature_dim,):
                raise ValueError("Invalid next state written to replay buffer")
            if stored_next_mask.shape != (6,) or not stored_next_mask.any():
                raise ValueError("A non-terminal next state needs a legal action")
            if not np.isfinite(stored_next_state).all():
                raise ValueError("Non-finite next state written to replay buffer")

        index = self.position
        self.states[index] = state
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.next_states[index] = stored_next_state
        self.dones[index] = bool(done)
        self.next_masks[index] = stored_next_mask

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        if self.size < batch_size:
            raise ValueError("Not enough transitions to sample this batch")
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "next_masks": self.next_masks[indices],
        }

    def state_dict(self):
        # Saving only the occupied part keeps early checkpoints reasonably small.
        used = self.size
        return {
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "position": self.position,
            "size": self.size,
            "states": self.states[:used].copy(),
            "actions": self.actions[:used].copy(),
            "rewards": self.rewards[:used].copy(),
            "next_states": self.next_states[:used].copy(),
            "dones": self.dones[:used].copy(),
            "next_masks": self.next_masks[:used].copy(),
        }

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.capacity:
            raise ValueError("Replay capacity does not match this experiment")
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("Replay feature dimension does not match")

        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= self.capacity:
            raise ValueError("Invalid replay size in checkpoint")
        if not 0 <= position < self.capacity:
            raise ValueError("Invalid replay position in checkpoint")

        expected_shapes = {
            "states": (size, self.feature_dim),
            "actions": (size,),
            "rewards": (size,),
            "next_states": (size, self.feature_dim),
            "dones": (size,),
            "next_masks": (size, 6),
        }
        for name, shape in expected_shapes.items():
            if np.asarray(state[name]).shape != shape:
                raise ValueError("Bad replay array shape for " + name)

        self.states.fill(0.0)
        self.actions.fill(0)
        self.rewards.fill(0.0)
        self.next_states.fill(0.0)
        self.dones.fill(False)
        self.next_masks.fill(False)

        self.states[:size] = state["states"]
        self.actions[:size] = state["actions"]
        self.rewards[:size] = state["rewards"]
        self.next_states[:size] = state["next_states"]
        self.dones[:size] = state["dones"]
        self.next_masks[:size] = state["next_masks"]
        self.size = size
        self.position = position
