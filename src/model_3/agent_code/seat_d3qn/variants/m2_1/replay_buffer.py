"""Loss-adjusted prioritized replay used by M2-1."""

import numpy as np


class _SumTree:
    def __init__(self, capacity):
        tree_capacity = 1
        while tree_capacity < int(capacity):
            tree_capacity *= 2
        self.capacity = int(capacity)
        self.tree_capacity = tree_capacity
        self.values = np.zeros(2 * tree_capacity, dtype=np.float64)

    @property
    def total(self):
        return float(self.values[1])

    def update(self, index, value):
        leaf = self.tree_capacity + int(index)
        self.values[leaf] = float(value)
        leaf //= 2
        while leaf >= 1:
            left = 2 * leaf
            self.values[leaf] = self.values[left] + self.values[left + 1]
            leaf //= 2

    def find_prefix(self, mass):
        if not 0.0 <= float(mass) < self.total:
            raise ValueError("Priority mass is outside the sum tree")
        node = 1
        while node < self.tree_capacity:
            left = 2 * node
            if mass < self.values[left]:
                node = left
            else:
                mass -= self.values[left]
                node = left + 1
        return node - self.tree_capacity


class LAPReplayBuffer:
    def __init__(self, capacity, feature_dim, alpha, priority_floor):
        if int(capacity) < 1:
            raise ValueError("Replay capacity must be positive")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("LAP alpha must be in [0, 1]")
        if not np.isfinite(float(priority_floor)) or float(priority_floor) <= 0.0:
            raise ValueError("LAP priority floor must be finite and positive")

        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.alpha = float(alpha)
        self.priority_floor = float(priority_floor)
        self.states = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.next_masks = np.zeros((capacity, 6), dtype=np.bool_)
        self.horizons = np.zeros(capacity, dtype=np.int64)
        self.raw_priorities = np.zeros(capacity, dtype=np.float64)
        self.tree = _SumTree(capacity)
        self.max_priority = self.priority_floor
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done, next_mask, horizon):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.feature_dim,) or not np.isfinite(state).all():
            raise ValueError("Invalid state written to LAP replay")
        if not 0 <= int(action) < 6:
            raise ValueError("Replay action index is outside the six-action space")
        if not np.isfinite(float(reward)):
            raise ValueError("Non-finite reward written to LAP replay")
        if int(horizon) < 1:
            raise ValueError("Replay horizon must be positive")

        if done:
            stored_next_state = np.zeros(self.feature_dim, dtype=np.float32)
            stored_next_mask = np.zeros(6, dtype=np.bool_)
        else:
            stored_next_state = np.asarray(next_state, dtype=np.float32)
            stored_next_mask = np.asarray(next_mask, dtype=np.bool_)
            if stored_next_state.shape != (self.feature_dim,):
                raise ValueError("Invalid next state written to LAP replay")
            if not np.isfinite(stored_next_state).all():
                raise ValueError("Non-finite next state written to LAP replay")
            if stored_next_mask.shape != (6,) or not stored_next_mask.any():
                raise ValueError("A non-terminal replay item needs a legal next action")

        index = self.position
        self.states[index] = state
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.next_states[index] = stored_next_state
        self.dones[index] = bool(done)
        self.next_masks[index] = stored_next_mask
        self.horizons[index] = int(horizon)

        # A new item gets the current maximum so it is sampled at least once.
        raw_priority = max(self.max_priority, self.priority_floor)
        self.raw_priorities[index] = raw_priority
        self.tree.update(index, raw_priority ** self.alpha)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        if self.size < int(batch_size):
            raise ValueError("Not enough transitions to sample this batch")
        total = self.tree.total
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("Invalid total LAP priority")

        # One draw per equal-mass segment reduces variance in a minibatch.
        segment = total / float(batch_size)
        indices = np.empty(batch_size, dtype=np.int64)
        probabilities = np.empty(batch_size, dtype=np.float32)
        for sample_index in range(batch_size):
            low = sample_index * segment
            high = (sample_index + 1) * segment
            mass = float(rng.uniform(low, high))
            if mass >= total:
                mass = np.nextafter(total, 0.0)
            replay_index = self.tree.find_prefix(mass)
            if replay_index >= self.size:
                raise RuntimeError("LAP sampled an unused replay slot")
            leaf = self.tree.tree_capacity + replay_index
            indices[sample_index] = replay_index
            probabilities[sample_index] = self.tree.values[leaf] / total

        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0.0):
            raise FloatingPointError("Invalid LAP sampling probability")
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "next_masks": self.next_masks[indices],
            "horizons": self.horizons[indices],
            "indices": indices,
            "sampling_probabilities": probabilities,
            "raw_priorities": self.raw_priorities[indices].astype(np.float32),
        }

    def update_from_td_errors(self, indices, td_errors):
        indices = np.asarray(indices, dtype=np.int64)
        td_errors = np.asarray(td_errors, dtype=np.float64)
        if indices.shape != td_errors.shape:
            raise ValueError("Priority indices and TD errors must have the same shape")
        if not np.isfinite(td_errors).all():
            raise FloatingPointError("LAP received a non-finite TD error")

        raw_priorities = np.maximum(np.abs(td_errors), self.priority_floor)
        for index, raw_priority in zip(indices, raw_priorities):
            if not 0 <= int(index) < self.size:
                raise IndexError("Priority update points outside the replay buffer")
            self.raw_priorities[index] = float(raw_priority)
            # Alpha is applied here once, and nowhere else in the training code.
            self.tree.update(int(index), float(raw_priority) ** self.alpha)
        self.max_priority = max(self.max_priority, float(raw_priorities.max()))

    def priority_statistics(self):
        used = self.raw_priorities[: self.size]
        if used.size == 0:
            return 0.0, 0.0, 0.0
        above_floor = float(np.mean(used > self.priority_floor))
        return float(used.mean()), float(used.max()), above_floor

    def state_dict(self):
        used = self.size
        return {
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "alpha": self.alpha,
            "priority_floor": self.priority_floor,
            "position": self.position,
            "size": self.size,
            "max_priority": self.max_priority,
            "states": self.states[:used].copy(),
            "actions": self.actions[:used].copy(),
            "rewards": self.rewards[:used].copy(),
            "next_states": self.next_states[:used].copy(),
            "dones": self.dones[:used].copy(),
            "next_masks": self.next_masks[:used].copy(),
            "horizons": self.horizons[:used].copy(),
            "raw_priorities": self.raw_priorities[:used].copy(),
        }

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.capacity:
            raise ValueError("Replay capacity does not match this experiment")
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("Replay feature dimension does not match")
        if not np.isclose(float(state["alpha"]), self.alpha):
            raise ValueError("Replay alpha does not match")
        if not np.isclose(float(state["priority_floor"]), self.priority_floor):
            raise ValueError("Replay priority floor does not match")

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
            "horizons": (size,),
            "raw_priorities": (size,),
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
        self.horizons.fill(0)
        self.raw_priorities.fill(0.0)
        self.tree = _SumTree(self.capacity)

        self.states[:size] = state["states"]
        self.actions[:size] = state["actions"]
        self.rewards[:size] = state["rewards"]
        self.next_states[:size] = state["next_states"]
        self.dones[:size] = state["dones"]
        self.next_masks[:size] = state["next_masks"]
        self.horizons[:size] = state["horizons"]
        self.raw_priorities[:size] = state["raw_priorities"]
        for index in range(size):
            raw_priority = float(self.raw_priorities[index])
            if not np.isfinite(raw_priority) or raw_priority < self.priority_floor:
                raise ValueError("Checkpoint contains an invalid LAP priority")
            self.tree.update(index, raw_priority ** self.alpha)

        self.size = size
        self.position = position
        self.max_priority = float(state["max_priority"])
        if not np.isfinite(self.max_priority) or self.max_priority < self.priority_floor:
            raise ValueError("Checkpoint contains an invalid maximum priority")
