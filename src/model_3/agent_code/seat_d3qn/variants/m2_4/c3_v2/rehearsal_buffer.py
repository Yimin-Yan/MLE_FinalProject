"""Read-only LAP sampler for the frozen C1 replay memory."""

import numpy as np


class FrozenRehearsalBuffer:
    """Keep the C1 data fixed while sampling with its saved LAP priorities."""

    _BATCH_FIELDS = (
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
        "next_masks",
        "horizons",
        "escape_labels",
        "escape_loss_masks",
    )

    def __init__(self, state, feature_dim, alpha, priority_floor):
        self.feature_dim = int(feature_dim)
        self.alpha = float(alpha)
        self.priority_floor = float(priority_floor)
        self.size = int(state["size"])

        if self.size < 1:
            raise ValueError("The C1 rehearsal buffer is empty")
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("C1 rehearsal has the wrong feature dimension")
        if not np.isclose(float(state["alpha"]), self.alpha):
            raise ValueError("C1 rehearsal has a different LAP alpha")
        if not np.isclose(float(state["priority_floor"]), self.priority_floor):
            raise ValueError("C1 rehearsal has a different priority floor")

        expected_shapes = {
            "states": (self.size, self.feature_dim),
            "actions": (self.size,),
            "rewards": (self.size,),
            "next_states": (self.size, self.feature_dim),
            "dones": (self.size,),
            "next_masks": (self.size, 6),
            "horizons": (self.size,),
            "escape_labels": (self.size, 6),
            "escape_loss_masks": (self.size, 6),
            "raw_priorities": (self.size,),
        }
        self.arrays = {}
        for name, shape in expected_shapes.items():
            value = np.asarray(state[name])
            if value.shape != shape:
                raise ValueError("Bad C1 rehearsal shape for " + name)
            self.arrays[name] = value.copy()

        raw = self.arrays["raw_priorities"].astype(np.float64, copy=False)
        if not np.isfinite(raw).all() or np.any(raw < self.priority_floor):
            raise ValueError("C1 rehearsal contains invalid priorities")
        weights = np.maximum(raw, self.priority_floor) ** self.alpha
        self.cumulative_weights = np.cumsum(weights, dtype=np.float64)
        self.total_weight = float(self.cumulative_weights[-1])
        if not np.isfinite(self.total_weight) or self.total_weight <= 0.0:
            raise FloatingPointError("C1 rehearsal has invalid total priority")

    def __len__(self):
        return self.size

    def sample(self, batch_size, rng):
        batch_size = int(batch_size)
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("Invalid frozen rehearsal batch size")

        segment = self.total_weight / float(batch_size)
        low = np.arange(batch_size, dtype=np.float64) * segment
        masses = low + rng.random(batch_size) * segment
        masses = np.minimum(masses, np.nextafter(self.total_weight, 0.0))
        indices = np.searchsorted(
            self.cumulative_weights, masses, side="right"
        ).astype(np.int64)
        if np.any(indices >= self.size):
            raise RuntimeError("Frozen rehearsal sampled an unused item")

        batch = {name: self.arrays[name][indices] for name in self._BATCH_FIELDS}
        batch["indices"] = indices
        batch["raw_priorities"] = self.arrays["raw_priorities"][indices].astype(
            np.float32
        )
        return batch
