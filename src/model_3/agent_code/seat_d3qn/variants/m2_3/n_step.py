"""Three-step transition accumulator with terminal flushing."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class OneStepTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray
    escape_labels: np.ndarray
    escape_loss_mask: np.ndarray


@dataclass
class NStepTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray
    horizon: int
    escape_labels: np.ndarray
    escape_loss_mask: np.ndarray


class NStepAccumulator:
    def __init__(self, n_step, gamma):
        if int(n_step) < 1:
            raise ValueError("n_step must be positive")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.queue = deque()

    def __len__(self):
        return len(self.queue)

    def clear(self):
        self.queue.clear()

    def append(self, transition):
        if self.queue and self.queue[-1].done:
            raise RuntimeError("A terminal transition was not flushed")
        self.queue.append(transition)
        if transition.done:
            return self.flush()
        if len(self.queue) >= self.n_step:
            ready = [self._build_oldest()]
            self.queue.popleft()
            return ready
        return []

    def flush(self):
        ready = []
        while self.queue:
            ready.append(self._build_oldest())
            self.queue.popleft()
        return ready

    def _build_oldest(self):
        if not self.queue:
            raise RuntimeError("Cannot build an n-step sample from an empty queue")

        total_reward = 0.0
        last = None
        horizon = 0
        for index, item in enumerate(self.queue):
            if index >= self.n_step:
                break
            total_reward += (self.gamma ** index) * float(item.reward)
            horizon = index + 1
            last = item
            if item.done:
                break

        first = self.queue[0]
        if last.done:
            next_state = None
            next_mask = None
        else:
            next_state = np.asarray(last.next_state, dtype=np.float32).copy()
            next_mask = np.asarray(last.next_mask, dtype=np.bool_).copy()
        return NStepTransition(
            state=np.asarray(first.state, dtype=np.float32).copy(),
            action=int(first.action),
            reward=float(total_reward),
            next_state=next_state,
            done=bool(last.done),
            next_mask=next_mask,
            horizon=horizon,
            escape_labels=np.asarray(first.escape_labels, dtype=np.float32).copy(),
            escape_loss_mask=np.asarray(first.escape_loss_mask, dtype=np.bool_).copy(),
        )

    def state_dict(self):
        return {
            "n_step": self.n_step,
            "gamma": self.gamma,
            "queue": list(self.queue),
        }

    def load_state_dict(self, state):
        if int(state["n_step"]) != self.n_step:
            raise ValueError("Checkpoint n-step length does not match")
        if not np.isclose(float(state["gamma"]), self.gamma):
            raise ValueError("Checkpoint discount factor does not match")
        self.queue = deque(state.get("queue", []))
