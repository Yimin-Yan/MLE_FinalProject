"""State-synchronised wrapper around the official rule-based agent."""

from collections import Counter, deque
from dataclasses import dataclass
from types import SimpleNamespace
import random

import numpy as np

from agent_code.rule_based_agent import callbacks as rule_callbacks

from .config import ACTIONS, TEACHER_QUERY_COUNT, TEACHER_SEED


class _QuietLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


@dataclass(frozen=True)
class TeacherVote:
    action: object
    confidence: float
    votes: tuple


class RuleTeacher:
    """Query independent rule-policy clones without changing learner RNG state."""

    def __init__(self, seed=TEACHER_SEED):
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        self.ignore_others_timer = 0
        self.current_round = 0
        self.rng = np.random.default_rng(seed)
        self.logger = _QuietLogger()

    def _reset_round(self):
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        self.ignore_others_timer = 0

    def _clone_before_observation(self):
        return SimpleNamespace(
            bomb_history=deque(self.bomb_history, 5),
            coordinate_history=deque(self.coordinate_history, 20),
            ignore_others_timer=int(self.ignore_others_timer),
            current_round=int(self.current_round),
            logger=self.logger,
        )

    def _advance_canonical_history(self, game_state):
        round_number = int(game_state["round"])
        if round_number != self.current_round:
            self._reset_round()
            self.current_round = round_number

        position = tuple(game_state["self"][3])
        if self.coordinate_history.count(position) > 2:
            self.ignore_others_timer = 5
        else:
            self.ignore_others_timer -= 1
        self.coordinate_history.append(position)

    def query(self, game_state):
        """Return a majority vote from clones sharing one pre-query history."""
        actions = []
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            for _ in range(TEACHER_QUERY_COUNT):
                clone = self._clone_before_observation()
                py_seed = int(self.rng.integers(0, 2**32 - 1))
                np_seed = int(self.rng.integers(0, 2**32 - 1))
                random.seed(py_seed)
                np.random.seed(np_seed)
                action = rule_callbacks.act(clone, game_state)
                # The official rule agent can exhaust its proposal queue and
                # implicitly return None. That query is an abstention, not WAIT.
                if action is not None and action not in ACTIONS:
                    raise RuntimeError("Rule teacher returned an unknown action")
                actions.append(action)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

        # The canonical history follows the learner's observation exactly once.
        # Teacher-recommended bombs stay inside their temporary clones.
        self._advance_canonical_history(game_state)
        counts = Counter(action for action in actions if action in ACTIONS)
        votes = tuple(counts.get(candidate, 0) for candidate in ACTIONS)
        if not counts:
            return TeacherVote(None, 0.0, votes)
        action, count = max(
            counts.items(),
            key=lambda item: (item[1], -ACTIONS.index(item[0])),
        )
        return TeacherVote(action, count / float(TEACHER_QUERY_COUNT), votes)

    def commit_learner_action(self, game_state, action):
        """Apply only the action that the learner really sent to the game."""
        if action == "BOMB":
            self.bomb_history.append(tuple(game_state["self"][3]))

    def state_dict(self):
        return {
            "bomb_history": list(self.bomb_history),
            "coordinate_history": list(self.coordinate_history),
            "ignore_others_timer": self.ignore_others_timer,
            "current_round": self.current_round,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        self.bomb_history = deque(state["bomb_history"], 5)
        self.coordinate_history = deque(state["coordinate_history"], 20)
        self.ignore_others_timer = int(state["ignore_others_timer"])
        self.current_round = int(state["current_round"])
        self.rng.bit_generator.state = state["rng_state"]
