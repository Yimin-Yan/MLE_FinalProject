"""Known-hazard time planner used for labels and safe exploration."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .config import (
    ACTIONS,
    BOMB_POWER,
    BOMB_TIMER,
    DEAD_END_DEPTH_CAP,
    EXPLOSION_DANGER_STEPS,
    TEMPORAL_HORIZON,
)


MOVE_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass
class TemporalAnalysis:
    legal_mask: np.ndarray
    escape_feasible: np.ndarray
    danger_arrival: np.ndarray
    safe_continuation: np.ndarray
    escape_slack: np.ndarray
    global_features: np.ndarray
    known_hazard_present: bool


@dataclass
class _Timeline:
    field: np.ndarray
    bombs: tuple
    explosion_times: tuple
    dangers: dict
    crate_destroyed_at: dict
    last_danger_step: int


class KnownHazardPlanner:
    """Compute one temporal analysis per observation and cache that result."""

    def __init__(self):
        self._cache_key = None
        self._cache_fingerprint = None
        self._cache_value = None
        self.requests = 0
        self.computations = 0
        self.cache_hits = 0

    @staticmethod
    def state_key(game_state):
        if game_state is None:
            return None
        x, y = game_state["self"][3]
        return int(game_state["round"]), int(game_state["step"]), int(x), int(y)

    def analyze(self, game_state, need_escape_labels=True):
        if game_state is None:
            raise ValueError("The temporal planner needs a live game state")

        self.requests += 1
        key = self.state_key(game_state), bool(need_escape_labels)
        fingerprint = self._fingerprint(game_state)
        if key == self._cache_key and fingerprint == self._cache_fingerprint:
            self.cache_hits += 1
            return self._cache_value

        # The framework keeps the same step number before and after an action.
        # A changed fingerprint is therefore a new observation, not stale data.
        analysis = self._compute(game_state, bool(need_escape_labels))
        self.computations += 1
        self._cache_key = key
        self._cache_fingerprint = fingerprint
        self._cache_value = analysis
        return analysis

    @staticmethod
    def _fingerprint(game_state):
        field = np.asarray(game_state["field"], dtype=np.int8)
        explosion = np.asarray(game_state["explosion_map"], dtype=np.float32)
        bombs = tuple(
            sorted((int(x), int(y), int(timer)) for (x, y), timer in game_state["bombs"])
        )
        others = tuple(
            sorted((int(other[3][0]), int(other[3][1])) for other in game_state["others"])
        )
        own = game_state["self"]
        return (
            field.tobytes(),
            explosion.tobytes(),
            bombs,
            others,
            bool(own[2]),
            int(own[3][0]),
            int(own[3][1]),
        )

    def _compute(self, game_state, need_escape_labels):
        field = np.asarray(game_state["field"], dtype=np.int16)
        explosion_map = np.asarray(game_state["explosion_map"], dtype=np.float32)
        _, _, bombs_left, (x, y) = game_state["self"]
        start = (int(x), int(y))
        bombs = tuple(
            ((int(position[0]), int(position[1])), int(timer))
            for position, timer in game_state["bombs"]
        )
        others = {tuple(map(int, other[3])) for other in game_state["others"]}
        legal_mask = _legal_mask(field, start, bombs, others, bool(bombs_left))

        current_hazard = bool(bombs) or bool(np.any(explosion_map > 0))
        escape = np.zeros(len(ACTIONS), dtype=np.bool_)
        arrival = np.zeros(len(ACTIONS), dtype=np.float32)
        continuation = np.zeros(len(ACTIONS), dtype=np.float32)
        slack = np.zeros(len(ACTIONS), dtype=np.float32)

        base_timeline = self._make_timeline(field, explosion_map, bombs)
        for action_index, action in enumerate(ACTIONS):
            if not legal_mask[action_index]:
                continue
            timeline = base_timeline
            if action == "BOMB":
                new_bombs = bombs + ((start, BOMB_TIMER),)
                timeline = self._make_timeline(field, explosion_map, new_bombs)

            first_position = _position_after_action(start, action)
            if need_escape_labels:
                escape[action_index] = self._has_escape_path(first_position, timeline)
            danger_step = _first_danger_step(first_position, timeline)
            if danger_step is None:
                arrival[action_index] = 1.0
                slack[action_index] = 1.0
            else:
                # Dividing by H leaves a separate value for "no danger".
                arrival[action_index] = (danger_step - 1) / float(TEMPORAL_HORIZON)
                distance = self._static_escape_distance(first_position, timeline)
                if distance is None:
                    slack[action_index] = -1.0
                else:
                    raw_slack = (danger_step - distance - 1) / float(TEMPORAL_HORIZON)
                    slack[action_index] = float(np.clip(raw_slack, -1.0, 1.0))

            continuation[action_index] = self._safe_continuation_count(
                first_position,
                timeline,
            ) / 5.0

        min_timer = min((timer for _, timer in bombs), default=None)
        dead_end_depth = _dead_end_depth(field, start)
        global_features = np.asarray(
            (
                float(current_hazard),
                min(len(bombs), 4) / 4.0,
                1.0 if min_timer is None else float(np.clip(min_timer / 4.0, 0.0, 1.0)),
                min(dead_end_depth, DEAD_END_DEPTH_CAP) / float(DEAD_END_DEPTH_CAP),
            ),
            dtype=np.float32,
        )
        return TemporalAnalysis(
            legal_mask=legal_mask,
            escape_feasible=escape,
            danger_arrival=arrival,
            safe_continuation=continuation,
            escape_slack=slack,
            global_features=global_features,
            known_hazard_present=current_hazard,
        )

    def _make_timeline(self, field, explosion_map, bombs):
        dangers = {step: set() for step in range(1, TEMPORAL_HORIZON + 1)}
        for px, py in np.argwhere(explosion_map > 0):
            remaining = max(1, int(math.ceil(float(explosion_map[px, py]))))
            for step in range(1, min(remaining, TEMPORAL_HORIZON) + 1):
                dangers[step].add((int(px), int(py)))

        explosion_times = []
        crate_destroyed_at = {}
        for position, timer in bombs:
            if not 0 <= int(timer) <= BOMB_TIMER:
                raise ValueError("Observed bomb timer is outside the official range")
            explosion_step = int(timer) + 1
            explosion_times.append(explosion_step)
            blast = _blast_cells(field, position)
            for offset in range(EXPLOSION_DANGER_STEPS):
                danger_step = explosion_step + offset
                if danger_step <= TEMPORAL_HORIZON:
                    dangers[danger_step].update(blast)
            for cell in blast:
                if field[cell] == 1:
                    previous = crate_destroyed_at.get(cell)
                    if previous is None or explosion_step < previous:
                        crate_destroyed_at[cell] = explosion_step

        used_steps = [step for step, cells in dangers.items() if cells]
        last_danger = max(used_steps, default=0)
        if last_danger > TEMPORAL_HORIZON:
            raise RuntimeError("Known hazard extends beyond the planner horizon")
        return _Timeline(
            field=field,
            bombs=tuple(position for position, _ in bombs),
            explosion_times=tuple(explosion_times),
            dangers=dangers,
            crate_destroyed_at=crate_destroyed_at,
            last_danger_step=last_danger,
        )

    def _has_escape_path(self, first_position, timeline):
        if first_position in timeline.dangers.get(1, set()):
            return False
        positions = {first_position}
        for step in range(2, timeline.last_danger_step + 1):
            next_positions = set()
            for position in positions:
                for candidate in _next_positions(position, step, timeline):
                    if candidate not in timeline.dangers.get(step, set()):
                        next_positions.add(candidate)
            if not next_positions:
                return False
            positions = next_positions
        return True

    @staticmethod
    def _safe_continuation_count(position, timeline):
        if position in timeline.dangers.get(1, set()):
            return 0
        safe = 0
        for candidate in _next_positions(position, 2, timeline):
            if candidate not in timeline.dangers.get(2, set()):
                safe += 1
        return safe

    @staticmethod
    def _static_escape_distance(start, timeline):
        future_danger = set()
        for cells in timeline.dangers.values():
            future_danger.update(cells)

        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            position, distance = queue.popleft()
            if position not in future_danger:
                return distance
            for candidate in _movement_targets(position, 2, timeline):
                if candidate not in visited:
                    visited.add(candidate)
                    queue.append((candidate, distance + 1))
        return None


def _legal_mask(field, start, bombs, other_positions, bombs_left):
    mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    bomb_positions = {position for position, _ in bombs}
    for action_index, (dx, dy) in enumerate(MOVE_DELTAS):
        target = (start[0] + dx, start[1] + dy)
        if _inside(field, target):
            mask[action_index] = (
                field[target] == 0
                and target not in bomb_positions
                and target not in other_positions
            )
    mask[4] = True
    mask[5] = bool(bombs_left)
    return mask


def _position_after_action(position, action):
    if action in ("WAIT", "BOMB"):
        return position
    action_index = ACTIONS.index(action)
    dx, dy = MOVE_DELTAS[action_index]
    return position[0] + dx, position[1] + dy


def _blast_cells(field, position):
    x, y = position
    cells = [(x, y)]
    for dx, dy in MOVE_DELTAS:
        for distance in range(1, BOMB_POWER + 1):
            cell = (x + dx * distance, y + dy * distance)
            if not _inside(field, cell) or field[cell] == -1:
                break
            # This framework clears crates but does not stop the blast at them.
            cells.append(cell)
    return tuple(cells)


def _next_positions(position, step, timeline):
    positions = [position]
    positions.extend(_movement_targets(position, step, timeline))
    return positions


def _movement_targets(position, step, timeline):
    targets = []
    for dx, dy in MOVE_DELTAS:
        target = position[0] + dx, position[1] + dy
        if _tile_is_free_at(target, step, timeline):
            targets.append(target)
    return targets


def _tile_is_free_at(position, step, timeline):
    if not _inside(timeline.field, position):
        return False
    tile = int(timeline.field[position])
    if tile == -1:
        return False
    if tile == 1:
        destroyed_at = timeline.crate_destroyed_at.get(position)
        if destroyed_at is None or destroyed_at >= step:
            return False
    for bomb_position, explosion_step in zip(timeline.bombs, timeline.explosion_times):
        if position == bomb_position and step <= explosion_step:
            return False
    return True


def _first_danger_step(position, timeline):
    for step in range(1, TEMPORAL_HORIZON + 1):
        if position in timeline.dangers.get(step, set()):
            return step
    return None


def _dead_end_depth(field, start):
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        position, distance = queue.popleft()
        degree = 0
        neighbours = []
        for dx, dy in MOVE_DELTAS:
            neighbour = position[0] + dx, position[1] + dy
            if _inside(field, neighbour) and field[neighbour] == 0:
                degree += 1
                neighbours.append(neighbour)
        if degree >= 3:
            return min(distance, DEAD_END_DEPTH_CAP)
        if distance >= DEAD_END_DEPTH_CAP:
            continue
        for neighbour in neighbours:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
    return DEAD_END_DEPTH_CAP


def _inside(field, position):
    x, y = position
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]
