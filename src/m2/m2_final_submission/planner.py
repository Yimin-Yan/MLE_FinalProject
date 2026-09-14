"""Exact-timing survival and trap-only tactical decisions for V19."""

import numpy as np

from .config import (
    ACTIONS,
    BOMB_TIMER,
    MOVE_DELTAS,
)
from .features import (
    blast_coordinates,
    danger_urgency_map,
    robust_action_features,
    valid_action_mask,
)


BOMB_INDEX = ACTIONS.index("BOMB")
WAIT_INDEX = ACTIONS.index("WAIT")
MOVE_INDICES = tuple(range(4))
NEW_BOMB_DETONATION_HORIZON = BOMB_TIMER + 1
MIN_BOMB_ESCAPE_DIVERSITY = 0.5


def exact_survival_metrics(game_state: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return per-action survival and terminal escape diversity.

    Existing state bombs with timer ``t`` explode after ``t + 1`` actions. A
    bomb placed by the candidate action is created at timer four and therefore
    explodes after five actions, including the placement action.
    """
    physical = valid_action_mask(game_state)
    survivable = np.zeros(len(ACTIONS), dtype=bool)
    diversity = np.zeros(len(ACTIONS), dtype=np.float32)
    for index, action in enumerate(ACTIONS):
        if physical[index]:
            survivable[index], diversity[index] = _action_survival(game_state, action)
    return survivable, diversity


def _action_survival(game_state: dict, action: str) -> tuple[bool, float]:
    field = np.asarray(game_state["field"])
    start = tuple(game_state["self"][3])
    opponents = {tuple(opponent[3]) for opponent in game_state["others"]}
    bombs = {tuple(position): int(timer) + 1 for position, timer in game_state["bombs"]}

    if action == "BOMB":
        bombs[start] = NEW_BOMB_DETONATION_HORIZON
        first = start
        left_start = False
    elif action == "WAIT":
        first = start
        left_start = start not in bombs
    else:
        dx, dy = MOVE_DELTAS[action]
        first = (start[0] + dx, start[1] + dy)
        left_start = True

    horizon = max([1, *bombs.values()])
    blasts = {time: set() for time in range(1, horizon + 1)}
    for position, time in bombs.items():
        blasts[time].update(blast_coordinates(field, position))
    current_explosion = np.asarray(game_state["explosion_map"]) > 0
    blasts[1].update(map(tuple, np.argwhere(current_explosion)))
    if first in blasts[1]:
        return False, 0.0

    frontier = {(first, left_start)}
    if horizon == 1:
        return True, 1.0
    for time in range(2, horizon + 1):
        active_bombs = {
            position for position, explosion_time in bombs.items()
            if explosion_time >= time
        }
        following = set()
        for position, has_left_start in frontier:
            for dx, dy in (*MOVE_DELTAS.values(), (0, 0)):
                nxt = (position[0] + dx, position[1] + dy)
                if not (0 <= nxt[0] < field.shape[0] and 0 <= nxt[1] < field.shape[1]):
                    continue
                if field[nxt] != 0 or nxt in opponents or nxt in blasts[time]:
                    continue
                next_left = has_left_start or nxt != start
                if nxt in active_bombs and not (nxt == start and not next_left):
                    continue
                following.add((nxt, next_left))
        if not following:
            return False, 0.0
        frontier = following

    terminal_positions = {position for position, _ in frontier}
    return True, float(min(len(terminal_positions) / 4.0, 1.0))


def exact_bomb_pressure(game_state: dict) -> tuple[bool, float]:
    """Return strict-trap status and maximum opponent restriction for BOMB now."""
    opponents = [tuple(item[3]) for item in game_state["others"]]
    if not opponents or not bool(game_state["self"][2]):
        return False, 0.0
    field = np.asarray(game_state["field"])
    own_position = tuple(game_state["self"][3])
    bomb_times = {
        tuple(position): int(timer) + 1 for position, timer in game_state["bombs"]
    }
    if own_position in bomb_times:
        return False, 0.0
    bomb_times[own_position] = NEW_BOMB_DETONATION_HORIZON

    schedule: dict[int, set[tuple[int, int]]] = {}
    for position, detonation in bomb_times.items():
        schedule.setdefault(detonation, set()).update(blast_coordinates(field, position))
    current_explosion = np.asarray(game_state["explosion_map"]) > 0
    if np.any(current_explosion):
        schedule.setdefault(1, set()).update(map(tuple, np.argwhere(current_explosion)))
    horizon = max(schedule, default=1)

    trapped = False
    maximum_restriction = 0.0
    for start in opponents:
        opponent_trapped, restriction = _opponent_pressure(
            field, start, schedule, bomb_times, horizon
        )
        trapped = trapped or opponent_trapped
        maximum_restriction = max(maximum_restriction, restriction)
    return trapped, float(np.clip(maximum_restriction, 0.0, 1.0))


def _opponent_pressure(field, start, schedule, bomb_times, horizon):
    frontier = {start}
    restriction = 0.0
    targeted = False
    for time in range(1, horizon + 1):
        active_bombs = {
            position for position, detonation in bomb_times.items()
            if detonation >= time
        }
        candidates = set()
        for position in frontier:
            for dx, dy in ((0, 0), *MOVE_DELTAS.values()):
                nxt = (position[0] + dx, position[1] + dy)
                if not (0 <= nxt[0] < field.shape[0] and 0 <= nxt[1] < field.shape[1]):
                    continue
                if field[nxt] != 0:
                    continue
                if nxt in active_bombs and nxt != position:
                    continue
                candidates.add(nxt)
        if not candidates:
            return targeted, max(restriction, float(targeted))
        affected = candidates & schedule.get(time, set())
        if affected:
            targeted = True
            restriction = max(restriction, len(affected) / len(candidates))
        frontier = candidates - affected
        if not frontier:
            return targeted, max(restriction, float(targeted))
    return False, float(np.clip(restriction, 0.0, 1.0))


def tactical_action(game_state, legal, q_values, rng) -> int | None:
    """Choose an exact-timing tactical action, or return None for Q fallback."""
    if legal.shape != (len(ACTIONS),) or q_values.shape != (len(ACTIONS),):
        raise ValueError("Unexpected action-vector shape")
    survivable, diversity = exact_survival_metrics(game_state)
    allowed = legal & survivable
    if not np.any(allowed):
        return None

    own_position = tuple(game_state["self"][3])
    in_danger = float(danger_urgency_map(game_state)[own_position]) > 0.0
    if in_danger:
        escape = allowed.copy()
        escape[BOMB_INDEX] = False
        if np.any(escape):
            best_diversity = float(np.max(diversity[escape]))
            candidates = escape & (diversity == best_diversity)
            return _q_tiebreak(candidates, q_values, rng)

    trapped, _ = exact_bomb_pressure(game_state)
    if (
        allowed[BOMB_INDEX]
        and diversity[BOMB_INDEX] >= MIN_BOMB_ESCAPE_DIVERSITY
        and trapped
    ):
        return BOMB_INDEX

    tactical = robust_action_features(game_state).reshape(len(ACTIONS), -1)
    progress = tactical[:, 0]
    movement = np.zeros(len(ACTIONS), dtype=bool)
    movement[list(MOVE_INDICES)] = allowed[list(MOVE_INDICES)]
    if np.any(movement):
        best_progress = float(np.max(progress[movement]))
        if best_progress > 0.0:
            candidates = movement & (progress == best_progress)
            return _q_tiebreak(candidates, q_values, rng)
    return None


def exact_action_mask(game_state: dict) -> tuple[np.ndarray, np.ndarray]:
    """Restrict physical actions to exact-survivable actions when available."""
    physical = valid_action_mask(game_state)
    survivable, diversity = exact_survival_metrics(game_state)
    restricted = physical & survivable
    if np.any(restricted):
        return restricted, diversity
    fallback = physical.copy()
    if np.any(fallback[:BOMB_INDEX]):
        fallback[BOMB_INDEX] = False
    return fallback, diversity


def _q_tiebreak(candidates, q_values, rng) -> int:
    masked = np.where(candidates, q_values, -np.inf)
    best = np.flatnonzero(candidates & (masked == np.max(masked)))
    return int(rng.choice(best))
