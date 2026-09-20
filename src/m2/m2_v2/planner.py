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
    shortest_path_distance,
    valid_action_mask,
)


BOMB_INDEX = ACTIONS.index("BOMB")
WAIT_INDEX = ACTIONS.index("WAIT")
MOVE_INDICES = tuple(range(4))
NEW_BOMB_DETONATION_HORIZON = BOMB_TIMER + 1
MIN_BOMB_ESCAPE_DIVERSITY = 0.5
COIN_HORIZON = 999
CHASE_RANGE = 6
MAX_FOES_TO_FARM = 0
MIN_CRATES = 1
FARM_DIVERSITY = 0.5
COIN_CONTEST = 0
BOMB_VETO = 1
VETO_FOE_RANGE = 4
CHASE_LIMIT = 999


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


def _approach(game_state, allowed, q_values, rng, targets, horizon):
    """Pick a survivable move that strictly shortens the path to any target."""
    if not targets:
        return None
    current = shortest_path_distance(game_state, targets)
    if current is None or current > horizon:
        return None
    own_position = tuple(game_state["self"][3])
    me = tuple(game_state["self"])
    candidates = np.zeros(len(ACTIONS), dtype=bool)
    for index in MOVE_INDICES:
        if not allowed[index]:
            continue
        dx, dy = MOVE_DELTAS[ACTIONS[index]]
        destination = (own_position[0] + dx, own_position[1] + dy)
        moved_state = dict(game_state)
        moved_state["self"] = (*me[:3], destination)
        distance = shortest_path_distance(moved_state, targets)
        if distance is not None and distance < current:
            candidates[index] = True
    if not np.any(candidates):
        return None
    return _q_tiebreak(candidates, q_values, rng)


def _uncontested(game_state, coins):
    """Drop coins that an opponent would reach strictly sooner than we would."""
    own_distance = {}
    for coin in coins:
        distance = shortest_path_distance(game_state, [coin])
        if distance is not None:
            own_distance[coin] = distance
    if not own_distance:
        return []
    keep = []
    for coin, mine in own_distance.items():
        contested = False
        for opponent in game_state["others"]:
            probe = dict(game_state)
            probe["self"] = (*tuple(game_state["self"])[:3], tuple(opponent[3]))
            probe["others"] = [item for item in game_state["others"] if item is not opponent]
            theirs = shortest_path_distance(probe, [coin])
            if theirs is not None and theirs < mine:
                contested = True
                break
        if not contested:
            keep.append(coin)
    return keep


def coin_push(game_state, allowed, q_values, rng) -> int | None:
    """Prefer a survivable move that gets strictly closer to the nearest coin."""
    coins = [tuple(coin) for coin in game_state["coins"]]
    if not coins:
        return None
    if COIN_CONTEST:
        uncontested = _uncontested(game_state, coins)
        if uncontested:
            coins = uncontested
    return _approach(game_state, allowed, q_values, rng, coins, COIN_HORIZON)


def crate_push(game_state, allowed, q_values, rng, diversity) -> int | None:
    """Bomb a worthwhile crate cluster, or walk towards the nearest crate."""
    field = np.asarray(game_state["field"])
    crates = {tuple(position) for position in np.argwhere(field == 1)}
    if not crates:
        return None
    own_position = tuple(game_state["self"][3])
    if allowed[BOMB_INDEX] and diversity[BOMB_INDEX] >= FARM_DIVERSITY:
        blast = set(blast_coordinates(field, own_position))
        if len(blast & crates) >= MIN_CRATES:
            return BOMB_INDEX
    approach_tiles = set()
    for x, y in crates:
        for dx, dy in MOVE_DELTAS.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 0:
                approach_tiles.add((nx, ny))
    return _approach(game_state, allowed, q_values, rng, list(approach_tiles), 999)


def foe_within(game_state, limit) -> bool:
    """True when the closest reachable opponent is no further than limit."""
    opponents = [tuple(item[3]) for item in game_state["others"]]
    if not opponents:
        return False
    distance = shortest_path_distance(game_state, opponents)
    return distance is not None and distance <= limit


def bomb_is_pointless(game_state) -> bool:
    """A bomb that reaches neither a crate nor a nearby opponent is wasted."""
    field = np.asarray(game_state["field"])
    own_position = tuple(game_state["self"][3])
    blast = set(blast_coordinates(field, own_position))
    if any(field[tile] == 1 for tile in blast):
        return False
    opponents = [tuple(item[3]) for item in game_state["others"]]
    if not opponents:
        return True
    distance = shortest_path_distance(game_state, opponents)
    return distance is None or distance > VETO_FOE_RANGE


def may_farm(game_state) -> bool:
    """Farming is allowed only in a thin field with no opponent nearby."""
    opponents = [tuple(item[3]) for item in game_state["others"]]
    if len(opponents) > MAX_FOES_TO_FARM:
        return False
    if not opponents:
        return True
    distance = shortest_path_distance(game_state, opponents)
    return distance is None or distance > CHASE_RANGE


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

    coin_move = coin_push(game_state, allowed, q_values, rng)
    if coin_move is not None:
        return coin_move

    if may_farm(game_state):
        crate_move = crate_push(game_state, allowed, q_values, rng, diversity)
        if crate_move is not None:
            return crate_move

    if not foe_within(game_state, CHASE_LIMIT):
        return None

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
    if BOMB_VETO and restricted[BOMB_INDEX] and bomb_is_pointless(game_state):
        vetoed = restricted.copy()
        vetoed[BOMB_INDEX] = False
        if np.any(vetoed):
            restricted = vetoed
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
