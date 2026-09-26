"""Self-contained board representation and physical action mask for M2."""

from collections import deque

import numpy as np

from .config import (
    ACTIONS,
    ACTION_FEATURES_PER_ACTION,
    BASE_AUX_FEATURES,
    BOARD_CHANNELS,
    BOARD_SIZE,
    BOMB_POWER,
    BOMB_TIMER,
    MOVE_DELTAS,
)


def state_to_features(game_state: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    if game_state is None:
        return None
    field = np.asarray(game_state["field"])
    if field.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected a {BOARD_SIZE}x{BOARD_SIZE} board, got {field.shape}")
    board = np.zeros((BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    board[0] = field == -1
    board[1] = field == 1
    board[2] = field == 0
    for x, y in game_state["coins"]:
        board[3, x, y] = 1.0
    own_position = tuple(game_state["self"][3])
    board[4, own_position[0], own_position[1]] = 1.0
    for opponent in game_state["others"]:
        x, y = opponent[3]
        board[5, x, y] = 1.0
    for (x, y), timer in game_state["bombs"]:
        board[6, x, y] = 1.0 - np.clip(timer / BOMB_TIMER, 0.0, 1.0)
    board[7] = np.clip(np.asarray(game_state["explosion_map"], dtype=np.float32), 0, 2) / 2
    board[8] = danger_urgency_map(game_state)
    for coordinate in blast_coordinates(field, own_position):
        board[9, coordinate[0], coordinate[1]] = 1.0

    playable = max(int(np.count_nonzero(field != -1)), 1)
    coins = [tuple(item) for item in game_state["coins"]]
    opponents = [tuple(item[3]) for item in game_state["others"]]
    coin_distance, coin_first = path_guidance(game_state, coins)
    opponent_distance, opponent_first = path_guidance(game_state, opponents)
    scale = max(sum(field.shape) - 2, 1)
    danger = danger_urgency_map(game_state)
    legal = valid_action_mask(game_state)
    safe = np.zeros(5, dtype=np.float32)
    for index, action in enumerate(ACTIONS[:5]):
        destination = own_position if action == "WAIT" else (
            own_position[0] + MOVE_DELTAS[action][0],
            own_position[1] + MOVE_DELTAS[action][1],
        )
        safe[index] = float(legal[index] and danger[destination] == 0)
    prospective_blast = set(blast_coordinates(field, own_position))
    escape_exists, escape_first = bomb_escape_guidance(game_state)
    aux = np.asarray(
        [
            float(game_state["self"][2]),
            np.clip(game_state.get("step", 0) / 400.0, 0.0, 1.0),
            np.tanh(float(game_state["self"][1]) / 10.0),
            len(opponents) / 3.0,
            min(len(coins) / 50.0, 1.0),
            np.count_nonzero(field == 1) / playable,
            float(bool(coins)),
            1.0 if coin_distance is None else min(coin_distance / scale, 1.0),
            *coin_first,
            float(bool(opponents)),
            1.0 if opponent_distance is None else min(opponent_distance / scale, 1.0),
            *opponent_first,
            float(danger[own_position]),
            *safe,
            min(sum(field[item] == 1 for item in prospective_blast) / 12.0, 1.0),
            min(sum(item in prospective_blast for item in opponents) / 3.0, 1.0),
            float(escape_exists),
            *escape_first,
        ],
        dtype=np.float32,
    )
    if aux.shape != (BASE_AUX_FEATURES,):
        raise RuntimeError(f"Expected {BASE_AUX_FEATURES} base auxiliary features, got {aux.shape}")
    action_features = robust_action_features(game_state)
    return _center_board(board, own_position), np.concatenate((aux, action_features))


def robust_action_features(game_state: dict) -> np.ndarray:
    """Seven bounded tactical features per action, flattened action-major.

    The representation separates partial blast coverage from strict trapping,
    measures the diversity of the agent's own time-expanded escape frontier,
    and exposes their conjunction for BOMB instead of forcing the network to
    infer safety from a single Boolean.
    """
    field = np.asarray(game_state["field"])
    own_position = tuple(game_state["self"][3])
    opponents = [tuple(item[3]) for item in game_state["others"]]
    physical = valid_action_mask(game_state)
    survivable, escape_diversity = survival_action_metrics(game_state)
    scale = max(sum(field.shape) - 2, 1)
    current_distance = shortest_path_distance(game_state, opponents)
    bomb_ready = bool(game_state["self"][2])
    values = np.zeros((len(ACTIONS), ACTION_FEATURES_PER_ACTION), dtype=np.float32)

    for index, action in enumerate(ACTIONS):
        if not physical[index]:
            continue
        destination = own_position
        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            destination = (own_position[0] + dx, own_position[1] + dy)

        if opponents and current_distance is not None:
            moved_state = dict(game_state)
            me = tuple(game_state["self"])
            moved_state["self"] = (*me[:3], destination)
            next_distance = shortest_path_distance(moved_state, opponents)
            if next_distance is not None:
                values[index, 0] = np.clip(
                    (current_distance - next_distance) / scale, -1.0, 1.0
                )

        if opponents and bomb_ready:
            future_blast = set(blast_coordinates(field, destination))
            values[index, 1] = sum(position in future_blast for position in opponents) / len(opponents)
        values[index, 2] = float(survivable[index])
        values[index, 3] = escape_diversity[index]

    bomb_index = ACTIONS.index("BOMB")
    if physical[bomb_index] and survivable[bomb_index] and opponents:
        hypothetical = dict(game_state)
        hypothetical["bombs"] = [*game_state["bombs"], (own_position, BOMB_TIMER - 1)]
        any_trapped, maximum_restriction = scheduled_opponent_pressure(hypothetical)
        values[bomb_index, 4] = any_trapped
        values[bomb_index, 5] = maximum_restriction
        values[bomb_index, 6] = any_trapped * escape_diversity[bomb_index]

    return values.reshape(-1)


def scheduled_opponent_pressure(game_state: dict) -> tuple[float, float]:
    """Return any-strict-trap and maximum restriction in [0, 1]."""
    opponents = [tuple(item[3]) for item in game_state["others"]]
    if not opponents or not game_state["bombs"]:
        return 0.0, 0.0
    field = np.asarray(game_state["field"])
    schedule = {}
    bomb_times = {}
    for position, timer in game_state["bombs"]:
        coordinate = tuple(position)
        detonation = max(int(timer) + 1, 1)
        bomb_times[coordinate] = min(detonation, bomb_times.get(coordinate, BOMB_TIMER + 1))
        schedule.setdefault(detonation, set()).update(blast_coordinates(field, coordinate))
    current_explosion = np.asarray(game_state["explosion_map"]) > 0
    if np.any(current_explosion):
        schedule.setdefault(1, set()).update(map(tuple, np.argwhere(current_explosion)))
    horizon = max(schedule, default=1)

    any_trapped = False
    maximum_restriction = 0.0
    for start in opponents:
        trapped, restriction = _scheduled_pressure_for_start(
            field, start, schedule, bomb_times, horizon
        )
        any_trapped = any_trapped or trapped
        maximum_restriction = max(maximum_restriction, restriction)
    return float(any_trapped), float(np.clip(maximum_restriction, 0.0, 1.0))


def _scheduled_pressure_for_start(field, start, schedule, bomb_times, horizon):
    frontier = {start}
    restriction = 0.0
    targeted = False
    for time in range(1, horizon + 1):
        candidates = set()
        active_bombs = {position for position, detonation in bomb_times.items() if detonation >= time}
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
    return False, restriction


def _center_board(board: np.ndarray, own_position: tuple[int, int]) -> np.ndarray:
    """Translate the board without toroidal wrap; outside space is wall."""
    center = BOARD_SIZE // 2
    shift_x, shift_y = center - own_position[0], center - own_position[1]
    centered = np.roll(board, shift=(shift_x, shift_y), axis=(1, 2))
    invalid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    if shift_x > 0:
        invalid[:shift_x, :] = True
    elif shift_x < 0:
        invalid[shift_x:, :] = True
    if shift_y > 0:
        invalid[:, :shift_y] = True
    elif shift_y < 0:
        invalid[:, shift_y:] = True
    centered[:, invalid] = 0.0
    centered[0, invalid] = 1.0
    return centered


def valid_action_mask(game_state: dict) -> np.ndarray:
    field = np.asarray(game_state["field"])
    x, y = game_state["self"][3]
    blocked = {tuple(position) for position, _ in game_state["bombs"]}
    blocked.update(tuple(opponent[3]) for opponent in game_state["others"])
    mask = np.zeros(len(ACTIONS), dtype=bool)
    for index, action in enumerate(ACTIONS[:4]):
        dx, dy = MOVE_DELTAS[action]
        target = (x + dx, y + dy)
        mask[index] = (
            0 <= target[0] < field.shape[0]
            and 0 <= target[1] < field.shape[1]
            and field[target] == 0
            and target not in blocked
        )
    mask[4] = True
    mask[5] = bool(game_state["self"][2]) and (x, y) not in blocked
    return mask


def curriculum_action_mask(game_state: dict, stage: str) -> np.ndarray:
    mask = valid_action_mask(game_state)
    if stage == "task1":
        mask[5] = False
        return mask
    survivable = survival_action_mask(game_state)
    restricted = mask & survivable
    return restricted if np.any(restricted) else mask


def survival_action_mask(game_state: dict) -> np.ndarray:
    """Return actions with at least one path through all scheduled blasts."""
    return survival_action_metrics(game_state)[0]


def survival_action_metrics(game_state: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return survival availability and normalized terminal-path diversity."""
    physical = valid_action_mask(game_state)
    result = np.zeros(len(ACTIONS), dtype=bool)
    diversity = np.zeros(len(ACTIONS), dtype=np.float32)
    for index, action in enumerate(ACTIONS):
        if physical[index]:
            result[index], diversity[index] = _action_survival_metrics(game_state, action)
    return result, diversity


def _action_has_survival_path(game_state: dict, action: str) -> bool:
    return _action_survival_metrics(game_state, action)[0]


def _action_survival_metrics(game_state: dict, action: str) -> tuple[bool, float]:
    field = np.asarray(game_state["field"])
    start = tuple(game_state["self"][3])
    opponents = {tuple(opponent[3]) for opponent in game_state["others"]}
    bombs = {tuple(position): int(timer) + 1 for position, timer in game_state["bombs"]}
    if action == "BOMB":
        bombs[start] = BOMB_TIMER
        first = start
        left_start = False
    elif action == "WAIT":
        first = start
        left_start = start not in bombs
    else:
        dx, dy = MOVE_DELTAS[action]
        first = (start[0] + dx, start[1] + dy)
        left_start = first != start

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
        active_bombs = {position for position, explosion_time in bombs.items() if explosion_time >= time}
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
    diversity = min(len(terminal_positions) / 4.0, 1.0)
    return bool(frontier), float(diversity)


def blast_coordinates(field: np.ndarray, origin: tuple[int, int]):
    yield origin
    for dx, dy in MOVE_DELTAS.values():
        for distance in range(1, BOMB_POWER + 1):
            coordinate = (origin[0] + dx * distance, origin[1] + dy * distance)
            if not (0 <= coordinate[0] < field.shape[0] and 0 <= coordinate[1] < field.shape[1]):
                break
            if field[coordinate] == -1:
                break
            yield coordinate


def danger_urgency_map(game_state: dict) -> np.ndarray:
    field = np.asarray(game_state["field"])
    urgency = np.clip(np.asarray(game_state["explosion_map"], dtype=np.float32), 0, 1)
    for position, timer in game_state["bombs"]:
        value = 1.0 - np.clip(int(timer) / (BOMB_TIMER + 1.0), 0.0, 1.0)
        for coordinate in blast_coordinates(field, tuple(position)):
            urgency[coordinate] = max(float(urgency[coordinate]), float(value))
    return urgency.astype(np.float32, copy=False)


def shortest_path_distance(game_state: dict, targets: list[tuple[int, int]]) -> int | None:
    if not targets:
        return None
    field = np.asarray(game_state["field"])
    start = tuple(game_state["self"][3])
    blocked = {tuple(position) for position, _ in game_state["bombs"]}
    blocked.update(tuple(opponent[3]) for opponent in game_state["others"])
    targets = set(targets)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        position, distance = queue.popleft()
        if position in targets:
            return distance
        for dx, dy in MOVE_DELTAS.values():
            nxt = (position[0] + dx, position[1] + dy)
            if (
                0 <= nxt[0] < field.shape[0]
                and 0 <= nxt[1] < field.shape[1]
                and field[nxt] == 0
                and (nxt not in blocked or nxt in targets)
                and nxt not in seen
            ):
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def path_guidance(game_state: dict, targets: list[tuple[int, int]]):
    first = np.zeros(4, dtype=np.float32)
    if not targets:
        return None, first
    field = np.asarray(game_state["field"])
    start = tuple(game_state["self"][3])
    blocked = {tuple(position) for position, _ in game_state["bombs"]}
    blocked.update(tuple(opponent[3]) for opponent in game_state["others"])
    target_set = set(targets)
    candidates = []
    for index, action in enumerate(ACTIONS[:4]):
        dx, dy = MOVE_DELTAS[action]
        neighbour = (start[0] + dx, start[1] + dy)
        if not _passable(field, neighbour, blocked, target_set):
            continue
        distance = _grid_distance(field, neighbour, target_set, blocked)
        if distance is not None:
            candidates.append((distance + 1, index))
    if start in target_set:
        return 0, first
    if not candidates:
        return None, first
    best = min(distance for distance, _ in candidates)
    for distance, index in candidates:
        if distance == best:
            first[index] = 1.0
    return best, first


def bomb_escape_guidance(game_state: dict):
    first = np.zeros(5, dtype=np.float32)
    if not game_state["self"][2]:
        return False, first
    field = np.asarray(game_state["field"])
    start = tuple(game_state["self"][3])
    own_blast = set(blast_coordinates(field, start))
    blocked = {tuple(position) for position, _ in game_state["bombs"]} | {start}
    blocked.update(tuple(opponent[3]) for opponent in game_state["others"])
    danger = danger_urgency_map(game_state)
    successes = []
    for index, action in enumerate(ACTIONS[:5]):
        if action == "WAIT":
            neighbour = start
        else:
            dx, dy = MOVE_DELTAS[action]
            neighbour = (start[0] + dx, start[1] + dy)
            if not _passable(field, neighbour, blocked - {start}, set()):
                continue
        queue = deque([(neighbour, 1)])
        seen = {(neighbour, 1)}
        success_step = None
        while queue:
            position, step = queue.popleft()
            if danger[position] == 0 and position not in own_blast:
                success_step = step
                break
            if step >= BOMB_TIMER:
                continue
            for dx, dy in (*MOVE_DELTAS.values(), (0, 0)):
                nxt = (position[0] + dx, position[1] + dy)
                state = (nxt, step + 1)
                if state in seen:
                    continue
                if nxt == start and position != start:
                    continue
                if _passable(field, nxt, blocked - {start}, set()):
                    seen.add(state)
                    queue.append(state)
        if success_step is not None:
            successes.append((success_step, index))
    if not successes:
        return False, first
    best = min(step for step, _ in successes)
    for step, index in successes:
        if step == best:
            first[index] = 1.0
    return True, first


def _grid_distance(field, start, targets, blocked):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        position, distance = queue.popleft()
        if position in targets:
            return distance
        for dx, dy in MOVE_DELTAS.values():
            nxt = (position[0] + dx, position[1] + dy)
            if nxt not in seen and _passable(field, nxt, blocked, targets):
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def _passable(field, position, blocked, targets):
    return (
        0 <= position[0] < field.shape[0]
        and 0 <= position[1] < field.shape[1]
        and field[position] == 0
        and (position not in blocked or position in targets)
    )
