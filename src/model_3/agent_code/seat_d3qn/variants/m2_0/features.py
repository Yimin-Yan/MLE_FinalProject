"""Feature extraction and observed legal-action masking for M2-0."""

from collections import deque

import numpy as np

from .config import (
    ACTIONS,
    BFS_DISTANCE_CAP,
    BOMB_TIMER_SCALE,
    CRATE_POTENTIAL_WEIGHT,
    FEATURE_DIM,
    MANHATTAN_DISTANCE_SCALE,
    MAX_GAME_STEPS,
    RELATIVE_COORD_SCALE,
)


MOVE_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

# Feature layout:
# 0:14 local state, 14:22 coin target, 22:30 crate target,
# 30:35 nearest bomb, 35:39 nearest opponent, 39:42 round context.
# The remaining entries are zero in M2-0 and belong to later variants.


def legal_action_mask(game_state):
    """Return actions available in the observation received by the agent.

    This cannot prevent a race invalid action: another agent may move first and
    occupy a tile that was free in this snapshot.
    """
    mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    if game_state is None:
        return mask

    field = np.asarray(game_state["field"])
    _, _, bombs_left, (x, y) = game_state["self"]
    bomb_positions = {tuple(position) for position, _ in game_state["bombs"]}
    other_positions = {tuple(other[3]) for other in game_state["others"]}

    for action_index, (dx, dy) in enumerate(MOVE_DELTAS):
        nx, ny = x + dx, y + dy
        inside = 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
        if inside:
            mask[action_index] = (
                field[nx, ny] == 0
                and (nx, ny) not in bomb_positions
                and (nx, ny) not in other_positions
            )

    mask[4] = True  # WAIT is always accepted by the official environment.
    mask[5] = bool(bombs_left)
    return mask


def _static_bfs(field, start):
    """Shortest-path distances when only walls and crates are blocking."""
    distances = np.full(field.shape, -1, dtype=np.int16)
    sx, sy = start
    if field[sx, sy] != 0:
        return distances

    distances[sx, sy] = 0
    queue = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        next_distance = int(distances[x, y]) + 1
        for dx, dy in MOVE_DELTAS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] == 0 and distances[nx, ny] < 0:
                distances[nx, ny] = next_distance
                queue.append((nx, ny))
    return distances


def _nearest_coin(field, coins, distances_from_self):
    candidates = []
    for coin in coins:
        cx, cy = int(coin[0]), int(coin[1])
        if not (0 <= cx < field.shape[0] and 0 <= cy < field.shape[1]):
            continue
        distance = int(distances_from_self[cx, cy])
        if field[cx, cy] == 0 and distance >= 0:
            candidates.append((distance, cx, cy))
    if not candidates:
        return None
    _, x, y = min(candidates)
    return x, y


def _nearest_crate_target(field, distances_from_self):
    candidates = []
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            distance = int(distances_from_self[x, y])
            if field[x, y] != 0 or distance < 0:
                continue
            touches_crate = False
            for dx, dy in MOVE_DELTAS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                    touches_crate = touches_crate or field[nx, ny] == 1
            if touches_crate:
                candidates.append((distance, x, y))
    if not candidates:
        return None
    _, x, y = min(candidates)
    return x, y


def _target_block(field, self_position, target, move_mask):
    """Encode one navigation target in eight values."""
    block = np.zeros(8, dtype=np.float32)
    if target is None:
        return block

    x, y = self_position
    tx, ty = target
    distances_to_target = _static_bfs(field, target)
    current_distance = int(distances_to_target[x, y])
    if current_distance < 0:
        return block

    block[0] = 1.0
    block[1] = np.clip((tx - x) / RELATIVE_COORD_SCALE, -1.0, 1.0)
    block[2] = np.clip((ty - y) / RELATIVE_COORD_SCALE, -1.0, 1.0)
    block[3] = min(current_distance, BFS_DISTANCE_CAP) / BFS_DISTANCE_CAP

    for action_index, (dx, dy) in enumerate(MOVE_DELTAS):
        if not move_mask[action_index]:
            continue
        nx, ny = x + dx, y + dy
        neighbour_distance = int(distances_to_target[nx, ny])
        if neighbour_distance < 0 or neighbour_distance > current_distance:
            block[4 + action_index] = -1.0
        elif neighbour_distance < current_distance:
            block[4 + action_index] = 1.0
        # Equal distance stays at zero. It is useful to keep this neutral.
    return block


def state_to_features(game_state):
    """Convert an official game-state dictionary into the fixed 64-D vector."""
    if game_state is None:
        return None

    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    field = np.asarray(game_state["field"])
    explosion_map = np.asarray(game_state["explosion_map"])
    _, own_score, bombs_left, (x, y) = game_state["self"]
    x, y = int(x), int(y)

    mask = legal_action_mask(game_state)
    features[0:4] = mask[0:4].astype(np.float32)
    features[4] = float(bool(bombs_left))

    for index, (dx, dy) in enumerate(MOVE_DELTAS):
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            features[5 + index] = float(field[nx, ny] == 1)
            features[9 + index] = float(explosion_map[nx, ny] > 0)

    bomb_positions = [tuple(position) for position, _ in game_state["bombs"]]
    features[13] = float((x, y) in bomb_positions)

    distances_from_self = _static_bfs(field, (x, y))
    coin_target = _nearest_coin(field, game_state["coins"], distances_from_self)
    crate_target = _nearest_crate_target(field, distances_from_self)
    features[14:22] = _target_block(field, (x, y), coin_target, mask[0:4])
    features[22:30] = _target_block(field, (x, y), crate_target, mask[0:4])

    if game_state["bombs"]:
        def bomb_key(bomb):
            (bx, by), timer = bomb
            return abs(bx - x) + abs(by - y), timer, bx, by

        (bx, by), timer = min(game_state["bombs"], key=bomb_key)
        distance = abs(bx - x) + abs(by - y)
        features[30] = 1.0
        features[31] = np.clip((bx - x) / RELATIVE_COORD_SCALE, -1.0, 1.0)
        features[32] = np.clip((by - y) / RELATIVE_COORD_SCALE, -1.0, 1.0)
        features[33] = min(distance / MANHATTAN_DISTANCE_SCALE, 1.0)
        features[34] = np.clip(timer / BOMB_TIMER_SCALE, 0.0, 1.0)

    if game_state["others"]:
        def enemy_key(enemy):
            name, _, _, (ex, ey) = enemy
            return abs(ex - x) + abs(ey - y), ex, ey, str(name)

        _, _, _, (ex, ey) = min(game_state["others"], key=enemy_key)
        distance = abs(ex - x) + abs(ey - y)
        features[35] = 1.0
        features[36] = np.clip((ex - x) / RELATIVE_COORD_SCALE, -1.0, 1.0)
        features[37] = np.clip((ey - y) / RELATIVE_COORD_SCALE, -1.0, 1.0)
        features[38] = min(distance / MANHATTAN_DISTANCE_SCALE, 1.0)

    features[39] = np.clip(game_state["step"] / MAX_GAME_STEPS, 0.0, 1.0)
    features[40] = np.tanh(float(own_score) / 10.0)
    if game_state["others"]:
        highest_opponent_score = max(float(other[1]) for other in game_state["others"])
    else:
        highest_opponent_score = float(own_score)
    features[41] = np.tanh((float(own_score) - highest_opponent_score) / 10.0)

    # 42:64 are deliberately left at zero for later M2 variants.
    if not np.isfinite(features).all():
        raise ValueError("Non-finite value found in the M2-0 feature vector")
    return features


def potential_from_features(features):
    """Navigation potential derived from the exact vector used by the network."""
    if features is None:
        return 0.0

    coin_potential = 0.0
    if features[14] > 0.5:
        coin_potential = max(0.0, 1.0 - float(features[17]))

    crate_potential = 0.0
    if features[22] > 0.5:
        crate_potential = CRATE_POTENTIAL_WEIGHT * max(0.0, 1.0 - float(features[25]))

    return max(coin_potential, crate_potential)
