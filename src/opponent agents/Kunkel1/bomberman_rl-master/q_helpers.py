
# features.py
import numpy as np
from collections import deque
from enum import IntEnum

import os
import sys

from settings import BOMB_POWER, BOMB_TIMER, EXPLOSION_TIMER

from typing import List
import events as e

# ---------------------------------------------------------------------------
# constants & small helpers (unchanged unless noted)
# ---------------------------------------------------------------------------


class TransitionFields(IntEnum):
    ITERATION          = 0
    STATE              = 1
    LEGAL_ACTIONS      = 2
    ACTION             = 3
    REWARD             = 4
    DONE               = 5
    NEXT_STATE         = 6
    NEXT_LEGAL_ACTIONS = 7

DIR_VECS    = [(0, -1), (1, 0), (0, 1), (-1, 0)]          # URDL
DIRS        = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTS        = DIRS + ["BOMB", "WAIT"]
OBJS        = ["NONE", "ENEMY", "CRATE", "COIN"]
OCCS        = ["EMPTY", "WALL", "COIN", "CRATE", "ENEMY", "BOMB", "DANGER"]

DIR_BITS    = {a: i for i, a in enumerate(DIRS)}
ACT_BITS    = {a: i for i, a in enumerate(ACTS)}
OBJ_BITS    = {a: i for i, a in enumerate(OBJS)}
OCC_BITS    = {a: i for i, a in enumerate(OCCS)}

N_ACTIONS   = len(ACTS)
N_STATES    = 2**18


def in_bounds(x, y, rows, cols):
    return 0 <= x < rows and 0 <= y < cols

def is_free(nx, ny, arena, bombs, others):
    return arena[nx, ny] == 0     \
       and all((nx, ny) != pos for pos,_ in bombs) \
       and all((nx, ny) != pos for *_n,pos in others)

def dir_and_dist_bfs(start, goals, arena, bombs, others):
    """
    Return (dir, dist) where
        dir  = 0-4 (WURDL) for the *nearest* goal (None if no goal)
        dist = #steps to that goal   (None if no goal)
    """
    if not goals:
        return None, None
    rows, cols = arena.shape
    Q = deque([(start, None, 0)])
    seen = {start}
    while Q:
        (cx, cy), first, dist = Q.popleft()
        if (cx, cy) in goals:
            return (0 if first is None else first), dist
        for d, (dx, dy) in enumerate(DIR_VECS):
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny, rows, cols):  continue
            if not arena[nx, ny] == 0:             continue
            if (nx, ny) in seen:                   continue
            seen.add((nx, ny))
            Q.append(((nx, ny), d if first is None else first, dist + 1))
    return None, None

def dir_and_dist_bfs_to_adjacent(start, goals, arena, bombs, others):
    """Same as above, but goals = *tiles adjacent to any crate*."""
    if not goals:
        return None, None
    rows, cols = arena.shape
    # collect all passable neighbours of every crate
    adj = set()
    for gx, gy in goals:
        for dx, dy in DIR_VECS:
            nx, ny = gx + dx, gy + dy
            if in_bounds(nx, ny, rows, cols) and \
               arena[nx, ny] == 0:
                adj.add((nx, ny))
    return dir_and_dist_bfs(start, adj, arena, bombs, others)

def compute_blast_map(arena, bombs):
    """
    Returns an int array M where:
        M[x,y] = smallest timer of any bomb that will blast (x,y)
                 or 99 if no bomb reaches it.
    """
    rows, cols = arena.shape
    INF = 99
    M   = np.full((rows, cols), INF, dtype=np.int8)

    for (bx, by), t in bombs:
        for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
            for k in range(0 if dx==dy==0 else 1, BOMB_POWER+1):
                x, y = bx + dx*k, by + dy*k
                if not (0 <= x < rows and 0 <= y < cols):
                    break
                if arena[x, y] == -1:        # solid wall blocks blast and ray
                    break
                M[x, y] = min(M[x, y], t)
    return M          # 0/1/2/3/4/… / 99



def can_survive_current_bombs(start_xy, arena, bombs, others, expl_map, blast_map) -> bool:
    """
    Return True  → there exists a sequence of moves/WAITs that avoids
                   every blast until all current bombs are finished.
           False → every possible action sequence ends in an explosion.
    """
    if not bombs:                         # trivial: no bombs at all
        return True

    # --- how long do we need to survive?
    last_timer = max(t for _pos, t in bombs)          # 1…4
    horizon    = last_timer                           # ticks to simulate
    rows, cols = arena.shape
    DIR_WAIT   = DIR_VECS + [(0, 0)]                  # 4 moves + WAIT

    Q     = deque([(start_xy[0], start_xy[1], 0)])    # (x, y, t)
    seen  = {(start_xy, 0)}

    while Q:
        x, y, t = Q.popleft()

        # blown up already (exploding now at t=0) or scheduled to explode by time t
        if (t == 0 and expl_map[x, y] > 0) or (blast_map[x, y] <= t):
            continue

        # reached or passed horizon → survived!
        if t >= horizon:
            return True

        # expand to next tick
        for dx, dy in DIR_WAIT:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, rows, cols):
                continue
            if arena[nx, ny] != 0:
                continue
            if any((nx, ny) == pos for pos, _ in bombs):
                continue
            if any((nx, ny) == pos for *_n, pos in others):
                continue

            key = (nx, ny, t + 1)
            if key not in seen:
                seen.add(key)
                Q.append((nx, ny, t + 1))

    # queue exhausted → every path died
    return False

def is_safe_tile(nx: int, ny: int,
                 arena: np.ndarray,
                 bombs, expl_map, blast_map, others) -> bool:
    """
    True  → stepping onto (nx,ny) still leaves at least ONE survivable action
            (including WAIT) according to best_safe_dir from that tile.
    False → certain death.
    """
    if expl_map[nx, ny] > 0:                     # already exploding
        return False

    # --- and is there a full survival path for the existing bombs?
    return can_survive_current_bombs((nx, ny), arena, bombs, others, expl_map, blast_map)


# ---------------------------------------------------------------------------
# NEW   ──  compact 18‑bit state encoding  ───────────────────────────────────
# layout (LSB→MSB):
#   0 – 11   neighbour occupancy (4 × 3 bits URDL)
#   12 – 13  direction (2 bits) of nearest object‑of‑interest 00 up, 01 right, 10 down, 11 left
#   14 – 15  object type (2 bits) 00 none, 01 enemy, 10 crate, 11 coin
#   16 – 17  safety bits for Bomb and Wait (1 = safe action, 0 = deadly)
#
# total: 18 bits
# ---------------------------------------------------------------------------


def state_to_tabular_features(game_state: dict | None) -> int | None:
    if game_state is None:
        return None

    arena      = game_state["field"]
    bombs      = game_state["bombs"]
    coins      = set(map(tuple, game_state["coins"]))
    others     = game_state["others"]
    expl_map   = game_state["explosion_map"]
    name, score, bombs_left, (x, y) = game_state["self"]
    rows, cols = arena.shape
    # ---------------------------------------------------------------- helpers
    blast_map  = compute_blast_map(arena, bombs)
    crate_pos  = {(cx, cy) for cx in range(rows) for cy in range(cols)
                  if arena[cx, cy] == 1}
    enemies    = {pos for *_n, pos in others}

    # nearest XYZ (dir=0 means none found)
    coin_dir,  coin_dist  = dir_and_dist_bfs((x, y), coins,
                                             arena, bombs, others)
    crate_dir, crate_dist = dir_and_dist_bfs_to_adjacent((x, y), crate_pos,
                                             arena, bombs, others)
    enemy_dir, enemy_dist = dir_and_dist_bfs((x, y), enemies,
                                             arena, bombs, others)

    # ---------- determine “object of interest” according to priority rules --
    obj_bits  = OBJ_BITS["NONE"]
    dir_bits  = DIR_BITS["UP"]

    if crate_dist != None and crate_dist < 4:
        obj_bits = OBJ_BITS["CRATE"]
        dir_bits = crate_dir
    if enemy_dist != None and enemy_dist < 4:
        obj_bits = OBJ_BITS["ENEMY"]
        dir_bits = enemy_dir
    if coin_dist != None:
        obj_bits = OBJ_BITS["COIN"]
        dir_bits = coin_dir
    elif crate_dist != None:
        obj_bits = OBJ_BITS["CRATE"]
        dir_bits = crate_dir
    elif enemy_dist != None:
        obj_bits = OBJ_BITS["ENEMY"]
        dir_bits = enemy_dir
    # else keep NONE / 00

    # ---------------------------------------------------------------- per‑direction info
    neighbour_bits = 0   # 12 bits

    for d, (dx, dy) in enumerate(DIR_VECS, start=1):   # d = 1..4 (URDL)
        nx, ny = x + dx, y + dy


        # ---- 3‑bit OCCUPANCY ----------------------------------------------
        if arena[nx, ny] == -1:
            occ = OCC_BITS["WALL"]
        elif arena[nx, ny] == 1:
            occ = OCC_BITS["CRATE"]
        elif any((nx, ny) == pos for pos, _ in bombs):
            occ = OCC_BITS["BOMB"]
        elif any((nx, ny) == pos for *_n, pos in others):
            occ = OCC_BITS["ENEMY"]
        elif not is_safe_tile(nx, ny, arena, bombs, expl_map, blast_map, others):
            occ = OCC_BITS["DANGER"]
        elif (nx, ny) in coins:
            occ = OCC_BITS["COIN"]
        else:
            occ = OCC_BITS["EMPTY"]

        neighbour_bits |= (occ & 0b111) << (3 * (4 - d))  # 3 bits per dir

    # ---------------------------------------------------------------- tile‑related bits

    # would it still be safe after placing a bomb here?
    bombs_with_self     = list(bombs) + [((x, y), BOMB_TIMER)]
    blast_map_with_self = compute_blast_map(arena, bombs_with_self)
    here_safe_with_bomb = is_safe_tile(x, y, arena, bombs_with_self, expl_map,
                                       blast_map_with_self, others)

    # check if agent is standing on a bomb
    bomb_on_tile = any((bx, by) == (x, y) for (bx, by), _ in bombs)
    # is (placing a bomb allowed & there is no bomb on the current tile & placing a bomb would not kill us)?
    bomb_bit    = int(bombs_left > 0 and not bomb_on_tile and here_safe_with_bomb)

    # is waiting safe?
    wait_bit = is_safe_tile(x, y, arena, bombs, expl_map, blast_map, others)

    # ---------------------------------------------------------------- pack bits into int
    state_id = (
        (wait_bit     << 17) |
        (bomb_bit     << 16) |
        (obj_bits     << 14) |
        (dir_bits     << 12) |
        neighbour_bits
    )

    return state_id

# ---------------------------------------------------------------------------
# helper: human‑readable description of a 18‑bit state id
# ---------------------------------------------------------------------------

def describe_tabular_state(state_id: int) -> str:
    wait_bit        = (state_id >> 17) & 0b1      # May safely wait
    bomb_bit        = (state_id >> 16) & 0b1      # May (safely) use bomb
    obj_bits        = (state_id >> 14) & 0b11     # 2 bits for ["NONE", "ENEMY", "CRATE", "COIN"]
    dir_bits        = (state_id >> 12) & 0b11     # 2 bits for ["UP", "RIGHT", "DOWN", "LEFT"]
    neighbour_bits  = (state_id >>  0) & 0xFFF    # 12 bits for four directions ["UP", "RIGHT", "DOWN", "LEFT"] with the 3-bit states ["EMPTY", "WALL", "COIN", "CRATE", "ENEMY", "BOMB", "DANGER", "ENEMY_IN_DANGER"]
    dir_name        = DIRS[dir_bits]
    obj_name        = OBJS[obj_bits]

    # decode neighbour occupancy
    neigh_occ = []
    for shift in (9, 6, 3, 0):
        code = (neighbour_bits >> shift) & 0b111  # extract 3 bits at a time
        neigh_occ.append(OCCS[code])

    return (
        f"State : {state_id:018b}\n"
        f"Goto  : {obj_name} ({dir_name})\n"
        f"Wait  : {'OK' if wait_bit else 'DANGER'}\n"
        f"Bomb  : {'OK' if bomb_bit else 'DANGER'}\n"
        f"Up    : {neigh_occ[0]}\n"
        f"Right : {neigh_occ[1]}\n"
        f"Down  : {neigh_occ[2]}\n"
        f"Left  : {neigh_occ[3]}"
    )




def compute_danger_and_explosion_map(arena, bombs, explosion):
    """
    Returns an int array M where:
        M[x,y] = smallest timer of any bomb that will blast (x,y)
                or 99 if no bomb reaches it.
    """
    rows, cols = arena.shape
    INF = 99
    danger   = np.full((rows, cols), INF, dtype=np.int8)

    for (bx, by), t in bombs:
        for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
            for k in range(0 if dx==dy==0 else 1, BOMB_POWER+1):
                x, y = bx + dx*k, by + dy*k
                if not (0 <= x < rows and 0 <= y < cols):
                    break
                if arena[x, y] == -1:        # solid wall blocks blast and ray
                    break
                danger[x, y] = min(danger[x, y], t)

    explosion = explosion.copy()
    # The explosion map does not cover bombs that are just exploding (t=0)
    explosion[danger == 0] = EXPLOSION_TIMER
    # We don't want active explosions in the danger map
    danger[danger == 0] = INF

    # Make it such that higher danger values correspond to higher danger
    danger =  danger.astype(np.float32)
    danger[danger != INF] -= 1 # Shift everything by one since we removed active explosions
    danger[danger != INF]  = (BOMB_TIMER - 1 - danger[danger != INF]) / (BOMB_TIMER - 1)
    danger[danger == INF]  = 0

    return danger, explosion

def state_to_cnn_features(game_state: dict) -> np.ndarray:
    """
    Convert the game state into a multi-channel one-hot feature grid for CNN input.

    Channels (so far):
    - wall, free, crate (one-hot split of 'field')
    - bomb_timer (scaled 0..1)
    - explosion_map (scaled 0..1)
    - coin_map
    - self_pos
    - opp_pos
    - can_bomb (self)
    - opp_can_bomb (per-tile binary where opponent can bomb)
    """
    if game_state is None:
        return None

    field = game_state['field']

    wall_map = (field == -1).astype(np.float32)
    free_map = (field == 0).astype(np.float32)
    crate_map = (field == 1).astype(np.float32)

    # Bomb map: normalize timers
    bomb_map = np.zeros_like(field, dtype=np.float32)
    for (x, y), t in game_state['bombs']:
        # Normalise such that 0 = no bomb, 0.25 = very urgent bomb, 0.5 = urgent bomb, 0.75 = not so urgent bomb and 1 = new bomb
        # The nonlinearity should be able to learn it even if we set bomb_map = t / BOMB_TIMER, but this might be easier and certainly does not hurt
        bomb_map[x, y] = (BOMB_TIMER - t) / BOMB_TIMER


    danger_map, explosion_map = compute_danger_and_explosion_map(field, game_state['bombs'], game_state['explosion_map'])

    # Explosion map: normalize to [0,1]
    explosion_map = explosion_map / EXPLOSION_TIMER

    # Coins
    coin_map = np.zeros_like(field, dtype=np.float32)
    for (x, y) in game_state['coins']:
        coin_map[x, y] = 1.0

    # Self
    self_pos_channel = np.zeros_like(field, dtype=np.float32)
    sx, sy = game_state['self'][3]
    self_pos_channel[sx, sy] = 1.0

    # Opponents
    opp_pos_channel = np.zeros_like(field, dtype=np.float32)
    opp_can_bomb_channel = np.zeros_like(field, dtype=np.float32)
    for opp in game_state['others']:
        ox, oy = opp[3]
        opp_pos_channel[ox, oy] = 1.0
        opp_can_bomb_channel[ox, oy] = float(opp[2])

    # Self bomb ability
    can_bomb_channel = np.ones_like(field, dtype=np.float32) * int(game_state['self'][2])

    # Stack all channels
    multi_channel_grid = np.stack((
        wall_map, free_map, crate_map,
        bomb_map, danger_map, explosion_map,
        coin_map, self_pos_channel, opp_pos_channel,
        can_bomb_channel, opp_can_bomb_channel
    ), axis=-1)

    return multi_channel_grid




def describe_cnn_state(tensor: np.ndarray, max_channels: int = 20):
    """
    Pretty-print the CNN input tensor channel by channel.

    Args:
        tensor: np.ndarray of shape (rows, cols, channels)
        channel_names: optional list of names for each channel
        max_channels: limit to print to avoid huge dumps
    """
    rows, cols, C = tensor.shape
    print(f"[DEBUG] CNN encoding: shape = ({rows}, {cols}, {C})")



    channel_names = [
        "wall_map", "free_map", "crate_map",
        "bomb_timer", "danger_map", "explosion_map",
        "coin_map", "self_pos", "opp_pos",
        "can_bomb", "opp_can_bomb"
    ]

    if channel_names is None:
        channel_names = [f"ch{c}" for c in range(C)]

    for c in range(min(C, max_channels)):
        print(f"\n--- Channel {c} : {channel_names[c]} ---")
        print(tensor[:, :, c].T)
        # Give a little interpretation if channel is binary
        unique_vals = np.unique(tensor[:, :, c])
        if np.all(np.isin(unique_vals, [0,1])):
            print("   (binary map: 1 = presence, 0 = absence)")
        elif np.all(unique_vals == 0):
            print("   (all zero channel)")
        else:
            print(f"   values in [{tensor[:,:,c].min()}, {tensor[:,:,c].max()}]")

    if C > max_channels:
        print(f"... skipped {C - max_channels} channels ...")


def reward_from_events(events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED:      0.2,
        e.KILLED_OPPONENT:     1.0,
        e.CRATE_DESTROYED:     0.1,
        e.KILLED_SELF:        -0.9,
        e.BOMB_DROPPED:        0.02,
        e.GOT_KILLED:         -1.0,
        e.WAITED:             -0.02,
        e.INVALID_ACTION:     -0.02,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum


def get_legal_actions(game_state) -> np.ndarray:
    """
    Return the list of ACTION IDs that are *physically legal* in the current Bomberman state.
    Legal = the move keeps you on the board, lands on a free tile (0 in `arena`)
            with no bomb and no other agent occupying it.
            'WAIT' is always legal.
            'BOMB' is legal iff you still have bombs_left and no bomb already on your tile.
    Explosion/danger is NOT checked here – leave that to the policy.
    """
    # If we have no game state (first call or game over)--everything is allowed.
    if game_state is None:
        return np.arange(N_ACTIONS)

    arena      = game_state["field"]          # 2-D int8 array, -1: wall, 1: crate, 0: free
    bombs      = game_state["bombs"]          # [((x, y), timer), …]
    others     = game_state["others"]         # [(name, score, bombs_left, (x, y)), …]
    name, score, bombs_left, (x, y) = game_state["self"]
    rows, cols = arena.shape

    # ----- helpers -----------------------------------------------------------
    def in_bounds(cx, cy):
        return 0 <= cx < rows and 0 <= cy < cols

    def tile_is_free(cx, cy):
        """Free = no wall/crate, no bomb"""
        if not in_bounds(cx, cy):
            return False
        if arena[cx, cy] != 0:                # wall or crate
            return False
        for (bx, by), _ in bombs:
            if bx == cx and by == cy:
                return False
        for *_ignore, (ox, oy) in others:
            if ox == cx and oy == cy:
                return False
        return True
    # -------------------------------------------------------------------------

    legal = []

    # Movement actions
    for act, (dx, dy) in zip(DIRS, DIR_VECS):
        if tile_is_free(x + dx, y + dy):
            legal.append(ACT_BITS[act])

    # WAIT is always legal
    legal.append(ACT_BITS['WAIT'])

    # BOMB: at least one bomb left *and* no bomb already on current tile
    if bombs_left > 0 and all((bx, by) != (x, y) for (bx, by), _ in bombs):
        legal.append(ACT_BITS['BOMB'])

    return np.array(legal, dtype=int)

