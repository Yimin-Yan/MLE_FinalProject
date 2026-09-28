from collections import deque
from random import shuffle

import numpy as np

import settings as s


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a
DIRECTIONS = {
    'UP':    (0, -1),
    'RIGHT': (1,  0),
    'DOWN':  (0,  1),
    'LEFT':  (-1, 0),
}
DIR_ORDER = ['UP', 'RIGHT', 'DOWN', 'LEFT']
BOMB_POWER = 3
BOMB_TIMER = 4
def _in_bounds(field, pos):
    x, y = pos
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]
def is_free(field, pos, blocked=frozenset()):
    """能否走进该格。field: 0=空地, 1=箱子, -1=石墙。"""
    return _in_bounds(field, pos) and field[pos] == 0 and pos not in blocked
def bfs_first_step(field, start, targets, blocked=frozenset()):
    """返回通往最近目标的最短路第一步方向索引，不可达返回 None。"""
    targets = set(targets)
    if not targets or start in targets:
        return None

    queue = deque()
    visited = {start}

    for idx, name in enumerate(DIR_ORDER):
        dx, dy = DIRECTIONS[name]
        nxt = (start[0] + dx, start[1] + dy)
        if nxt in visited:
            continue
        if nxt in targets:
            return idx
        if is_free(field, nxt, blocked):
            visited.add(nxt)
            queue.append((nxt, idx))

    while queue:
        pos, first = queue.popleft()
        for name in DIR_ORDER:
            dx, dy = DIRECTIONS[name]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in visited:
                continue
            if nxt in targets:
                return first
            if not is_free(field, nxt, blocked):
                continue
            visited.add(nxt)
            queue.append((nxt, first))

    return None
def bfs_distance(field, start, targets, blocked=frozenset(), max_depth=10):
    """到最近目标的 BFS 距离；超过 max_depth 或不可达返回 None。

    限深是为了控制开销：对手在十步以外时，具体距离对当前决策没有
    意义。
    """
    targets = set(targets)
    if not targets:
        return None
    if start in targets:
        return 0

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        pos, d = queue.popleft()
        if d >= max_depth:
            continue
        for name in DIR_ORDER:
            dx, dy = DIRECTIONS[name]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in visited:
                continue
            if nxt in targets:
                return d + 1
            if is_free(field, nxt, blocked):
                visited.add(nxt)
                queue.append((nxt, d + 1))
    return None
def blast_tiles(field, bomb_pos):
    """一颗炸弹的爆炸范围：十字形，延伸 BOMB_POWER 格，被石墙/箱子挡住。"""
    tiles = {bomb_pos}
    for name in DIR_ORDER:
        dx, dy = DIRECTIONS[name]
        for r in range(1, BOMB_POWER + 1):
            pos = (bomb_pos[0] + dx * r, bomb_pos[1] + dy * r)
            if not _in_bounds(field, pos) or field[pos] == -1:
                break
            tiles.add(pos)
            if field[pos] == 1:
                break
    return tiles
def danger_map(field, bombs, explosion_map):
    """返回 {格子: 剩余危险步数}，数字越大越紧急（4 = 马上炸）。"""
    danger = {}
    for bomb_pos, timer in bombs:
        bomb_pos = (int(bomb_pos[0]), int(bomb_pos[1]))
        urgency = BOMB_TIMER - timer
        for tile in blast_tiles(field, bomb_pos):
            danger[tile] = max(danger.get(tile, 0), urgency)

    xs, ys = np.where(explosion_map > 0)
    for x, y in zip(xs, ys):
        danger[(int(x), int(y))] = 4

    return danger
def safe_tiles(field, danger, blocked=frozenset()):
    result = set()
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            pos = (x, y)
            if is_free(field, pos, blocked) and pos not in danger:
                result.add(pos)
    return result
def can_escape_after_bomb(field, start, existing_danger, blocked=frozenset()):
    """假设此刻在 start 放炸弹，BOMB_TIMER 步内能否走到安全格。"""
    new_blast = blast_tiles(field, start)

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        pos, dist = queue.popleft()
        if dist > 0 and pos not in new_blast and pos not in existing_danger:
            return True
        if dist >= BOMB_TIMER:
            continue
        for name in DIR_ORDER:
            dx, dy = DIRECTIONS[name]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in visited or not is_free(field, nxt, blocked):
                continue
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return False
def crate_targets(field, blocked=frozenset()):
    targets = set()
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x, y] != 1:
                continue
            for name in DIR_ORDER:
                dx, dy = DIRECTIONS[name]
                pos = (x + dx, y + dy)
                if is_free(field, pos, blocked):
                    targets.add(pos)
    return targets
def crates_in_blast(field, pos):
    return sum(1 for t in blast_tiles(field, pos) if field[t] == 1)
def opponents_in_blast(field, pos, others):
    """在 pos 放弹能炸到几个对手。与 crates_in_blast 同一模式。"""
    blast = blast_tiles(field, pos)
    return sum(1 for o in others if o in blast)
def state_to_features(game_state: dict, last_dir: int = 4):
    """14 维特征元组。

      [0:4]  上/右/下/左是否可走
      [4]    危险等级 0-2
      [5]    逃生方向 0-3，安全或无路时 4
      [6]    目标方向（金币优先，其次箱子）0-3，无目标 4
      [7]    此处放弹是否有活路 0/1
      [8]    炸弹是否可用 0/1
      [9]    此处放弹能否炸到箱子 0/1
      [10]   上一步移动方向 0-3，非移动动作 4
      [11]   最近对手的方向 0-3，无对手或不可达 4      <- 新增
      [12]   是否已逼近对手（BFS 距离 <= 2）0/1        <- 新增
      [13]   此处放弹能否炸到对手 0/1                  <- 新增
    """
    if game_state is None:
        return None

    field = game_state['field']
    x, y = (int(v) for v in game_state['self'][3])
    me = (x, y)
    bombs_available = int(game_state['self'][2])

    bombs = [((int(bx), int(by)), int(t)) for (bx, by), t in game_state['bombs']]
    coins = [(int(cx), int(cy)) for cx, cy in game_state['coins']]
    others = [(int(ox), int(oy)) for _, _, _, (ox, oy) in game_state['others']]

    blocked = set(pos for pos, _ in bombs) | set(others)
    danger = danger_map(field, bombs, game_state['explosion_map'])

    walkable = tuple(
        1 if is_free(field, (x + DIRECTIONS[d][0], y + DIRECTIONS[d][1]), blocked) else 0
        for d in DIR_ORDER
    )

    raw_danger = danger.get(me, 0)
    if raw_danger == 0:
        my_danger = 0
    elif raw_danger >= 3:
        my_danger = 2
    else:
        my_danger = 1

    if my_danger > 0:
        safe = safe_tiles(field, danger, blocked)
        step = bfs_first_step(field, me, safe, blocked)
        escape_dir = 4 if step is None else step
    else:
        escape_dir = 4

    step = bfs_first_step(field, me, coins, blocked)
    if step is None:
        step = bfs_first_step(field, me, crate_targets(field, blocked), blocked)
    target_dir = 4 if step is None else step

    bomb_safe = int(can_escape_after_bomb(field, me, danger, blocked))
    crate_gain = 1 if crates_in_blast(field, me) > 0 else 0

    # --- 对手相关特征 ---
    # 寻路到对手时不能把对手本身当障碍，否则永远到不了；
    # 但炸弹仍然挡路，所以单独构造一个 blocked 集合
    blocked_no_others = set(pos for pos, _ in bombs)

    step = bfs_first_step(field, me, others, blocked_no_others)
    opp_dir = 4 if step is None else step

    d = bfs_distance(field, me, others, blocked_no_others)
    opp_near = 1 if (d is not None and d <= 2) else 0

    opp_in_blast = 1 if opponents_in_blast(field, me, others) > 0 else 0

    return walkable + (my_danger, escape_dir, target_dir, bomb_safe,
                       bombs_available, crate_gain, last_dir,
                       opp_dir, opp_near, opp_in_blast)