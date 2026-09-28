"""
callbacks.py —— 攻击特征版（14 维特征 / 32 维输入）

替换 agent_code/my_agent_dqn_attack/callbacks.py 的全部内容。

相对 n-step DQN 版新增三位对手相关特征。动机：此前所有版本的
score 与 coins 两列几乎完全相同，说明智能体从未击杀过任何对手 ——
因为特征里只把 others 当作障碍物，智能体根本"看不见"对手在哪。
击杀分是完全未触及的一块。

新增特征刻意保持为状态描述而非动作建议：智能体知道"对手在哪个方
向"、"是否已逼近"、"此处放弹能否命中"，但何时该攻击、攻击后如何撤
离仍需自行学习。
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ALLOWED_ACTIONS = ACTIONS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "dqn-model.pt")

DIRECTIONS = {
    'UP':    (0, -1),
    'RIGHT': (1,  0),
    'DOWN':  (0,  1),
    'LEFT':  (-1, 0),
}
DIR_ORDER = ['UP', 'RIGHT', 'DOWN', 'LEFT']

BOMB_POWER = 3
BOMB_TIMER = 4

INPUT_DIM = 32          # 见 features_to_vector 的布局说明
HIDDEN = 128


# ============================================================
# 网络
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN, n_actions=len(ACTIONS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def features_to_vector(features):
    """把 14 维特征元组展开成 32 维浮点向量。

    方向类特征必须 one-hot：网络会把整数当作有序量，而"方向 2"并不
    比"方向 1"大一倍，直接输入会引入虚假的序关系。

    布局：
      [0:4]   四方向可走          （原样，4 维）
      [4:7]   危险等级 0-2        （one-hot，3 维）
      [7:12]  逃生方向 0-4        （one-hot，5 维）
      [12:17] 目标方向 0-4        （one-hot，5 维）
      [17]    放弹是否有活路
      [18]    炸弹是否可用
      [19]    能否炸到箱子
      [20:25] 上一步方向 0-4      （one-hot，5 维）
      [25:30] 最近对手方向 0-4    （one-hot，5 维）   <- 新增
      [30]    是否逼近对手                            <- 新增
      [31]    放弹能否炸到对手                        <- 新增
    """
    if features is None:
        return None

    v = np.zeros(INPUT_DIM, dtype=np.float32)

    v[0:4] = features[0:4]
    v[4 + features[4]] = 1.0
    v[7 + features[5]] = 1.0
    v[12 + features[6]] = 1.0
    v[17] = features[7]
    v[18] = features[8]
    v[19] = features[9]
    v[20 + features[10]] = 1.0
    v[25 + features[11]] = 1.0
    v[30] = features[12]
    v[31] = features[13]

    return v


# ============================================================
# 框架回调
# ============================================================

def setup(self):
    self.model = QNetwork()

    if os.path.isfile(MODEL_PATH):
        self.logger.info("Loading DQN weights from saved state.")
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    else:
        self.logger.info("Setting up DQN from scratch.")

    self.model.eval()


def act(self, game_state: dict) -> str:
    if game_state['step'] == 1:
        self.last_dir = 4

    features = state_to_features(game_state, getattr(self, 'last_dir', 4))
    epsilon = getattr(self, 'epsilon', 0.0) if self.train else 0.0

    if random.random() < epsilon:
        action = random.choice(ALLOWED_ACTIONS)
    else:
        vec = features_to_vector(features)
        with torch.no_grad():
            q = self.model(torch.from_numpy(vec).unsqueeze(0))
        action = ACTIONS[int(torch.argmax(q, dim=1).item())]

    self.last_dir = DIR_ORDER.index(action) if action in DIR_ORDER else 4
    return action


# ============================================================
# 基础工具
# ============================================================

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


# ============================================================
# 特征提取
# ============================================================

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