"""
callbacks.py —— DQN 版本

替换 agent_code/my_agent_dqn/callbacks.py 的全部内容。

与表格版的差别只在模型部分：特征提取（BFS、危险图、逃生判断等）
完全复用，state_to_features 仍然返回那个 11 维元组，再由
features_to_vector 展开成神经网络的输入向量。
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

# 展开后的输入维度，见 features_to_vector
INPUT_DIM = 25
HIDDEN = 128


# ============================================================
# 网络
# ============================================================

class QNetwork(nn.Module):
    """两层隐藏层的全连接网络，输入状态向量，输出 6 个动作的 Q 值。

    规模刻意保持很小：特征已经是人工设计的低维表示，不需要从像素
    学起。25-128-128-6 在 CPU 上单步推理远低于 0.5 秒的限制。
    """

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
    """把 11 维特征元组展开成 25 维浮点向量。

    方向类特征（0-4）必须做 one-hot：神经网络会把整数当成有序的
    数值，而"方向 2"并不比"方向 1"大一倍，直接输入会引入虚假的
    序关系。二值特征则可以原样使用。

    布局：
      [0:4]   四方向可走          （原样，4 维）
      [4:7]   危险等级 0-2        （one-hot，3 维）
      [7:12]  逃生方向 0-4        （one-hot，5 维）
      [12:17] 目标方向 0-4        （one-hot，5 维）
      [17]    放弹是否有活路      （原样）
      [18]    炸弹是否可用        （原样）
      [19]    能否炸到箱子        （原样）
      [20:25] 上一步方向 0-4      （one-hot，5 维）
    """
    if features is None:
        return None

    v = np.zeros(INPUT_DIM, dtype=np.float32)

    v[0:4] = features[0:4]                 # 四方向可走
    v[4 + features[4]] = 1.0               # 危险等级 (3)
    v[7 + features[5]] = 1.0               # 逃生方向 (5)
    v[12 + features[6]] = 1.0              # 目标方向 (5)
    v[17] = features[7]                    # 放弹有活路
    v[18] = features[8]                    # 炸弹可用
    v[19] = features[9]                    # 能炸到箱子
    v[20 + features[10]] = 1.0             # 上一步方向 (5)

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

    self.model.eval()          # 训练时 train.py 会切回 train 模式


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
# 以下特征提取部分与表格版完全相同
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
        if nxt in targets and is_free(field, nxt, blocked):
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
            if not is_free(field, nxt, blocked):
                continue
            if nxt in targets:
                return first
            visited.add(nxt)
            queue.append((nxt, first))

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
    """返回 {格子: 剩余危险步数}。数字越大越紧急（4 = 马上炸）。"""
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
    """所有可站立且不在危险中的格子。"""
    result = set()
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            pos = (x, y)
            if is_free(field, pos, blocked) and pos not in danger:
                result.add(pos)
    return result


def can_escape_after_bomb(field, start, existing_danger, blocked=frozenset()):
    """假设此刻在 start 放炸弹，BOMB_TIMER 步内能否走到安全格。

    这是状态描述（此处放弹是否安全），不是动作建议 ——
    agent 仍需自行学会「何时该放」以及「放在哪里有价值」。
    """
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
    """箱子旁边可站立的空地 —— 想炸箱子得先走到这里。"""
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
    """在 pos 放弹能炸到几个箱子。"""
    return sum(1 for t in blast_tiles(field, pos) if field[t] == 1)


def state_to_features(game_state: dict, last_dir: int = 4):
    """11 维特征元组。与表格版（压缩后）完全一致。

      [0:4]  上/右/下/左是否可走
      [4]    危险等级 0-2
      [5]    逃生方向 0-3，安全或无路时 4
      [6]    目标方向 0-3，无目标 4
      [7]    此处放弹是否有活路 0/1
      [8]    炸弹是否可用 0/1
      [9]    此处放弹能否炸到箱子 0/1
      [10]   上一步移动方向 0-3，非移动动作 4
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

    return walkable + (my_danger, escape_dir, target_dir, bomb_safe,
                       bombs_available, crate_gain, last_dir)