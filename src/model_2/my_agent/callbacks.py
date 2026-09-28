# ============================================================
# callbacks.py —— 任务 2 版本
# 替换整个文件
# ============================================================

import os
import pickle
import random
from collections import deque

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# 任务 2 起放开全部动作
ALLOWED_ACTIONS = ACTIONS
ALLOWED_IDX = list(range(len(ACTIONS)))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "my-saved-model.pt")

DIRECTIONS = {
    'UP':    (0, -1),
    'RIGHT': (1,  0),
    'DOWN':  (0,  1),
    'LEFT':  (-1, 0),
}
DIR_ORDER = ['UP', 'RIGHT', 'DOWN', 'LEFT']

BOMB_POWER = 3        # 爆炸向四个方向延伸的格数
BOMB_TIMER = 4        # 放下后几步爆炸


def q_values(model, features):
    """取某状态下的 Q 向量；没见过的状态初始化为全 0。"""
    if features not in model:
        model[features] = np.zeros(len(ACTIONS))
    return model[features]


def setup(self):
    if os.path.isfile(MODEL_PATH):
        self.logger.info("Loading model from saved state.")
        with open(MODEL_PATH, "rb") as file:
            self.model = pickle.load(file)
        self.logger.info(f"Q-table has {len(self.model)} states.")
    else:
        self.logger.info("Setting up model from scratch.")
        self.model = {}


def act(self, game_state: dict) -> str:
    if game_state['step'] == 1:
        self.last_dir = 4                    # 每局开头重置

    features = state_to_features(game_state, getattr(self, 'last_dir', 4))
    epsilon = getattr(self, 'epsilon', 0.0) if self.train else 0.0

    if random.random() < epsilon:
        action = random.choice(ALLOWED_ACTIONS)
    else:
        q = q_values(self.model, features)
        action = random.choice(ALLOWED_ACTIONS) if not np.any(q) else ACTIONS[int(np.argmax(q))]

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


# ============================================================
# 炸弹与危险
# ============================================================

def blast_tiles(field, bomb_pos):
    """一颗炸弹的爆炸范围：十字形，延伸 BOMB_POWER 格，被石墙/箱子挡住。"""
    tiles = {bomb_pos}
    for name in DIR_ORDER:
        dx, dy = DIRECTIONS[name]
        for r in range(1, BOMB_POWER + 1):
            pos = (bomb_pos[0] + dx * r, bomb_pos[1] + dy * r)
            if not _in_bounds(field, pos) or field[pos] == -1:
                break          # 石墙挡住，不再延伸
            tiles.add(pos)
            if field[pos] == 1:
                break          # 箱子会被炸掉，但爆炸不穿过去
    return tiles


def danger_map(field, bombs, explosion_map):
    """返回 {格子: 剩余危险步数}。数字越大越紧急（4 = 马上炸）。

    正在燃烧的爆炸（explosion_map > 0）视为最高危险。
    """
    danger = {}
    for bomb_pos, timer in bombs:
        bomb_pos = (int(bomb_pos[0]), int(bomb_pos[1]))
        # timer 从 3 递减到 0，0 表示即将爆炸 → 危险度 4
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

    # BFS 限深，找不在新爆炸范围也不在既有危险中的格子
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
    """在 pos 放弹能炸到几个箱子。用来区分「有价值的弹」和「浪费的弹」。"""
    return sum(1 for t in blast_tiles(field, pos) if field[t] == 1)

# ============================================================
# 特征提取
# ============================================================

def state_to_features(game_state: dict, last_dir: int = 4):
    """11 维特征元组，作为 Q 表的 key。

    任务 3/4 版本：为控制状态空间，[4] 和 [9] 相比任务 2 做了粗化。
    对手场景下状态数会翻倍，表格法必须靠压缩才可能收敛。

      [0:4]  上/右/下/左是否可走（含炸弹、对手阻挡）
      [4]    危险等级 0-2（0 安全 / 1 有炸弹威胁 / 2 即将爆炸或正在燃烧）
      [5]    逃生方向 0-3，安全或无路时 4
      [6]    目标方向（金币优先，其次箱子）0-3，无目标 4
      [7]    此处放弹是否有活路 0/1
      [8]    炸弹是否可用 0/1
      [9]    此处放弹能否炸到箱子 0/1
      [10]   上一步移动方向 0-3，非移动动作 4

    状态空间上限 2^4 x 3 x 5 x 5 x 2 x 2 x 2 x 5 = 24000，
    约为任务 2 版本（11 维、5 档 danger、4 档 crate）的 40%。
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

    # 炸弹和对手挡路
    blocked = set(pos for pos, _ in bombs) | set(others)

    danger = danger_map(field, bombs, game_state['explosion_map'])

    # [0:4] 四周可通行
    walkable = tuple(
        1 if is_free(field, (x + DIRECTIONS[d][0], y + DIRECTIONS[d][1]), blocked) else 0
        for d in DIR_ORDER
    )

    # [4] 危险等级，粗化为三档
    # 保留「安全 / 还有时间跑 / 必须马上跑」这三种质变，
    # 丢掉「还剩 3 步还是 4 步」这种量的区别——后者对决策影响很小，
    # 却让状态数多出近一倍。
    raw_danger = danger.get(me, 0)
    if raw_danger == 0:
        my_danger = 0
    elif raw_danger >= 3:
        my_danger = 2
    else:
        my_danger = 1

    # [5] 逃生方向
    if my_danger > 0:
        safe = safe_tiles(field, danger, blocked)
        step = bfs_first_step(field, me, safe, blocked)
        escape_dir = 4 if step is None else step
    else:
        escape_dir = 4

    # [6] 目标方向：金币优先，否则找箱子
    step = bfs_first_step(field, me, coins, blocked)
    if step is None:
        step = bfs_first_step(field, me, crate_targets(field, blocked), blocked)
    target_dir = 4 if step is None else step

    # [7] 放弹后有无活路
    bomb_safe = int(can_escape_after_bomb(field, me, danger, blocked))

    # [9] 能否炸到箱子，二值化
    # 「炸得到」和「炸不到」是决策分界；炸 1 个还是 3 个对「要不要放」
    # 影响不大，不值得为此把状态数翻一倍。
    crate_gain = 1 if crates_in_blast(field, me) > 0 else 0

    return walkable + (my_danger, escape_dir, target_dir, bomb_safe,
                       bombs_available, crate_gain, last_dir)