"""
train.py —— SARSA 版本

替换 agent_code/my_agent_sarsa/train.py 的全部内容。
callbacks.py 不需要任何改动。

与 Q-learning 版的唯一区别是 TD 目标的计算方式：

    Q-learning (off-policy):  target = r + γ · max_a' Q(s', a')
    SARSA      (on-policy):   target = r + γ · Q(s', a'_实际执行)

动机来自我们自己的实验数据：v5 在训练时（ε=0.05）自杀率约 70%，
纯贪婪评估时仅 3.5%。这说明 off-policy 学到的是"假设之后每步都
完美执行"的最优路径，完全不考虑自身 5% 的探索噪声。SARSA 用实际
执行的动作更新，理论上会学出留有余量的安全策略（经典的 cliff
walking 例子）。

实现上的麻烦：在第 t 步调用 game_events_occurred 时，第 t+1 步的
动作还没选出来。因此采用延迟一步更新——把当前 transition 暂存在
self.pending，等下一步知道了实际动作再补做更新。
"""

import csv
import os
import pickle
from collections import namedtuple, deque
from typing import List

import numpy as np

import events as e
from .callbacks import (
    ACTIONS, ALLOWED_IDX, MODEL_PATH, DIR_ORDER,
    state_to_features, q_values,
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'next_action', 'reward'))

# --- 超参数：与表格 Q-learning 版保持一致，便于受控对比 ---
TRANSITION_HISTORY_SIZE = 10000
ALPHA = 0.0
GAMMA = 0.9
EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY_ROUNDS = 8000

# --- 自定义事件（与 Q-learning 版完全相同）---
MOVED_TOWARDS_TARGET = "MOVED_TOWARDS_TARGET"
MOVED_AWAY_FROM_TARGET = "MOVED_AWAY_FROM_TARGET"
ESCAPED_DANGER = "ESCAPED_DANGER"
STAYED_IN_DANGER = "STAYED_IN_DANGER"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"
SUICIDAL_BOMB = "SUICIDAL_BOMB"
USEFUL_BOMB = "USEFUL_BOMB"
WASTED_BOMB = "WASTED_BOMB"

STATS_PATH = os.path.join(os.path.dirname(__file__), "training_stats.csv")


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.epsilon = EPS_START
    self.round_counter = 0
    self.train_last_dir = 4
    self.pending = None          # 等待 next_action 的上一步转移

    _reset_round_stats(self)

    with open(STATS_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "round", "epsilon", "score", "steps", "coins",
            "invalid_actions", "killed_self", "q_states", "total_reward",
            "bombs", "suicidal_bombs",
        ])
    self.logger.info(f"Training stats -> {STATS_PATH}")


def _reset_round_stats(self):
    self.stat_steps = 0
    self.stat_coins = 0
    self.stat_invalid = 0
    self.stat_reward = 0.0
    self.stat_bombs = 0
    self.stat_suicidal = 0


def add_custom_events(self, old_features, self_action, new_features, events):
    if old_features is None:
        return

    old_danger = old_features[4]
    escape_dir = old_features[5]
    target_dir = old_features[6]
    bomb_safe = old_features[7]

    if self_action == 'BOMB':
        if not old_features[8]:
            return
        if not bomb_safe:
            events.append(SUICIDAL_BOMB)
        elif not old_features[9]:
            events.append(WASTED_BOMB)
        elif old_danger == 0:
            events.append(USEFUL_BOMB)
        return

    if old_danger > 0:
        if escape_dir < 4 and self_action == ACTIONS[escape_dir]:
            events.append(ESCAPED_DANGER)
        else:
            events.append(STAYED_IN_DANGER)
        return

    if new_features is not None and new_features[4] > 0:
        events.append(MOVED_INTO_DANGER)

    if target_dir < 4:
        if self_action == ACTIONS[target_dir]:
            events.append(MOVED_TOWARDS_TARGET)
        elif self_action in ACTIONS[:4]:
            events.append(MOVED_AWAY_FROM_TARGET)


def sarsa_update(self, old_features, self_action, new_features, next_action, reward):
    """SARSA 更新：
        Q(s,a) <- Q(s,a) + α [ r + γ Q(s', a') - Q(s,a) ]

    a' 是实际执行的下一个动作（含探索时的随机动作），而不是
    Q-learning 的 max。终止状态或没有下一个动作时，γ 项取 0。
    """
    if old_features is None or self_action is None:
        return

    a = ACTIONS.index(self_action)
    q_old = q_values(self.model, old_features)

    if new_features is None or next_action is None:
        target = reward
    else:
        q_next = q_values(self.model, new_features)
        target = reward + GAMMA * q_next[ACTIONS.index(next_action)]

    q_old[a] += ALPHA * (target - q_old[a])


def _flush_pending(self, next_action):
    """把暂存的上一步转移用刚知道的 next_action 补做更新。"""
    if self.pending is None:
        return
    s, a, s_next, r = self.pending
    self.transitions.append(Transition(s, a, s_next, next_action, r))
    sarsa_update(self, s, a, s_next, next_action, r)
    self.pending = None


def _update_stats(self, reward, events):
    self.stat_steps += 1
    self.stat_reward += reward
    self.stat_coins += events.count(e.COIN_COLLECTED)
    if e.INVALID_ACTION in events:
        self.stat_invalid += 1
    if e.BOMB_DROPPED in events:
        self.stat_bombs += 1
    if SUICIDAL_BOMB in events:
        self.stat_suicidal += 1


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    prev_dir = getattr(self, 'train_last_dir', 4)
    old_features = state_to_features(old_game_state, prev_dir)

    cur_dir = DIR_ORDER.index(self_action) if self_action in DIR_ORDER else 4
    new_features = state_to_features(new_game_state, cur_dir)
    self.train_last_dir = cur_dir

    add_custom_events(self, old_features, self_action, new_features, events)
    reward = reward_from_events(self, events)

    # self_action 正是上一步转移所缺的 a'
    _flush_pending(self, self_action)

    # 本步的转移要等下一步才知道 a'，先暂存
    self.pending = (old_features, self_action, new_features, reward)

    _update_stats(self, reward, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    last_features = state_to_features(last_game_state,
                                      getattr(self, 'train_last_dir', 4))

    add_custom_events(self, last_features, last_action, None, events)
    reward = reward_from_events(self, events)

    # 先把上一步补完，last_action 就是它缺的 a'
    _flush_pending(self, last_action)

    # 终止转移：没有下一个动作，γ 项为 0
    self.transitions.append(Transition(last_features, last_action, None, None, reward))
    sarsa_update(self, last_features, last_action, None, None, reward)
    self.pending = None

    _update_stats(self, reward, events)

    self.round_counter += 1
    frac = min(1.0, self.round_counter / EPS_DECAY_ROUNDS)
    self.epsilon = EPS_START + frac * (EPS_END - EPS_START)

    with open(STATS_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            self.round_counter,
            round(self.epsilon, 4),
            last_game_state['self'][1],
            self.stat_steps,
            self.stat_coins,
            self.stat_invalid,
            int(e.KILLED_SELF in events),
            len(self.model),
            round(self.stat_reward, 2),
            self.stat_bombs,
            self.stat_suicidal,
        ])
    _reset_round_stats(self)
    self.train_last_dir = 4

    if ALPHA > 0:                     # α=0 表示评估模式，不写盘
        with open(MODEL_PATH, "wb") as file:
            pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> float:
    """与表格 Q-learning 版 v7 完全一致，保证对比是受控的。"""
    game_rewards = {
        e.COIN_COLLECTED: 30,
        e.KILLED_OPPONENT: 30,
        e.KILLED_SELF: -60,
        e.GOT_KILLED: -40,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 5,
        e.INVALID_ACTION: -3,
        e.WAITED: -1,

        SUICIDAL_BOMB: -20,
        WASTED_BOMB: -3,
        USEFUL_BOMB: 1,
        ESCAPED_DANGER: 5,
        STAYED_IN_DANGER: -6,
        MOVED_INTO_DANGER: -4,
        MOVED_TOWARDS_TARGET: 1,
        MOVED_AWAY_FROM_TARGET: -1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum