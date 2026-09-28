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
                        ('state', 'action', 'next_state', 'reward'))

# --- 超参数（这些都是要调的，报告里应该有对比实验）---
TRANSITION_HISTORY_SIZE = 10000   # 经验缓冲区大小
ALPHA = 0.1                       # 学习率
GAMMA = 0.9                       # 折扣因子
EPS_START = 1.0                   # 初始探索率
EPS_END = 0.05                 # 最终探索率
EPS_DECAY_ROUNDS = 8000           # 在多少回合内线性衰减到 EPS_END

# --- 自定义事件 ---
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
    """在 callbacks.py 的 setup 之后调用，初始化只在训练时需要的变量。"""
    self.round_counter = 0
    self.train_last_dir = 4
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.epsilon = EPS_START
    self.round_counter = 0

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
    """特征索引见 callbacks.state_to_features 的文档字符串。"""
    if old_features is None:
        return

    old_danger = old_features[4]
    escape_dir = old_features[5]
    target_dir = old_features[6]
    bomb_safe  = old_features[7]

    # --- 放弹判断（优先级最高）---
    if self_action == 'BOMB':
        bombs_available = old_features[8]
        if not bombs_available:
            return                          # 放不出来，交给 INVALID_ACTION 处理
        crate_gain = old_features[9]
        if not bomb_safe:
            events.append(SUICIDAL_BOMB)
        elif crate_gain == 0:
            events.append(WASTED_BOMB)
        elif old_danger == 0:
            events.append(USEFUL_BOMB)
        return

    # --- 危险中：只看逃生 ---
    if old_danger > 0:
        if escape_dir < 4 and self_action == ACTIONS[escape_dir]:
            events.append(ESCAPED_DANGER)
        else:
            events.append(STAYED_IN_DANGER)
        return                      # 逃命时不管金币

    # --- 安全时：走向目标，且别踏进危险 ---
    if new_features is not None and new_features[4] > 0:
        events.append(MOVED_INTO_DANGER)

    if target_dir < 4:
        if self_action == ACTIONS[target_dir]:
            events.append(MOVED_TOWARDS_TARGET)
        elif self_action in ACTIONS[:4]:
            events.append(MOVED_AWAY_FROM_TARGET)

def q_update(self, old_features, self_action, new_features, reward):
    """Q(s,a) <- Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
       终止状态（new_features 为 None）时 max 项取 0。"""
    if old_features is None or self_action is None:
        return

    a = ACTIONS.index(self_action)
    q_old = q_values(self.model, old_features)

    if new_features is None:
        target = reward
    else:
        q_next = q_values(self.model, new_features)
        target = reward + GAMMA * np.max(q_next[ALLOWED_IDX])

    q_old[a] += ALPHA * (target - q_old[a])


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """每步调用一次（最后一步除外）。在这里做 Q 更新。"""
    prev_dir = getattr(self, 'train_last_dir', 4)
    old_features = state_to_features(old_game_state, prev_dir)
    cur_dir = DIR_ORDER.index(self_action) if self_action in DIR_ORDER else 4
    new_features = state_to_features(new_game_state, cur_dir)
    self.train_last_dir = cur_dir

    add_custom_events(self, old_features, self_action, new_features, events)
    reward = reward_from_events(self, events)

    self.transitions.append(
        Transition(old_features, self_action, new_features, reward))
    q_update(self, old_features, self_action, new_features, reward)

    self.stat_steps += 1
    self.stat_reward += reward
    self.stat_coins += events.count(e.COIN_COLLECTED)
    if e.INVALID_ACTION in events:
        self.stat_invalid += 1
    if e.BOMB_DROPPED in events:
        self.stat_bombs += 1
    if SUICIDAL_BOMB in events:
        self.stat_suicidal += 1


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """每回合结束：处理最后一个 transition、衰减 ε、存盘、写统计。"""
    last_features = state_to_features(last_game_state, getattr(self, 'train_last_dir', 4))

    add_custom_events(self, last_features, last_action, None, events)
    reward = reward_from_events(self, events)

    self.transitions.append(Transition(last_features, last_action, None, reward))
    q_update(self, last_features, last_action, None, reward)   # 终止状态

    self.stat_steps += 1
    self.stat_reward += reward
    self.stat_coins += events.count(e.COIN_COLLECTED)
    if e.INVALID_ACTION in events:
        self.stat_invalid += 1

    # ε 线性衰减
    if e.BOMB_DROPPED in events:
        self.stat_bombs += 1
    if SUICIDAL_BOMB in events:
        self.stat_suicidal += 1
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
    """奖励塑形。这些数值需要做对比实验来调，不是拍脑袋定死的。"""
    game_rewards = {
        e.COIN_COLLECTED: 30,
        e.KILLED_OPPONENT: 30,
        e.KILLED_SELF: -60,
        e.GOT_KILLED: -40,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 5,
        e.INVALID_ACTION: -3,
        e.WAITED: -1,

        SUICIDAL_BOMB: -20,      # 放了没活路的弹
        USEFUL_BOMB: 1,          # 安全前提下主动放弹
        WASTED_BOMB: -3,
        ESCAPED_DANGER: 5,       # 朝逃生方向走
        STAYED_IN_DANGER: -6,    # 危险中没往正确方向跑
        MOVED_INTO_DANGER: -4,   # 从安全踏进危险
        MOVED_TOWARDS_TARGET: 1,
        MOVED_AWAY_FROM_TARGET: -1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum