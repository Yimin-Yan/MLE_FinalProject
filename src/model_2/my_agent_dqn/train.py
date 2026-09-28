"""
train.py —— DQN 版本

替换 agent_code/my_agent_dqn/train.py 的全部内容。

与表格版的差别：
  - Q 更新从「直接改表项」变成「小批量梯度下降」
  - 引入经验回放：每步从缓冲区随机采样，打破样本的时间相关性
  - 引入目标网络：TD target 用一个滞后的网络副本计算，
    否则目标随主网络一起漂移，训练很难稳定
自定义事件和奖励表与表格版保持一致，便于两个模型做受控对比。
"""

import csv
import copy
import os
import random
from collections import namedtuple, deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import events as e
from .callbacks import (
    ACTIONS, MODEL_PATH, DIR_ORDER,
    state_to_features, features_to_vector, QNetwork,
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# --- 超参数 ---
BUFFER_SIZE = 50000        # 经验回放缓冲区
BATCH_SIZE = 64            # 每步采样多少条做梯度更新
LEARNING_RATE = 5e-4
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_ROUNDS = 40000
TARGET_UPDATE_STEPS = 500  # 每多少步同步一次目标网络
LEARN_START = 1000         # 缓冲区攒够多少条才开始训练
TRAIN_MODE = True          # 设为 False 做纯评估（不更新、不写盘）

# --- 自定义事件（与表格版一致）---
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
    self.transitions = deque(maxlen=BUFFER_SIZE)
    self.epsilon = EPS_START if TRAIN_MODE else 0.0
    self.round_counter = 0
    self.step_counter = 0
    self.train_last_dir = 4
    self.recent_loss = 0.0

    # 目标网络：主网络的一个滞后副本
    self.target_model = QNetwork()
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.criterion = nn.SmoothL1Loss()      # Huber，比 MSE 对离群 TD 误差更稳

    _reset_round_stats(self)

    with open(STATS_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "round", "epsilon", "score", "steps", "coins",
            "invalid_actions", "killed_self", "loss", "total_reward",
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


# ============================================================
# 自定义事件（逻辑与表格版完全相同）
# ============================================================

def add_custom_events(self, old_features, self_action, new_features, events):
    if old_features is None:
        return

    old_danger = old_features[4]
    escape_dir = old_features[5]
    target_dir = old_features[6]
    bomb_safe = old_features[7]

    if self_action == 'BOMB':
        if not old_features[8]:          # 炸弹不可用，交给 INVALID_ACTION
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


# ============================================================
# 训练
# ============================================================

def optimize(self):
    """从经验回放中采一批，做一次梯度更新。返回本次 loss。"""
    if len(self.transitions) < LEARN_START:
        return 0.0

    batch = random.sample(self.transitions, BATCH_SIZE)

    states = torch.from_numpy(np.stack([t.state for t in batch]))
    actions = torch.tensor([ACTIONS.index(t.action) for t in batch],
                           dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)

    # 终止转移的 next_state 是 None，用掩码区分
    non_final = torch.tensor([t.next_state is not None for t in batch],
                             dtype=torch.bool)
    next_states = [t.next_state for t in batch if t.next_state is not None]

    self.model.train()
    q_pred = self.model(states).gather(1, actions).squeeze(1)

    # TD target 用目标网络计算，避免目标随主网络一起漂移
    q_next = torch.zeros(BATCH_SIZE)
    if next_states:
        with torch.no_grad():
            nxt = torch.from_numpy(np.stack(next_states))
            q_next[non_final] = self.target_model(nxt).max(1).values
    target = rewards + GAMMA * q_next

    loss = self.criterion(q_pred, target)

    self.optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪：TD 误差偶尔很大，不裁剪容易一步把权重打飞
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
    self.optimizer.step()
    self.model.eval()

    return float(loss.item())


def _record(self, old_features, self_action, new_features, events):
    """公共部分：加事件、算奖励、存 transition、更新统计。"""
    add_custom_events(self, old_features, self_action, new_features, events)
    reward = reward_from_events(self, events)

    old_vec = features_to_vector(old_features)
    new_vec = features_to_vector(new_features)
    if old_vec is not None and self_action is not None:
        self.transitions.append(
            Transition(old_vec, self_action, new_vec, reward))

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

    _record(self, old_features, self_action, new_features, events)

    if TRAIN_MODE:
        self.recent_loss = optimize(self)
        self.step_counter += 1
        if self.step_counter % TARGET_UPDATE_STEPS == 0:
            self.target_model.load_state_dict(self.model.state_dict())


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    last_features = state_to_features(last_game_state,
                                      getattr(self, 'train_last_dir', 4))
    _record(self, last_features, last_action, None, events)

    if TRAIN_MODE:
        self.recent_loss = optimize(self)

    self.round_counter += 1
    if TRAIN_MODE:
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
            round(self.recent_loss, 4),
            round(self.stat_reward, 2),
            self.stat_bombs,
            self.stat_suicidal,
        ])
    _reset_round_stats(self)
    self.train_last_dir = 4

    if TRAIN_MODE:
        torch.save(self.model.state_dict(), MODEL_PATH)


def reward_from_events(self, events: List[str]) -> float:
    """与表格版 v5/v7 使用的奖励表一致，便于两个模型做受控对比。"""
    game_rewards = {
        e.COIN_COLLECTED: 3.0,
        e.KILLED_OPPONENT: 3.0,
        e.KILLED_SELF: -6.0,
        e.GOT_KILLED: -4.0,
        e.CRATE_DESTROYED: 0.05,
        e.COIN_FOUND: 0.5,
        e.INVALID_ACTION: -0.3,
        e.WAITED: -0.1,

        SUICIDAL_BOMB: -2.0,
        WASTED_BOMB: -0.3,
        USEFUL_BOMB: 0.1,
        ESCAPED_DANGER: 0.5,
        STAYED_IN_DANGER: -0.6,
        MOVED_INTO_DANGER: -0.4,
        MOVED_TOWARDS_TARGET: 0.1,
        MOVED_AWAY_FROM_TARGET: -0.1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum