"""
train.py —— n-step DQN 版本

替换 agent_code/my_agent_dqn_nstep/train.py 的全部内容。
callbacks.py 与单步 DQN 版完全相同，不需要改动。

与 v3full 的唯一区别是 TD 目标的构造方式：

    1-step:  target = r_t + γ · max_a Q(s_{t+1}, a)
    n-step:  target = R_t^(n) + γ^n · max_a Q(s_{t+n}, a)
             其中 R_t^(n) = r_t + γ r_{t+1} + ... + γ^(n-1) r_{t+n-1}

动机来自 v1--v3full 的诊断：DQN 的无效动作极少（泛化生效），但自杀
率高达 63%--90%，说明它学会了空间约束却没学会时间规划。死亡惩罚
要经过 4 次单步 TD 传播才能到达"放弹"那一刻，每次都叠加网络的估
计误差。n-step 用前 n 步的真实观测奖励替代中间的估计，缩短了信用
分配的链条。

n 取 4，与炸弹倒计时长度一致 —— 这样"放弹"和"爆炸"落在同一个回报
窗口内。

其余超参数、奖励表、自定义事件与 v3full 完全一致，保证对比受控。
"""

import csv
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
                        ('state', 'action', 'next_state', 'reward', 'n'))

# --- 超参数（除 N_STEP 外与 v3full 相同）---
N_STEP = 4                 # 与炸弹倒计时一致
BUFFER_SIZE = 50000
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_ROUNDS = 40000
TARGET_UPDATE_STEPS = 500
LEARN_START = 1000
TRAIN_MODE = True          # 设为 False 做纯评估（不更新、不写盘）

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
    self.transitions = deque(maxlen=BUFFER_SIZE)
    self.nstep_buffer = deque(maxlen=N_STEP)     # 滑动窗口
    self.epsilon = EPS_START if TRAIN_MODE else 0.0
    self.round_counter = 0
    self.step_counter = 0
    self.train_last_dir = 4
    self.recent_loss = 0.0

    self.target_model = QNetwork()
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.criterion = nn.SmoothL1Loss()

    _reset_round_stats(self)

    with open(STATS_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "round", "epsilon", "score", "steps", "coins",
            "invalid_actions", "killed_self", "loss", "total_reward",
            "bombs", "suicidal_bombs",
        ])
    self.logger.info(f"Training stats (n-step={N_STEP}) -> {STATS_PATH}")


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


# ============================================================
# n-step 回报
# ============================================================

def _pop_nstep(self):
    """从滑动窗口最老的一条构造 n-step 转移。

    返回 (s_t, a_t, s_{t+k}, R_t^(k), k)，其中 k 是窗口当前长度。
    s_{t+k} 为 None 表示这段窗口内发生了终止，bootstrap 项应取 0。
    """
    s0, a0, _, _ = self.nstep_buffer[0]

    R = 0.0
    for i, (_, _, s_next, r) in enumerate(self.nstep_buffer):
        R += (GAMMA ** i) * r
        if s_next is None:            # 中途终止，不再累加
            return s0, a0, None, R, i + 1

    s_last = self.nstep_buffer[-1][2]
    return s0, a0, s_last, R, len(self.nstep_buffer)


def _push_transition(self, old_vec, action, new_vec, reward):
    """把单步经验压入滑动窗口；窗口满了就吐出一条 n-step 转移。"""
    if old_vec is None or action is None:
        return

    self.nstep_buffer.append((old_vec, action, new_vec, reward))

    if len(self.nstep_buffer) == N_STEP:
        s0, a0, s_n, R, k = _pop_nstep(self)
        self.transitions.append(Transition(s0, a0, s_n, R, k))
        self.nstep_buffer.popleft()


def _flush_nstep(self):
    """回合结束时清空窗口：剩下的每一条都要吐出来。

    这些转移的时间跨度不足 n 步，但它们恰好是最接近终局的样本 ——
    死亡惩罚就在其中，不能丢弃。
    """
    while self.nstep_buffer:
        s0, a0, s_n, R, k = _pop_nstep(self)
        self.transitions.append(Transition(s0, a0, s_n, R, k))
        self.nstep_buffer.popleft()


def optimize(self):
    """从经验回放采一批做梯度更新。

    与单步版的区别：bootstrap 项的折扣是 γ^k 而非 γ，
    k 是该条转移实际跨越的步数。
    """
    if len(self.transitions) < LEARN_START:
        return 0.0

    batch = random.sample(self.transitions, BATCH_SIZE)

    states = torch.from_numpy(np.stack([t.state for t in batch]))
    actions = torch.tensor([ACTIONS.index(t.action) for t in batch],
                           dtype=torch.int64).unsqueeze(1)
    returns = torch.tensor([t.reward for t in batch], dtype=torch.float32)
    steps = torch.tensor([t.n for t in batch], dtype=torch.float32)

    non_final = torch.tensor([t.next_state is not None for t in batch],
                             dtype=torch.bool)
    next_states = [t.next_state for t in batch if t.next_state is not None]

    self.model.train()
    q_pred = self.model(states).gather(1, actions).squeeze(1)

    q_next = torch.zeros(BATCH_SIZE)
    if next_states:
        with torch.no_grad():
            nxt = torch.from_numpy(np.stack(next_states))
            q_next[non_final] = self.target_model(nxt).max(1).values

    target = returns + (GAMMA ** steps) * q_next

    loss = self.criterion(q_pred, target)

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
    self.optimizer.step()
    self.model.eval()

    return float(loss.item())


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

    _push_transition(self,
                     features_to_vector(old_features),
                     self_action,
                     features_to_vector(new_features),
                     reward)

    _update_stats(self, reward, events)

    if TRAIN_MODE:
        self.recent_loss = optimize(self)
        self.step_counter += 1
        if self.step_counter % TARGET_UPDATE_STEPS == 0:
            self.target_model.load_state_dict(self.model.state_dict())


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    last_features = state_to_features(last_game_state,
                                      getattr(self, 'train_last_dir', 4))

    add_custom_events(self, last_features, last_action, None, events)
    reward = reward_from_events(self, events)

    # 终止转移：next_state 为 None
    _push_transition(self,
                     features_to_vector(last_features),
                     last_action,
                     None,
                     reward)
    _flush_nstep(self)

    _update_stats(self, reward, events)

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
    """归一化后的奖励表，与 v3full 完全一致。"""
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