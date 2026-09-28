"""
train.py —— 演示数据收集器

放进 agent_code/demo_collector/train.py

这个 agent 的 act 沿用 rule_based_agent 的启发式决策（不要改它），
train.py 只负责把它的行为记录成"专家演示"：用我们自己的
state_to_features 把每一步转成 (特征, 动作, 奖励, 下一特征)。

用途：DQfD 的预训练数据。rule_based_agent 会躲炸弹、收金币、炸箱子，
这些行为正是我们的 DQN 在前四万局（ε 衰减期）靠随机探索慢慢摸索的
东西 —— 直接拿现成的能省掉这一段。

跑法：
    python main.py play --no-gui --agents demo_collector \
        rule_based_agent rule_based_agent rule_based_agent \
        --train 1 --n-rounds 2000

产出 demonstrations.pkl，约十几万条转移。
"""

import os
import pickle
from collections import namedtuple
from typing import List

import events as e
from .callbacks import state_to_features, DIR_ORDER

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# 与 DQN 版一致的自定义事件，保证演示数据的奖励口径与后续训练相同
MOVED_TOWARDS_TARGET = "MOVED_TOWARDS_TARGET"
MOVED_AWAY_FROM_TARGET = "MOVED_AWAY_FROM_TARGET"
ESCAPED_DANGER = "ESCAPED_DANGER"
STAYED_IN_DANGER = "STAYED_IN_DANGER"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"
SUICIDAL_BOMB = "SUICIDAL_BOMB"
USEFUL_BOMB = "USEFUL_BOMB"
WASTED_BOMB = "WASTED_BOMB"
BOMB_NEAR_OPPONENT = "BOMB_NEAR_OPPONENT"

Demo = namedtuple('Demo', ('state', 'action', 'next_state', 'reward'))

OUT_PATH = os.path.join(os.path.dirname(__file__), "demonstrations.pkl")
SAVE_EVERY = 200          # 每多少回合落盘一次，防止中途崩溃丢数据


def setup_training(self):
    self.demos = []
    self.train_last_dir = 4
    self.round_counter = 0
    self.logger.info(f"Demo collector -> {OUT_PATH}")


def add_custom_events(self, old_features, self_action, new_features, events):
    """与 DQN 版完全相同的事件逻辑，保证奖励口径一致。"""
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
        elif old_features[13]:
            events.append(BOMB_NEAR_OPPONENT)
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


def reward_from_events(self, events: List[str]) -> float:
    """与 attack 版归一化后的奖励表完全一致。"""
    game_rewards = {
        e.COIN_COLLECTED: 3.0,
        e.KILLED_OPPONENT: 10.0,
        e.KILLED_SELF: -6.0,
        e.GOT_KILLED: -4.0,
        e.CRATE_DESTROYED: 0.05,
        e.COIN_FOUND: 0.5,
        e.INVALID_ACTION: -0.3,
        e.WAITED: -0.1,

        SUICIDAL_BOMB: -2.0,
        WASTED_BOMB: -0.3,
        USEFUL_BOMB: 0.1,
        BOMB_NEAR_OPPONENT: 1.0,
        ESCAPED_DANGER: 0.5,
        STAYED_IN_DANGER: -0.6,
        MOVED_INTO_DANGER: -0.4,
        MOVED_TOWARDS_TARGET: 0.1,
        MOVED_AWAY_FROM_TARGET: -0.1,
    }
    return sum(game_rewards.get(ev, 0) for ev in events)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    prev_dir = getattr(self, 'train_last_dir', 4)
    old_features = state_to_features(old_game_state, prev_dir)

    cur_dir = DIR_ORDER.index(self_action) if self_action in DIR_ORDER else 4
    new_features = state_to_features(new_game_state, cur_dir)
    self.train_last_dir = cur_dir

    add_custom_events(self, old_features, self_action, new_features, events)
    reward = reward_from_events(self, events)

    if old_features is not None and self_action is not None:
        self.demos.append(Demo(old_features, self_action, new_features, reward))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    last_features = state_to_features(last_game_state,
                                      getattr(self, 'train_last_dir', 4))

    add_custom_events(self, last_features, last_action, None, events)
    reward = reward_from_events(self, events)

    if last_features is not None and last_action is not None:
        self.demos.append(Demo(last_features, last_action, None, reward))

    self.train_last_dir = 4
    self.round_counter += 1

    if self.round_counter % SAVE_EVERY == 0:
        _dump(self)


def _dump(self):
    with open(OUT_PATH, "wb") as f:
        pickle.dump(self.demos, f)
    self.logger.info(f"round {self.round_counter}: {len(self.demos)} transitions saved")
    print(f"[demo_collector] round {self.round_counter}: "
          f"{len(self.demos)} transitions -> {OUT_PATH}")
