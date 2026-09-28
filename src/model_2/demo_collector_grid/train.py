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
from .callbacks import state_to_features

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
    return          # 网格版不用自定义事件


def reward_from_events(self, events: List[str]) -> float:
    game_rewards = {
        e.COIN_COLLECTED: 3.0,
        e.KILLED_OPPONENT: 10.0,
        e.KILLED_SELF: -8.0,
        e.GOT_KILLED: -4.0,
        e.CRATE_DESTROYED: 0.05,
        e.COIN_FOUND: 0.5,
        e.INVALID_ACTION: -0.3,
        e.WAITED: -0.1,
    }
    return sum(game_rewards.get(ev, 0) for ev in events)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    old_features = state_to_features(old_game_state)

    new_features = state_to_features(new_game_state)

    reward = reward_from_events(self, events)

    if old_features is not None and self_action is not None:
        self.demos.append(Demo(old_features, self_action, new_features, reward))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    last_features = state_to_features(last_game_state)

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
