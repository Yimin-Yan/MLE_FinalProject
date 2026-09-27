from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS

import numpy as np

# Events
MOVED_BACK_AGAIN_EVENT = "MOVED_BACK_AGAIN"
DISTANCE_SMALLER = "DISTANCE_SMALLER"
DISTANCE_SAME = "DISTANCE_SAME"
DISTANCE_BIGGER = "DISTANCE_BIGGER"

# Trainig Hyperparams
LEARNING_RATE = 0.1
GAMMA = 0.5

def setup_training(self):
    self.logger.debug("Start training")
    self.lastMove = 'WAIT'
    self.currentDistance = 1000.0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    old_features, _ = state_to_features(old_game_state)
    new_features, distance_to_nearest_coin = state_to_features(new_game_state)

    if distance_to_nearest_coin < self.currentDistance:
        events.append(DISTANCE_SMALLER)
    elif distance_to_nearest_coin == self.currentDistance:
        events.append(DISTANCE_SAME)
    else:
        events.append(DISTANCE_BIGGER)

    # Add MOVED_BACK_AGAIN_EVENT
    for event in events:
        if (event, self.lastMove) in [('UP', 'DOWN'), ('DOWN', 'UP'), ('LEFT', 'RIGHT'), ('RIGHT', 'LEFT')]:
            events.append(MOVED_BACK_AGAIN_EVENT)

    self.currentDistance = distance_to_nearest_coin
    self.lastMove = self_action

    reward = reward_from_events(self, events)
    action = int(np.where(np.array(ACTIONS) == self_action)[0][0])


    expectedOldReward = self.model[old_features][action]
    expectedRewardAfterStep = np.amax(self.model[new_features])

    # "Q-Function" update
    self.model[old_features][action] = expectedOldReward + LEARNING_RATE * (reward + GAMMA * expectedRewardAfterStep - expectedOldReward)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)



def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 500,
        e.WAITED: -100,
        e.INVALID_ACTION: -200,
        MOVED_BACK_AGAIN_EVENT: -100,
        DISTANCE_BIGGER: -200,
        DISTANCE_SMALLER: 150,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum