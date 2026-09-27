from collections import namedtuple, deque
import numpy as np
import pickle
import os
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS, STATE, features_to_index

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Custom Events
DECREASED_DISTANCE = "DIST_DECREASED"
INCREASED_DISTANCE = "DIST_INCREASED"
BOMB_DROPPED_CORNER = "BOMB_CORNER"
BOMB_NEAR_CRATE = 'BOMB_NEAR_CRATE'
BOMB_NOT_NEAR_CRATE = 'BOMB_NOT_NEAR_CRATE'
MOVED_AWAY_FROM_BOMB = 'MOVED_AWAY_FROM_BOMB'
MOVED_TO_BOMB = 'MOVED_TO_BOMB'
DEAD_END = 'DEAD_END'
NOT_DEAD_END = 'NOT_DEAD_END'

# each direction has 2^4 = 16 states, there are 4 direction so 16^4 states, and the last  state is if the agent can
# drop a bomb or not.
TOTAL_STATES = 16 ** 4 * 2

# Keeps the previous states and actions
PREV_Q = {}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.debug("Parameters set up")
    self.alpha = 0.1
    self.gamma = 0.8

    if not os.path.isfile("Q_sa_upload.npy"):
        self.logger.info("Setting up Q_sa_upload function")
        construct_q_table(TOTAL_STATES)
        with open('Q_sa_upload.npy', 'rb') as f:
            self.q_sa = np.load(f, allow_pickle=True)
    else:
        self.logger.info("Loading Q_sa_upload function from saved state.")
        with open('Q_sa_upload.npy', 'rb') as f:
            self.q_sa = np.load(f, allow_pickle=True)
    self.q_sa = self.q_sa.tolist()
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # Add custom events
    if old_game_state is None and self_action == 'BOMB':
        events.append(BOMB_DROPPED_CORNER)
    if old_game_state is not None:
        events = bomb_check(new_game_state, self_action, events)
        if np.array(new_game_state['coins']).T.size > 0:
            events = coin_dist_check(old_game_state, new_game_state, events)
        if np.array(new_game_state['bombs']).T.size > 0:
            events = bomb_dist_check(old_game_state, new_game_state, events)
            events = dead_end_check2(old_game_state, new_game_state, events)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(self, events)
    update_table(self, old_game_state, self_action, new_game_state, reward)
    if old_game_state is not None:
        self.logger.debug(
            f"Q after reward: {list(self.q_sa[features_to_index(state_to_features(old_game_state))].values())}")
        self.logger.debug(f"Index: {features_to_index(state_to_features(old_game_state))}")
    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param events: The events that occurred in `old_game_state`
    :param last_action: The action that you took in the last round.
    :param last_game_state: The state that was passed to the last call of `act`.
    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    reward = reward_from_events(self, events)
    update_table(self, last_game_state, last_action, None, reward)
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward))
    # Store the model
    for s in self.q_sa:
        for a in self.q_sa[s]:
            self.q_sa[s][a] = round(self.q_sa[s][a], 5)

    with open('Q_sa_upload.npy', 'wb') as f:
        np.save(f, self.q_sa)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.WAITED: -0.1,
        e.INVALID_ACTION: -1.8,
        e.BOMB_EXPLODED: 0.14,
        e.BOMB_DROPPED: +0.1,
        # e.CRATE_DESTROYED: 0.02,
        # e.COIN_FOUND: 0.02,
        e.COIN_COLLECTED: 0.7,
        e.KILLED_OPPONENT: 1,
        e.KILLED_SELF: -1,
        e.GOT_KILLED: -1,
        e.OPPONENT_ELIMINATED: 0.5,
        e.SURVIVED_ROUND: 0.2,
        # custom events
        e.DECREASED_DISTANCE: 0.2,
        e.INCREASED_DISTANCE: -0.2,
        e.BOMB_DROPPED_CORNER: -0.8,
        e.BOMB_NEAR_CRATE: 0.6,
        e.BOMB_NOT_NEAR_CRATE: -0.4,
        e.MOVED_AWAY_FROM_BOMB: 0.4,
        e.MOVED_TO_BOMB: -0.6,
        e.DEAD_END: -0.9,
        e.NOT_DEAD_END: 0.7
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def update_table(self, old_game_state, action, new_game_state, reward):
    """
    Cal

    :param reward: The reward for the action after the previous game state.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param self: The same object that is passed to all of your callbacks.
    """
    global PREV_Q
    if old_game_state is None:
        old_state_idx = 0
    else:
        old_state_idx = features_to_index(state_to_features(old_game_state))

    if new_game_state is None:
        PREV_Q[len(PREV_Q)] = [old_state_idx, action]
        # If the last reward is positive distribute it to previous state actions
        if reward > 0:
            for old_ind in range(len(PREV_Q) - 1):
                model_old_q_value = self.q_sa[PREV_Q[old_ind][0]][PREV_Q[old_ind][1]]
                old_q = self.alpha * (
                        reward / len(PREV_Q) - self.gamma * model_old_q_value)
                self.q_sa[PREV_Q[old_ind][0]][PREV_Q[old_ind][1]] += old_q
        old_state_q_value = self.q_sa[old_state_idx][action]
        old_q_value = self.alpha * (
                reward / 3 - old_state_q_value)
    else:
        # Q-learning part
        new_state_idx = features_to_index(state_to_features(new_game_state))
        if reward != -0.08:  # excludes the invalid actions
            PREV_Q[len(PREV_Q)] = [new_state_idx, action]

        new_state_q_value = list(self.q_sa[new_state_idx].values())
        old_state_q_value = self.q_sa[old_state_idx][action]

        old_q_value = self.alpha * (
                reward + self.gamma * new_state_q_value[np.argmax(new_state_q_value)] -
                old_state_q_value)

    self.q_sa[old_state_idx][action] += old_q_value

    # Below part adds the same reward when the agent is in the current state's symmetry and takes the symmetric action.
    # This part has fasten the learning process.
    if old_game_state is not None:
        if action == 'LEFT':
            left_state = state_to_features(old_game_state)[:4]
            right_state = state_to_features(old_game_state)[4:8]
            rotated_state = np.array([right_state] + [left_state] + [state_to_features(old_game_state)[8:]])
            rotated_state = np.concatenate(rotated_state, axis=None)
            self.q_sa[features_to_index(rotated_state)]['RIGHT'] += old_q_value

            up_state = state_to_features(old_game_state)[8:12]
            down_state = state_to_features(old_game_state)[12:16]
            rotated_state_vert_1 = np.array([state_to_features(old_game_state)[:8]] + [down_state] + [up_state] + [
                state_to_features(old_game_state)[16]])
            rotated_state_vert_1 = np.concatenate(rotated_state_vert_1, axis=None)
            self.q_sa[features_to_index(rotated_state_vert_1)]['LEFT'] += old_q_value

            up_state = state_to_features(old_game_state)[8:12]
            down_state = state_to_features(old_game_state)[12:16]
            rotated_state_vert_2 = np.array([rotated_state[:8]] + [down_state] + [up_state] + [
                state_to_features(old_game_state)[16]])
            rotated_state_vert_2 = np.concatenate(rotated_state_vert_2, axis=None)
            self.q_sa[features_to_index(rotated_state_vert_2)]['RIGHT'] += old_q_value
        elif action == 'RIGHT':
            left_state = state_to_features(old_game_state)[:4]
            right_state = state_to_features(old_game_state)[4:8]
            rotated_state = np.array([right_state] + [left_state] + [state_to_features(old_game_state)[8:]])
            rotated_state = np.concatenate(rotated_state, axis=None)
            self.q_sa[features_to_index(rotated_state)]['LEFT'] += old_q_value

            up_state = state_to_features(old_game_state)[8:12]
            down_state = state_to_features(old_game_state)[12:16]
            rotated_state_vert_1 = np.array([state_to_features(old_game_state)[:8]] + [down_state] + [up_state] + [
                state_to_features(old_game_state)[16]])
            rotated_state_vert_1 = np.concatenate(rotated_state_vert_1, axis=None)
            self.q_sa[features_to_index(rotated_state_vert_1)]['RIGHT'] += old_q_value

            up_state = state_to_features(old_game_state)[8:12]
            down_state = state_to_features(old_game_state)[12:16]
            rotated_state_vert_2 = np.array([rotated_state[:8]] + [down_state] + [up_state] + [
                state_to_features(old_game_state)[16]])
            rotated_state_vert_2 = np.concatenate(rotated_state_vert_2, axis=None)
            self.q_sa[features_to_index(rotated_state_vert_2)]['LEFT'] += old_q_value
        elif action == 'UP':
            up_state = state_to_features(old_game_state)[8:12]
            down_state = state_to_features(old_game_state)[12:16]
            rotated_state = np.array([state_to_features(old_game_state)[:8]] + [down_state] + [up_state] + [
                state_to_features(old_game_state)[16]])
            rotated_state = np.concatenate(rotated_state, axis=None)
            self.q_sa[features_to_index(rotated_state)]['DOWN'] += old_q_value

            left_state = state_to_features(old_game_state)[:4]
            right_state = state_to_features(old_game_state)[4:8]
            rotated_state_hor_1 = np.array([right_state] + [left_state] + [state_to_features(old_game_state)[8:]])
            rotated_state_hor_1 = np.concatenate(rotated_state_hor_1, axis=None)
            self.q_sa[features_to_index(rotated_state_hor_1)]['DOWN'] += old_q_value

            left_state = state_to_features(old_game_state)[:4]
            right_state = state_to_features(old_game_state)[4:8]
            rotated_state_hor_2 = np.array([right_state] + [left_state] + [rotated_state[8:]])
            rotated_state_hor_2 = np.concatenate(rotated_state_hor_2, axis=None)
            self.q_sa[features_to_index(rotated_state_hor_2)]['UP'] += old_q_value
        elif action == 'DOWN':
            up_state = state_to_features(old_game_state)[8:12]
            down_state = state_to_features(old_game_state)[12:16]
            rotated_state = np.array([state_to_features(old_game_state)[:8]] + [down_state] + [up_state] + [
                state_to_features(old_game_state)[16]])
            rotated_state = np.concatenate(rotated_state, axis=None)
            self.q_sa[features_to_index(rotated_state)]['UP'] += old_q_value

            left_state = state_to_features(old_game_state)[:4]
            right_state = state_to_features(old_game_state)[4:8]
            rotated_state_hor_1 = np.array([right_state] + [left_state] + [state_to_features(old_game_state)[8:]])
            rotated_state_hor_1 = np.concatenate(rotated_state_hor_1, axis=None)
            self.q_sa[features_to_index(rotated_state_hor_1)]['DOWN'] += old_q_value

            left_state = state_to_features(old_game_state)[:4]
            right_state = state_to_features(old_game_state)[4:8]
            rotated_state_hor_2 = np.array([right_state] + [left_state] + [rotated_state[8:]])
            rotated_state_hor_2 = np.concatenate(rotated_state_hor_2, axis=None)
            self.q_sa[features_to_index(rotated_state_hor_2)]['UP'] += old_q_value
        elif action == 'WAIT' or action == 'BOMB':
            up_state = state_to_features(old_game_state)[8:12]
            down_state = state_to_features(old_game_state)[12:16]
            rotated_state = np.array([state_to_features(old_game_state)[:8]] + [down_state] + [up_state] + [
                state_to_features(old_game_state)[16]])
            rotated_state = np.concatenate(rotated_state, axis=None)
            self.q_sa[features_to_index(rotated_state)][action] += old_q_value

            left_state = state_to_features(old_game_state)[:4]
            right_state = state_to_features(old_game_state)[4:8]
            rotated_state_hor_1 = np.array([right_state] + [left_state] + [state_to_features(old_game_state)[8:]])
            rotated_state_hor_1 = np.concatenate(rotated_state_hor_1, axis=None)
            self.q_sa[features_to_index(rotated_state_hor_1)][action] += old_q_value

            left_state = state_to_features(old_game_state)[:4]
            right_state = state_to_features(old_game_state)[4:8]
            rotated_state_hor_2 = np.array([right_state] + [left_state] + [rotated_state[8:]])
            rotated_state_hor_2 = np.concatenate(rotated_state_hor_2, axis=None)
            self.q_sa[features_to_index(rotated_state_hor_2)][action] += old_q_value

    # Give an upper bound otherwise, values may go to infinity and also it takes less time to calculate small numbers.
    if self.q_sa[old_state_idx][action] > 1:
        for act in ACTIONS:
            if self.q_sa[old_state_idx][act] > 1:
                self.q_sa[old_state_idx][act] = self.q_sa[old_state_idx][act] / self.q_sa[old_state_idx][action]


def construct_q_table(state_count):
    Q = {}
    for s in range(state_count):
        Q[s] = {}
        for a in ACTIONS:
            if a == "WAIT" or a == "BOMB":
                Q[s][a] = 0.2
            else:
                Q[s][a] = 0.2

    with open('Q_sa_upload.npy', 'wb') as f:
        np.save(f, Q)


def coin_dist_check(old_game_state, new_game_state, events):
    """
    This function checks the distance between the agent and the coins which are closer than 5 blocks. If the distance
    between the agent and a coin is increased then the event INCREASED_DISTANCE, otherwise DECREASED_DISTANCE is appended
    to the events.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param new_game_state: The state the agent is in now.
    :return events: The events after coin distance check.
    """
    old_coins = np.array(old_game_state["coins"]).T
    new_coins = np.array(new_game_state["coins"]).T
    old_self_cor_channel = np.array(old_game_state["self"][3]).T
    new_self_cor_channel = np.array(new_game_state["self"][3]).T
    old_cor_turned = old_self_cor_channel.T
    if old_coins.size != 0 and new_coins.size != 0:
        if old_coins.size > new_coins.size:
            max_coin = old_coins
            min_coin = new_coins
        else:
            max_coin = new_coins
            min_coin = old_coins
        for ind in range(len(max_coin[0])):
            old_dist_coin = np.sum(np.abs(max_coin[:, ind] - old_cor_turned))
            if old_dist_coin < 4:
                for index in range(len(min_coin[0])):
                    if max_coin[0, ind] == min_coin[0, index] and max_coin[1, ind] == min_coin[1, index]:
                        max_dist = np.abs((max_coin[0, ind] - old_self_cor_channel[0])) + np.abs(
                            (max_coin[1, ind] - old_self_cor_channel[1]))
                        min_dist = np.abs((min_coin[0, index] - new_self_cor_channel[0])) + np.abs(
                            (min_coin[1, index] - new_self_cor_channel[1]))
                        if min_dist < max_dist and max_coin.all() == old_coins.all():
                            events.append(DECREASED_DISTANCE)
                        elif min_dist > max_dist and max_coin.all() == old_coins.all():
                            events.append(INCREASED_DISTANCE)
                        elif min_dist < max_dist and min_coin.all() == old_coins.all():
                            events.append(INCREASED_DISTANCE)
                        elif min_dist > max_dist and min_coin.all() == old_coins.all():
                            events.append(DECREASED_DISTANCE)
    return events


def bomb_check(new_game_state, self_action, events):
    """
    This function checks if the agent dropped a bomb near a crate or not.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`.
    :param self_action: TThe action that you took.
    :param new_game_state: The state the agent is in now.
    :return events: The events after bomb check.
    """
    self_cord = np.array(new_game_state["self"][3])
    if (self_cord[1] == 1 and self_cord[0] == 1) or (self_cord[1] == 1 and self_cord[0] == 15) or \
            (self_cord[1] == 15 and self_cord[0] == 1) or (self_cord[1] == 15 and self_cord[0] == 15):
        if self_action == 'BOMB':
            events.append(BOMB_DROPPED_CORNER)
    field = np.array(new_game_state["field"]).T
    if (field[self_cord[1], self_cord[0] + 1] == 1) or (field[self_cord[1], self_cord[0] - 1] == 1) or \
            (field[self_cord[1] + 1, self_cord[0]] == 1) or (field[self_cord[1] - 1, self_cord[0]] == 1):
        if self_action == 'BOMB':
            events.append(BOMB_NEAR_CRATE)
    else:
        if self_action == 'BOMB':
            events.append(BOMB_NOT_NEAR_CRATE)
    return events


def bomb_dist_check(old_game_state, new_game_state, events):
    """
    This function checks the distance between the agent and the bombs which are closer than 5 blocks. If the distance
    between the agent and a bomb is increased then the event MOVED_AWAY_FROM_BOMB, otherwise MOVED_TO_BOMB is appended
    to the events.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param new_game_state: The state the agent is in now.
    :return events: The events after bomb distance check.
    """
    old_self_cor_channel = np.array(old_game_state["self"][3])
    new_self_cor_channel = np.array(new_game_state["self"][3])
    for bomb in old_game_state['bombs']:
        bomb_location = bomb[0]

        old_dist_bomb = np.sum(np.abs(bomb_location - old_self_cor_channel))
        new_dist_bomb = np.sum(np.abs(bomb_location - new_self_cor_channel))
        if old_dist_bomb < 5:
            if new_dist_bomb > old_dist_bomb:
                events.append('MOVED_AWAY_FROM_BOMB')
            elif new_dist_bomb <= old_dist_bomb:
                events.append('MOVED_TO_BOMB')
    return events


def dead_end_check2(old_game_state, new_game_state, events):
    """
    This function adds the events DEAD_END or NOT_DEAD_END by checking the chosen direction to escape after dropping a
    bomb.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param new_game_state: The state the agent is in now.
    :return events: The events after dead end check.
    """
    new_own_location = np.asarray(new_game_state['self'][-1])
    old_own_location = np.asarray(old_game_state['self'][-1])
    feat = state_to_features(old_game_state)
    chosen_direction = new_own_location - old_own_location
    turn_direction = chosen_direction[::-1]
    for bomb in old_game_state['bombs']:
        bomb_location = bomb[0]
        new_dist_bomb = np.sum(np.abs(bomb_location - new_own_location))
        if new_dist_bomb < 5:
            if np.sum(chosen_direction) != 0:
                if (turn_direction == [1, 0]).all() and feat[15] == 1 and (bomb_location == new_own_location).any():
                    events.append('DEAD_END')
                elif (turn_direction == [-1, 0]).all() and feat[11] == 1 and (bomb_location == new_own_location).any():
                    events.append('DEAD_END')
                elif (turn_direction == [0, 1]).all() and feat[7] == 1 and (bomb_location == new_own_location).any():
                    events.append('DEAD_END')
                elif (turn_direction == [0, -1]).all() and feat[3] == 1 and (bomb_location == new_own_location).any():
                    events.append('DEAD_END')
                else:
                    events.append('NOT_DEAD_END')
            elif (turn_direction == [0, 0]).all() and (bomb_location != new_own_location).all():
                events.append('NOT_DEAD_END')
    return events
