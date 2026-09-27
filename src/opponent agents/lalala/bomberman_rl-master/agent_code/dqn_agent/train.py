from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
from .DQN import MEMORY_CAPACITY, TARGET_REPLACE_ITER, BATCH_SIZE, GAMMA, N_STATES, LR

import torch
import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
ESCAPED_FROM_BOMB = "ESCAPED_FROM_BOMB"
STUCK_IN_A_LOOP = "STUCK_IN_A_LOOP"
DROPPED_USELESS_BOMB = "DROPPED_USELESS_BOMB"
WAITED_USEFULLY = "WAITED_USEFULLY"
WAITED_DANGEROUSLY = "WAITED_DANGEROUSLY"
CLOSED_TO_COIN = "CLOSED_TO_COIN"
PLACED_SAFE_BOMB = "PLACED_SAFE_BOMB"
FREQUENT_BOMB = "FREQUENT_BOMB"
action_dict = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

EPSILON_END = 0.1
EPSILON_DECAY = 0.9999

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.action_history = deque([], 20)



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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    _, _, _, old_state = old_game_state['self']
    _, _, _, new_state = new_game_state['self']

    # Idea: Add your own events to hand out rewards
    if len(self.action_history) > 0:
        if stuck_in_loop(self.action_history, self_action):
            events.append(STUCK_IN_A_LOOP)
    if is_safe_to_place_bomb(old_game_state):
        events.append(PLACED_SAFE_BOMB)
    if escape_from_bomb(old_game_state, events):
        events.append(ESCAPED_FROM_BOMB)
    if drop_bomb_feature(events):
        events.append(DROPPED_USELESS_BOMB)
    if self_action == "WAIT":
        if waited_feature(old_game_state, events) == 1:
            events.append(WAITED_USEFULLY)
        if len(self.action_history) > 0 and self.action_history[-1] == "BOMB":
            events.append(WAITED_DANGEROUSLY)
    if self_action == "BOMB" and self.action_history.count("BOMB") > 0:
        if get_length_between_two_bombs(self.action_history) < 4:
            events.append(FREQUENT_BOMB)
    # print(events)

    self.action_history.append(self_action)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    reward = reward_from_events(self, events)

    store_transition(self, old_state, self_action, reward, new_state)  # Store Samples
    self.episode_reward_sum += reward  # Gradually add the REWARD for each STEP within an EPISODE

    if self.memory_counter > MEMORY_CAPACITY:  # If the number of accumulated transitions exceeds the fixed capacity of the memory
        learn(self)        # Start to learn


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    _, _, _, state = last_game_state['self']

    # Idea: Add your own events to hand out rewards
    if len(self.action_history) > 0:
        if stuck_in_loop(self.action_history, last_action):
            events.append(STUCK_IN_A_LOOP)
    if is_safe_to_place_bomb(last_game_state):
        events.append(PLACED_SAFE_BOMB)
    if escape_from_bomb(last_game_state, events):
        events.append(ESCAPED_FROM_BOMB)
    if drop_bomb_feature(events):
        events.append(DROPPED_USELESS_BOMB)
    if last_action == "WAIT":
        if waited_feature(last_game_state, events) == 1:
            events.append(WAITED_USEFULLY)
        if len(self.action_history) > 0 and self.action_history[-1] == "BOMB":
            events.append(WAITED_DANGEROUSLY)
    if last_action == "BOMB" and self.action_history.count("BOMB") > 0:
        if get_length_between_two_bombs(self.action_history) < 4:
            events.append(FREQUENT_BOMB)
    # print(events)

    self.action_history.append(last_action)

    # new_events = get_events(self, old_game_state, self_action, events)
    reward = reward_from_events(self, events)

    store_transition(self, state, last_action, reward, state)  # 存储样本
    self.episode_reward_sum += reward  # Gradually add the REWARD for each STEP within an EPISODE

    if self.memory_counter > MEMORY_CAPACITY:  # If the number of accumulated transitions exceeds the fixed capacity of the memor
        learn(self)

    # update greedy epsilon
    if self.epsilon > EPSILON_END:
        self.epsilon *= EPSILON_DECAY

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.eval_net, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        ESCAPED_FROM_BOMB: 3,
        PLACED_SAFE_BOMB: 2,
        WAITED_USEFULLY: 1,
        STUCK_IN_A_LOOP: -5,
        DROPPED_USELESS_BOMB: -0.5,
        WAITED_DANGEROUSLY: -3,
        FREQUENT_BOMB: -1,

        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.CRATE_DESTROYED: 1,
        e.INVALID_ACTION: -3,

        e.KILLED_SELF: 0,
        e.GOT_KILLED: 0,
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.COIN_FOUND: 0,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

# Define the memory storage function
def store_transition(self, s, a, r, s_):
    transition = np.hstack((s, [action_dict[a], r], s_))     # Splice arrays horizontally
    # If the memory bank is full, the old data is overwritten.
    index = self.memory_counter % MEMORY_CAPACITY
    self.memory[index, :] = transition
    # self.memory[index] = [s, action_dict[a], r, s_]
    self.memory_counter += 1

def learn(self):
    # Update target network parameters
    if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # Triggered at first, then every 100 steps.
        self.target_net.load_state_dict(self.eval_net.state_dict())  # Assign the parameters of the evaluation network to the target network
    self.learn_step_counter += 1

    # Extracting batches of data from the memory
    sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
    b_memory = self.memory[sample_index, :]  # Extract the 32 transitions corresponding to the 32 indexes into b_memory
    b_s = torch.FloatTensor(b_memory[:, :N_STATES])
    b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
    b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
    b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

    # Obtain evaluation and objective values for 32 transitions and evaluat network parameter updates using loss functions and optimizers
    q_eval = self.eval_net(b_s).gather(1, b_a)
    q_next = self.target_net(b_s_).detach()
    q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
    loss = self.loss_func(q_eval, q_target)

    self.optimizer.zero_grad()  # Clear the remnants of the previous step and update the parameters
    loss.backward()     # Error back propagation
    self.optimizer.step()  # Update all parameters of the evaluation network

# At old_state, the agent is within the bomb's blast range, and at new_state, the agent has run out of range.
def escape_from_bomb(old_game_state, events):
    _, _, _, (x_old, y_old) = old_game_state['self']

    for (bomb_x, bomb_y), bomb_timer in old_game_state['bombs']:
        flag = False
        # There's one more step to explode.
        if bomb_timer == 1:
            if x_old == bomb_x:
                if (y_old == bomb_y - 3) or (y_old == bomb_y + 3):
                    flag = True
            elif y_old == bomb_y:
                if (x_old == bomb_x - 3) or (x_old == bomb_x + 3):
                    flag = True
            if flag and 'KILLED_SELF' not in events and 'GOT_KILLED' not in events:
                return True
    return False

# If a placed bomb is useless (if it destroys a chest or an enemy)
def drop_bomb_feature(events):
    if 'BOMB_EXPLODED' in events and 'KILLED_OPPONENT' not in events and 'CRATE_DESTROYED' not in events:
        return True
    return False

def is_safe_to_place_bomb(game_state: dict) -> bool:
    # Check if the agent's current position is safe to place a bomb
    x, y = game_state['self'][3]
    explosion_map = game_state['explosion_map']
    bomb_map = np.ones(explosion_map.shape) * 5

    # Calculate the explosion range of all existing bombs
    bombs = game_state['bombs']
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # Check if the agent's current position is not within the explosion range
    return bomb_map[x, y] >= 6

# If waiting is useful (dodged an enemy bomb)
def waited_feature(old_game_state, events):
    _, _, _, (x_old, y_old) = old_game_state['self']

    for (bomb_x, bomb_y), bomb_timer in old_game_state['bombs']:
        flag = False
        if bomb_timer == 1:
            if x_old == bomb_x:
                if (y_old == bomb_y - 4) or (y_old == bomb_y + 4):
                    flag = True
            elif y_old == bomb_y:
                if (x_old == bomb_x - 4) or (x_old == bomb_x + 4):
                    flag = True
            if flag and 'KILLED_SELF' not in events and 'GOT_KILLED' not in events:
                return 1
    return -1

def get_length_between_two_bombs(action_history):
    index = 0
    tag = 0

    for i in action_history:
        if i == "BOMB":
            index = tag
        tag += 1

    return len(action_history) - index - 1

def stuck_in_loop(action_history, action):
    # Two consecutive actions coincide
    if action == action_history[-1] or action_history.count(action) > 2:
        return True

    # Cycle between two coordinates
    if len(action_history) > 2:
        if action_history[-1] == action_history[-3] and action == action_history[-2]:
            return True

    return False