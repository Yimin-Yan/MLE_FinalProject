import os
import pickle
import random
from collections import deque

import torch
import torch.nn as nn
import numpy as np

from .DQN import Net, MEMORY_CAPACITY, N_STATES, LR

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EPSILON = 0.9
EPSILON_END = 0.1

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    self.num_actions=6

    self.actions = ACTIONS
    self.learning_rate = 0.1
    self.discount_factor = 0.9
    self.epsilon = EPSILON  # Greedy strategy values
    self.num_actions = len(ACTIONS)
    self.features = []
    self.eval_net, self.target_net = Net(), Net()  # Create two neural networks with Net
    self.learn_step_counter = 0  # for target updating
    self.memory_counter = 0  # for storing memory
    self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # Initialize the memory, one line represents a transition
    self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # Using the Adam Optimizer
    self.loss_func = nn.MSELoss()  # Use the mean square loss function (loss(xi, yi)=(xi-yi)^2)
    self.episode_reward_sum = 0  # Initialize the total reward for the episode corresponding to this loop

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        np.random.seed()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.eval_net = pickle.load(file)

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    # Gather information about the game state
    _, _, _, state = game_state['self']
    if self.train:
        random_prob = self.epsilon
    else:
        random_prob = EPSILON_END

    state = torch.unsqueeze(torch.FloatTensor(state), 0)  # Convert state to 32-bit floating point form and add dimension of dimension 1 at dim=0
    if random.random() < random_prob:    # Select actions randomly
        action = ACTIONS[np.random.randint(0, 6)]
    else:   # Select the optimal action
        actions_value = self.eval_net.forward(state)  # The action value is obtained by forward propagating the input state STATE to the evaluation network
        action = torch.max(actions_value, 1)[1].data.numpy()  # Output the index of the maximum value of each row and transform it into a numpy ndarray
        # print(actions_value)
        action = ACTIONS[action[0]]
    # print(action)
    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    # if game_state is None:
    #     return None
    #
    # # For example, you could construct several channels of equal shape, ...
    # channels = []
    # channels.append(...)
    # # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # # and return them as a vector
    # return stacked_channels.reshape(-1)

    if game_state is None:
        return None

        # Initialize an empty list to store feature channels
    channels = []

    # Create a channel to represent the positions of crates
    crates = game_state['field'] == 1  # Assuming 1 represents crates in the field
    channels.append(crates.astype(int))

    # Create a channel to represent the positions of coins
    coins = game_state['coins']
    coin_channel = np.zeros_like(crates)  # Initialize a channel with zeros
    for coin in coins:
        x, y = coin
        coin_channel[x, y] = 1
    channels.append(coin_channel)

    walls = game_state['field'] == -1  # Assuming -1 represents walls in the field
    walls_channel = walls.astype(int)
    channels.append(walls_channel)

    _, _, bomb_left, (player_x, player_y) = game_state['self']
    player_position_channel = np.zeros_like(crates)
    player_position_channel[player_x, player_y] = 1
    bomb_left_channel = np.zeros_like(crates)
    bomb_left_channel[bomb_left] = 1
    channels.append(player_position_channel)
    channels.append(bomb_left_channel)

    bombs = game_state['bombs']
    bomb_channel = np.zeros_like(crates)

    for bomb in bombs:
        (x, y), _ = bomb
        bomb_channel[x, y] = 1
    channels.append(bomb_channel)

    # Stack the channels to form a feature tensor
    stacked_channels = np.stack(channels, axis=-1)

    # Flatten the feature tensor into a vector
    flattened_features = stacked_channels.reshape(-1)

    return flattened_features