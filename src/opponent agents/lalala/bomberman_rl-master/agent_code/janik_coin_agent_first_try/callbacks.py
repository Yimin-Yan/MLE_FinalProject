import os
import pickle
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'] # no bombs action for Coin agent

# Hyperparameter for Exploration vs exploitation
RANDOM_PROB = .1

# Hyperparameter random choices probabilities
RANDOM_CHOICES = [.25, .25, .25, .25, 0]

def setup(self):
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # 6 features with 3 possible states add up to 3^6 = 729 possible feature combinations
        # 5 possible actions if we don't allow dropping bombs
        self.model = np.zeros((729,5))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    if self.train and random.random() < RANDOM_PROB:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=RANDOM_CHOICES)

    self.logger.debug("Querying model for action.")
    features = state_to_features(game_state)
    rewardPredictions = self.model[features]
    bestIndex = np.argmax(rewardPredictions)

    return ACTIONS[int(bestIndex)]


def state_to_features(game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # relevant data in game state:
    position = (game_state['self'])[3]
    coins = game_state['coins']
    field = game_state['field']

    # Find the direction you need to move to get a coin:

    # Find nearest Coin:
    def distance(coin):
        return abs(coin[0] - position[0]) + abs(coin[1] - position[1])

    distances = list(map(distance, coins))
    def getIndex():
        if len(distances) == 0:
            return position
        else:

            nearest_coin_index = np.argmin(distances)
            return coins[nearest_coin_index]

    nearest_coin = getIndex()
    # add one feature horizontally and vertically
    channels = []

    channels.append(np.sign(nearest_coin[0] - position[0]))
    channels.append(np.sign(nearest_coin[1] - position[1]))

    # Look for walls in the way:
    channels.append(field[position[0] + 1][position[1]])
    channels.append(field[position[0] - 1][position[1]])
    channels.append(field[position[0]][position[1] + 1])
    channels.append(field[position[0]][position[1] - 1])

    # Note: All features have values in {-1, 0, 1}, so we compute a combinded index for all our features
    feature_value = 0
    factor = 1
    for channel in channels:
        feature_value += factor * (channel + 1)
        factor *= 3

    return feature_value
