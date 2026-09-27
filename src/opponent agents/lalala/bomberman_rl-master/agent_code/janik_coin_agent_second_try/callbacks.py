import os
import pickle
import random
from collections import deque

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'] # no bombs action for Coin agent

# Hyperparameter for Exploration vs exploitation
RANDOM_PROB = .2

# Hyperparameter random choices probabilities
RANDOM_CHOICES = [.25, .25, .25, .25, 0]

def setup(self):
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # 10 features with 3 possible states add up to 3^10 possible feature combinations
        # 5 possible actions if we don't allow dropping bombs
        featureSpaceSize = 3**8
        self.model = np.zeros((featureSpaceSize,5))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    if self.train and random.random() < RANDOM_PROB:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=RANDOM_CHOICES)

    self.logger.debug("Querying model for action.")
    features, _ = state_to_features(game_state)
    rewardPredictions = self.model[features]
    bestIndex = np.argmax(rewardPredictions)

    return ACTIONS[int(bestIndex)]


def state_to_features(game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # relevant data in game state:
    agentPosition = (game_state['self'])[3]
    coins = game_state['coins']
    field = game_state['field']

    # Find the direction you need to move to get a coin:

    # Find nearest Coin with BFS:
    def bfs():
        explored = [agentPosition]
        queue = [(agentPosition,0)]

        while queue:
            (position, distance) = queue.pop(0)
            neighbours = [[position[0] + 1, position[1]],[position[0] - 1, position[1]],[position[0], position[1] + 1],[position[0], position[1] - 1]]
            for neighbour in neighbours:
                if neighbour not in explored:
                    if field[neighbour[0]][neighbour[1]] == 0:
                        for coin in coins:
                            if neighbour[0] == coin[0] and neighbour[1] == coin[1]:
                                return neighbour, distance + 1
                        explored.append(neighbour)
                        queue.append((neighbour, distance + 1))
                    else:
                        explored.append(neighbour)
        return [10,10], 100.0
    nearest_coin, distance = bfs()

    # add one feature horizontally and vertically
    channels = []

    # add feature for walls around coin
    wallsAroundCoinFeature = 0
    if field[nearest_coin[0] + 1][nearest_coin[1]] != 0 and field[nearest_coin[0] - 1][nearest_coin[1]] != 0:
        wallsAroundCoinFeature = 1
    elif field[nearest_coin[0]][nearest_coin[1] + 1] != 0 and field[nearest_coin[0]][nearest_coin[1] - 1] != 0:
        wallsAroundCoinFeature = -1
    channels.append(wallsAroundCoinFeature)

    # add difference if coin is near or far in the respective direction
    #channels.append(np.sign(np.abs(nearest_coin[0] - agentPosition[0]) - 1))
    #channels.append(np.sign(np.abs(nearest_coin[1] - agentPosition[1]) - 1))

    channels.append(np.sign(nearest_coin[0] - agentPosition[0]))
    channels.append(np.sign(nearest_coin[1] - agentPosition[1]))

    # add a feature for differences in horizontal in vertical distance:
    channels.append(np.sign(np.abs(nearest_coin[0] - agentPosition[0]) - np.abs(nearest_coin[1] - agentPosition[1])))

    # Look for walls in the way:
    channels.append(field[agentPosition[0] + 1][agentPosition[1]])
    channels.append(field[agentPosition[0] - 1][agentPosition[1]])
    channels.append(field[agentPosition[0]][agentPosition[1] + 1])
    channels.append(field[agentPosition[0]][agentPosition[1] - 1])

    # Note: All features have values in {-1, 0, 1}, so we compute a combinded index for all our features
    feature_value = 0
    factor = 1
    for channel in channels:
        feature_value += factor * (channel + 1)
        factor *= 3

    return feature_value, distance
