import os
import pickle
import random
from collections import deque

import numpy as np
import settings


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyperparameter for Exploration vs exploitation
RANDOM_PROB = .2

# Hyperparameter random choices probabilities
RANDOM_CHOICES = [.2, .2, .2, .2, .1, .1]

def setup(self):
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # 12 features with 3 possible states add up to 3^10 possible feature combinations
        # 6 possible actions if we don't allow dropping bombs
        featureSpaceSize = 214990848
        self.model = np.zeros((featureSpaceSize,6))
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
    bombPossible = (game_state['self'])[2]
    neighbourFields = [[agentPosition[0] + 1, agentPosition[1]],
        [agentPosition[0] - 1, agentPosition[1]],
        [agentPosition[0], agentPosition[1] + 1],
        [agentPosition[0], agentPosition[1] - 1]
    ]
    coins = game_state['coins']
    field = game_state['field']
    explosions = game_state['explosion_map']
    bombs = game_state['bombs']
    bombPositions = list(map(lambda x: x[0], bombs))
    creates = []
    for i in range(settings.COLS):
        for j in range(settings.ROWS):
            if field[i][j] == 1:
                creates.append([i,j])


    # Find the direction you need to move to get a coin:

    # Find nearest Coin with BFS:
    def bfs(objects):
        if len(objects) == 0:
            return None, 0

        explored = [agentPosition]
        queue = [(agentPosition,0)]

        while queue:
            (position, distance) = queue.pop(0)
            neighbours = [[position[0] + 1, position[1]],[position[0] - 1, position[1]],[position[0], position[1] + 1],[position[0], position[1] - 1]]
            for neighbour in neighbours:
                if neighbour not in explored:
                    if field[neighbour[0]][neighbour[1]] == 0:
                        for object in objects:
                            if neighbour[0] == object[0] and neighbour[1] == object[1]:
                                return neighbour, distance + 1
                        explored.append(neighbour)
                        queue.append((neighbour, distance + 1))
                    else:
                        explored.append(neighbour)
        return None, 1000
    nearest_coin, distance_coin = bfs(coins)
    nearest_create, _ = bfs(creates)
    nearest_bomb, _ = bfs(bombPositions)

    # add one feature horizontally and vertically

    def coinFeatures():
        """
        values -1, 0, 1 for each feature
        """
        if nearest_coin == None:
            return np.array([(0,3),(0,3),(0,3),(0,3)])

        # add feature for walls around coin
        wallsAroundCoinFeature = 0
        if field[nearest_coin[0] + 1][nearest_coin[1]] != 0 and field[nearest_coin[0] - 1][nearest_coin[1]] != 0:
            wallsAroundCoinFeature = 1
        elif field[nearest_coin[0]][nearest_coin[1] + 1] != 0 and field[nearest_coin[0]][nearest_coin[1] - 1] != 0:
            wallsAroundCoinFeature = 2

        return [
            (wallsAroundCoinFeature, 3),
            # add a feature for horizontal in vertical direction:
            (np.sign(nearest_coin[0] - agentPosition[0])+ 1, 3),
            (np.sign(nearest_coin[1] - agentPosition[1])+ 1, 3),
            # add a feature for differences in horizontal in vertical distance:
            (np.sign(np.abs(nearest_coin[0] - agentPosition[0]) - np.abs(nearest_coin[1] - agentPosition[1]))+ 1, 3)
            ]

    channels = coinFeatures()

    # Look for walls in the way:
    def fieldFeatures():
        """
        values: 
        0 air
        1 create 
        2 wall
        3 explosion 
        
        for each direction
        """
        fieldResults = []

        for neighbour in neighbourFields:
            if field[neighbour[0]][neighbour[1]] == -1:
                fieldResults.append((2,4))
            elif explosions[neighbour[0]][neighbour[1]] != 0:
                fieldResults.append((3,4))
            else:
                fieldResults.append((field[neighbour[0]][neighbour[1]],4))

        return fieldResults

    channels = channels + fieldFeatures()

    def createFeatures():
        """
        values -1, 0, 1 for each feature
        """
        if nearest_create == None:
            return [(0,3),(0,3),(0,3),(0,3)]

        # add feature for walls around create
        wallsAroundCreateFeature = 0
        if field[nearest_create[0] + 1][nearest_create[1]] != 0 and field[nearest_create[0] - 1][nearest_create[1]] != 0:
            wallsAroundCreateFeature = 1
        elif field[nearest_create[0]][nearest_create[1] + 1] != 0 and field[nearest_create[0]][nearest_create[1] - 1] != 0:
            wallsAroundCreateFeature = 2

        return [
            (wallsAroundCreateFeature, 3),
            # add a feature for horizontal in vertical direction:
            (np.sign(nearest_create[0] - agentPosition[0]) + 1, 3),
            (np.sign(nearest_create[1] - agentPosition[1]) + 1, 3),
            # add a feature for differences in horizontal in vertical distance:
            (np.sign(np.abs(nearest_create[0] - agentPosition[0]) - np.abs(nearest_create[1] - agentPosition[1])) + 1, 3)
            ]
    
    channels = channels + createFeatures()

    def safetyFeatures():
        """
        for the actions left, right, down, up and wait indicate how it increases the safety
        """
        safetyResults = [0,0,0,0,0]
        if len(bombs) == 0:
            return [(0,2),(0,2),(0,2),(0,2),(0,2)]

        bombFields = []

        for bomb in bombs:
            (bomb_x, bomb_y), _ = bomb
            bombFields.append((bomb_x, bomb_y))
            i = 1
            while i < 4 and field[bomb_x + i][bomb_y] != -1:
                bombFields.append([(bomb_x + i, bomb_y)])
                i += 1
            i = 1
            while i < 4 and field[bomb_x - i][bomb_y] != -1:
                bombFields.append([(bomb_x - i, bomb_y)])
                i += 1
            i = 1
            while i < 4 and field[bomb_x][bomb_y + i] != -1:
                bombFields.append([(bomb_x, bomb_y + i)])
                i += 1
            i = 1
            while i < 4 and field[bomb_x][bomb_y - i] != -1:
                bombFields.append([(bomb_x, bomb_y - i)])
                i += 1

        if agentPosition in bombFields:
            safetyResults[4] = 1
        for i in range(4):
            neighbour = neighbourFields[i]
            if neighbour in bombFields or field[neighbour[0]][neighbour[1]] != 0:
                safetyResults[i] = 1
        
        return [
            (safetyResults[0], 2),
            (safetyResults[1], 2),
            (safetyResults[2], 2),
            (safetyResults[3], 2),
            (safetyResults[4], 2),
            ]

    channels = channels + safetyFeatures()
            
    def nearestBombFeatures():
        """
        Check wether the nearest bomb is on the same line
        """
        if nearest_bomb == None:
            return [(0,2),(0,2)]

        return [
            # add a feature for horizontal in vertical direction:
            (np.abs(np.sign(nearest_bomb[0] - agentPosition[0])), 2),
            (np.abs(np.sign(nearest_bomb[1] - agentPosition[1])), 2)
            ]

    channels = channels + nearestBombFeatures()

    # Note: All features have values in {-1, 0, 1}, so we compute a combinded index for all our features
    feature_value = 0
    factor = 1
    for channel, channelSize in channels:
        feature_value += factor * channel
        factor *= channelSize
    
    #print("needed Size: " + str(factor))

    return int(feature_value), distance_coin