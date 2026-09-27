import os
import pickle
import random

import numpy as np
import settings


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyperparameter for Exploration vs exploitation
RANDOM_PROB = .05

# Hyperparameter random choices probabilities
RANDOM_CHOICES = [.2, .2, .2, .2, .1, .1]

AMOUNT_FEATURES = 22

def setup(self):
    if not (os.path.isfile("my-seen-features.pt") and os.path.isfile("my-feature-rewards.pt")):
        self.logger.info("Setting up model from scratch.")
        self.seenFeatures = np.ndarray((0, AMOUNT_FEATURES))
        self.featureRewards = np.ndarray((0, len(ACTIONS)))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-seen-features.pt", "rb") as file1:
            self.seenFeatures = pickle.load(file1)
        with open("my-feature-rewards.pt", "rb") as file2:
            self.featureRewards = pickle.load(file2)

def act(self, game_state: dict) -> str:

    features, _ = state_to_features(self, game_state)

    if self.train and random.random() < RANDOM_PROB:
        action = np.random.choice(ACTIONS, p=RANDOM_CHOICES)
        self.logger.debug(f'Randomly chose action {action} in step {game_state["step"]}')
        #self.logger.debug(f'Random choice for features {features}')
        return action
    #equiGameStates = equivalentGameStates(game_state)
    #primaryGameStateIndex = getPrimaryGameStateIndex(equiGameStates)

    #features, _ = state_to_features(equiGameStates[primaryGameStateIndex])

    actionsLeft = np.array([0,1,2,3,4,5])

    if self.seenFeatures.size != 0:
        possible_features_indices = np.where((self.seenFeatures == features).all(axis=1))
        if len(possible_features_indices[0]) != 0:
            rewardPredictions = self.featureRewards[int(possible_features_indices[0][0])]
            #self.logger.debug(f'Predicted for features {features} the rewards {rewardPredictions} for the features itself')
            if (not (rewardPredictions == 0).all(axis=0)):
                if not np.amax(rewardPredictions) == 0:
                    #self.logger.debug(f'Predicted for features {features} the rewards {rewardPredictions}')
                    bestIncices = np.argwhere(rewardPredictions == np.amax(rewardPredictions)).flatten()
                    bestIndex = np.random.choice(bestIncices)
                    if np.max(rewardPredictions) < 0:
                        self.logger.warn('negative best move')
                    action = ACTIONS[int(bestIndex)]
                    self.logger.debug(f'Chose action {action} in step {game_state["step"]} after feature was found')
                    return action
                    #return getActionsForEquivGameStates(ACTIONS[int(bestIndex)], primaryGameStateIndex)
                else:
                    bestIncices = np.argwhere(rewardPredictions == np.amax(rewardPredictions)).flatten()
                    newActionsLeft = bestIncices[np.in1d(bestIncices, actionsLeft)]
                    if len(newActionsLeft) <= 1:
                        if len(newActionsLeft) == 0:
                            bestIndex = np.random.choice(actionsLeft)
                        elif len(newActionsLeft) == 1:
                            bestIndex = newActionsLeft[0]
                        self.logger.debug(f'Best index chosen: {bestIndex}')
                        action = ACTIONS[int(bestIndex)]
                        self.logger.debug(f'Chose action {action} in step {game_state["step"]} after feature was found')
                        return action
                    else:
                        actionsLeft = newActionsLeft
                        self.logger.debug(f'New actionsLeft: {actionsLeft}')
                        

                        

        # find "nearest" feature:
        for i in range(1, AMOUNT_FEATURES):
            cut_at = AMOUNT_FEATURES - i
            possible_features_indices = np.where((self.seenFeatures[:, :cut_at] == features[:cut_at]).all(axis=1))
            if len(possible_features_indices[0]) != 0:
                pred = self.featureRewards[possible_features_indices[0]]
                #self.logger.debug(f'pred was {pred}')
                bestNearFeatureIndex = possible_features_indices[0][np.unravel_index(np.argmax(pred, axis=None), pred.shape)[0]]
                rewardPredictions = self.featureRewards[int(bestNearFeatureIndex)]
                #.debug(f'Features with similar beginning were {self.seenFeatures[possible_features_indices[0]]} at index {bestNearFeatureIndex} that should contain {possible_features_indices[0]}')
                if (not (rewardPredictions == 0).all(axis=0)):
                    if not np.amax(rewardPredictions) == 0:
                        #self.logger.debug(f'Predicted for features {features} the rewards {rewardPredictions} with feature distance {i}')
                        #self.logger.debug(f'Similar feature was {self.seenFeatures[bestNearFeatureIndex]}')
                        bestIncices = np.argwhere(rewardPredictions == np.amax(rewardPredictions)).flatten()
                        newActionsLeft = bestIncices[np.in1d(bestIncices, actionsLeft)]
                        if len(newActionsLeft) == 0:
                            newActionsLeft = actionsLeft
                        bestIndex = np.random.choice(newActionsLeft)
                        if np.max(rewardPredictions) < 0:
                            self.logger.warn('negative best move')
                        action = ACTIONS[int(bestIndex)]
                        self.logger.debug(f'Chose action {action} in step {game_state["step"]} with feature distance {i}')
                        return action
                    else:
                        bestIncices = np.argwhere(rewardPredictions == np.amax(rewardPredictions)).flatten()
                        newActionsLeft = bestIncices[np.in1d(bestIncices, actionsLeft)]
                        if len(newActionsLeft) <= 1:
                            if len(newActionsLeft) == 0:
                                bestIndex = np.random.choice(actionsLeft)
                            elif len(newActionsLeft) == 1:
                                bestIndex = newActionsLeft[0]
                            self.logger.debug(f'Best index chosen: {bestIndex}')
                            action = ACTIONS[int(bestIndex)]
                            self.logger.debug(f'Chose action {action} in step {game_state["step"]} after feature was found')
                            return action
                        else:
                            actionsLeft = newActionsLeft
                            self.logger.debug(f'New actionsLeft: {actionsLeft}')

    action = np.random.choice(ACTIONS, p=RANDOM_CHOICES)
    self.logger.debug(f'No action found: Randomly chose action {action} in step {game_state["step"]}')
    return action

def getPrimaryGameStateIndex(equiGameStates):
    currentPossibilities = []
    currentMin = 1000
    for i in range(8):
        state = equiGameStates[i]
        if state['self'][3][0] < currentMin:
            currentPossibilities = [i]
            currentMin = state['self'][3][0]
        elif state['self'][3][0] == currentMin:
            currentPossibilities.append(i)
    if len(currentPossibilities) == 1:
        return currentPossibilities[0]

    currentPossibilities2 = []
    currentMin2 = 1000
    for i in currentPossibilities:
        state = equiGameStates[i]
        if state['self'][3][0] < currentMin2:
            currentPossibilities2 = [i]
            currentMin = state['self'][3][0]
        elif state['self'][3][0] == currentMin2:
            currentPossibilities2.append(i)

    return currentPossibilities2[0]

def getActionsForEquivGameStates(action, index):
    if action == 'UP':
        choices = ['UP', 'UP', 'LEFT', 'RIGHT', 'DOWN', 'DOWN',  'RIGHT', 'LEFT']
        return choices[index]
    elif action == 'RIGHT':
        choices = ['LEFT', 'RIGHT', 'DOWN', 'DOWN',  'RIGHT', 'LEFT', 'UP', 'UP']
        return choices[index]
    elif action == 'DOWN':
        choices = ['DOWN', 'DOWN',  'RIGHT', 'LEFT', 'UP', 'UP', 'LEFT', 'RIGHT']
        return choices[index]
    elif action == 'LEFT':
        choices = ['RIGHT', 'LEFT', 'UP', 'UP', 'LEFT', 'RIGHT', 'DOWN', 'DOWN']
        return choices[index]
    elif action == 'WAIT':
        return 'WAIT'
    elif action == 'BOMB':
        return 'BOMB'

        
def equivalentGameStates(game_state: dict):
    """
    Using mirroring and rotation, there are 8 equivalent game states
    """
    state1 = mirrorGame(game_state)
    state2 = transposeGame(state1)
    state3 = mirrorGame(state2)
    state4 = transposeGame(state3)
    state5 = mirrorGame(state4)
    state6 = transposeGame(state5)
    state7 = mirrorGame(state6)

    return [game_state, state1, state2, state3, state4, state5, state6, state7]

def mirrorGame(game_state: dict):
    new_state = game_state.copy()

    new_state['field'] = np.fliplr(game_state['field'])
    new_state['explosion_map'] = np.fliplr(game_state['explosion_map'])

    def mirrorPosition(position):
        return settings.COLS - position[0], position[1]
    def mirrorBomb(bomb):
        return mirrorPosition(bomb[0]), bomb[1]
    def mirrorOther(other):
        return other[0], other[1], other[2], mirrorPosition(other[3])
    
    new_state['self'][3] = mirrorPosition(game_state['self'][3])
    new_state['bombs'] = list(map(mirrorBomb, game_state['bombs']))
    new_state['coins'] = list(map(mirrorPosition, game_state['coins']))
    new_state['others'] = list(map(mirrorOther, game_state['others']))

    return new_state

def transposeGame(game_state: dict):
    new_state = game_state.copy()

    new_state['field'] = np.rot90(game_state['field'])
    new_state['explosion_map'] = np.rot90(game_state['explosion_map'])

    def transposePosition(position):
        return position[1], position[0]
    def transposeBomb(bomb):
        return transposePosition(bomb[0]), bomb[1]
    def transposeOther(other):
        return other[0], other[1], other[2], transposePosition(other[3])
    
    new_state['self'][3] = transposePosition(game_state['self'][3])
    new_state['bombs'] = list(map(transposeBomb, game_state['bombs']))
    new_state['coins'] = list(map(transposePosition, game_state['coins']))
    new_state['others'] = list(map(transposeOther, game_state['others']))

    return new_state

def state_to_features(self, game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # relevant data in game state:
    agentPosition = (game_state['self'])[3]
    bombPossible = (game_state['self'])[2]
    neighbourFields = [(agentPosition[0] + 1, agentPosition[1]),
        (agentPosition[0] - 1, agentPosition[1]),
        (agentPosition[0], agentPosition[1] + 1),
        (agentPosition[0], agentPosition[1] - 1)
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
                creates.append((i,j))

    others = game_state['others']
    othersPositions = list(map(lambda x: x[3], others))


    # Find the direction you need to move to get a coin:

    # Find nearest Coin with BFS:
    def bfs(objects):
        if len(objects) == 0:
            return None, 1000

        if agentPosition in objects:
            return agentPosition, 0

        explored = [agentPosition]
        queue = [(agentPosition,0)]


        while queue:
            (position, distance) = queue.pop(0)
            neighbours = [(position[0] + 1, position[1]),(position[0] - 1, position[1]),(position[0], position[1] + 1),(position[0], position[1] - 1)]
            for neighbour in neighbours:
                if neighbour not in explored:
                    for object in objects:
                        if neighbour[0] == object[0] and neighbour[1] == object[1]:
                            return neighbour, distance + 1
                    if field[neighbour[0]][neighbour[1]] == 0 and neighbour not in othersPositions:
                        explored.append(neighbour)
                        queue.append((neighbour, distance + 1))
                    else:
                        explored.append(neighbour)
        return None, 1000
    nearest_coin, distance_coin = bfs(coins)
    nearest_create, distance_create = bfs(creates)
    #self.logger.debug(f'Create Found from agen pos {agentPosition} at {nearest_create} with distance {distance_create}')
    nearest_bomb, distance_bomb = bfs(bombPositions)
    nearest_other, distance_other = bfs(othersPositions)

    #self.logger.debug(f'Other Found from agen pos {agentPosition} at {nearest_other} with distance {distance_other}')

    # add one feature horizontally and vertically

    def coinFeatures():
        """
        values -1, 0, 1 for each feature
        """
        if nearest_coin == None:
            return [0,0,0,0]

        # add feature for walls around coin
        wallsAroundCoinFeature = 0
        if field[nearest_coin[0] + 1][nearest_coin[1]] != 0 and field[nearest_coin[0] - 1][nearest_coin[1]] != 0:
            wallsAroundCoinFeature = 1
        elif field[nearest_coin[0]][nearest_coin[1] + 1] != 0 and field[nearest_coin[0]][nearest_coin[1] - 1] != 0:
            wallsAroundCoinFeature = -1

        return [
            wallsAroundCoinFeature,
            # add a feature for horizontal in vertical direction:
            np.sign(nearest_coin[0] - agentPosition[0]),
            np.sign(nearest_coin[1] - agentPosition[1]),
            # add a feature for differences in horizontal in vertical distance:
            np.sign(np.abs(nearest_coin[0] - agentPosition[0]) - np.abs(nearest_coin[1] - agentPosition[1]))
            ]

    # Look for walls in the way:
    def fieldFeatures():
        """
        values: 
        0 air
        1 create or other
        2 wall or explosion 
        
        for each direction
        """
        fieldResults = []

        for i in range(4):
            if field[neighbourFields[i][0]][neighbourFields[i][1]] == -1:
                fieldResults.append(2)
            elif explosions[neighbourFields[i][0]][neighbourFields[i][1]] != 0:
                fieldResults.append(2)
            elif field[neighbourFields[i][0]][neighbourFields[i][1]] == 0 and neighbourFields[i] in othersPositions:
                fieldResults.append(1)
            else:
                fieldResults.append(field[neighbourFields[i][0]][neighbourFields[i][1]])

        return fieldResults

    def createFeatures():
        """
        values -1, 0, 1 for each feature
        """
        if nearest_create == None:
            return [0,0,0,0]

        # add feature for walls around create
        wallsAroundCreateFeature = 0
        if field[nearest_create[0] + 1][nearest_create[1]] != 0 and field[nearest_create[0] - 1][nearest_create[1]] != 0:
            wallsAroundCreateFeature = 1
        elif field[nearest_create[0]][nearest_create[1] + 1] != 0 and field[nearest_create[0]][nearest_create[1] - 1] != 0:
            wallsAroundCreateFeature = -1

        return [
            wallsAroundCreateFeature,
            # add a feature for horizontal in vertical direction:
            np.sign(nearest_create[0] - agentPosition[0]),
            np.sign(nearest_create[1] - agentPosition[1]),
            # add a feature for differences in horizontal in vertical distance:
            np.sign(np.abs(nearest_create[0] - agentPosition[0]) - np.abs(nearest_create[1] - agentPosition[1]))
            ]

    def advancedSafetyFeatures():
        """
        for the actions left, right, down, up and wait indicate how it increases the safety
        """
        safetyResults = [0,0,0,0,0]
        if len(bombs) == 0:
            return [0,0,0,0,0]

        bombFields = []

        for bomb in bombs:
            (bomb_x, bomb_y), _ = bomb
            bombFields.append((bomb_x, bomb_y))
            i = 1
            while i < 4 and field[bomb_x + i][bomb_y] != -1:
                bombFields.append((bomb_x + i, bomb_y))
                i += 1
            i = 1
            while i < 4 and field[bomb_x - i][bomb_y] != -1:
                bombFields.append((bomb_x - i, bomb_y))
                i += 1
            i = 1
            while i < 4 and field[bomb_x][bomb_y + i] != -1:
                bombFields.append((bomb_x, bomb_y + i))
                i += 1
            i = 1
            while i < 4 and field[bomb_x][bomb_y - i] != -1:
                bombFields.append((bomb_x, bomb_y - i))
                i += 1
        #self.logger.debug("Current position: " + str(agentPosition))
        #self.logger.debug("Computed bomb fields: " + str(bombFields))

        if agentPosition not in bombFields:
            for i in range(4):
                neighbour = neighbourFields[i]
                if neighbour in bombFields:
                    safetyResults[i] = 1
        else:
            safetyResults[4] = 1

            def bfs_for_save_field(start_pos):

                if start_pos not in bombFields:
                    return 0
                explored = [start_pos]
                queue = [(start_pos,0)]

                while queue:
                    (position, distance) = queue.pop(0)
                    neighbours = [(position[0] + 1, position[1]),(position[0] - 1, position[1]),(position[0], position[1] + 1),(position[0], position[1] - 1)]
                    for neighbour in neighbours:
                        if neighbour not in explored:
                            if field[neighbour[0]][neighbour[1]] == 0 and neighbour not in othersPositions:
                                if neighbour not in bombFields:
                                    return distance + 1
                                explored.append(neighbour)
                                queue.append((neighbour, distance + 1))
                            else:
                                explored.append(neighbour)
                return 1000
            minDistance = 10000
            indeces = []
            for i in range(4):
                if field[neighbourFields[i][0]][neighbourFields[i][1]] == 0 and neighbourFields[i] not in othersPositions:
                    neighbourDist = bfs_for_save_field(neighbourFields[i])
                    #self.logger.debug("Computed dist " + str(neighbourDist) + " for direction " + str(i) + " and field " + str(neighbourFields[i]))
                    if neighbourDist < minDistance:
                        indeces = [i]
                        minDistance = neighbourDist
                    elif neighbourDist == minDistance:
                        indeces.append(i)
            for i in range(4):
                if i not in indeces:
                    safetyResults[i] = 1
        
        #self.logger.debug("Computed safety features: " + str(safetyResults))

        return safetyResults
            
    def nearestBombFeatures():
        """
        Check wether the nearest bomb is on the same line
        """
        if nearest_bomb == None:
            return [3,3]

        return [
            # add a feature for horizontal in vertical direction:
            np.sign(nearest_bomb[0] - agentPosition[0]) + 1,
            np.sign(nearest_bomb[1] - agentPosition[1]) + 1
            ]

    def bombPlacingFeatures():
        '''
        We want to use bombs if we can blow up creates or others
        '''
        if not bombPossible:
            return [0]
        if nearest_create in neighbourFields:
            return [1]
        for other in othersPositions:
            if other in neighbourFields:
                return [1]
        return [2]

    def othersFeatures():
        """
        values -1, 0, 1 for each feature
        """
        if nearest_other == None:
            return [0,0,0,0]

        # add feature for walls around create
        wallsAroundOtherFeature = 0
        if field[nearest_other[0] + 1][nearest_other[1]] != 0 and field[nearest_other[0] - 1][nearest_other[1]] != 0:
            wallsAroundOtherFeature = 1
        elif field[nearest_other[0]][nearest_other[1] + 1] != 0 and field[nearest_other[0]][nearest_other[1] - 1] != 0:
            wallsAroundOtherFeature = -1

        return [
            wallsAroundOtherFeature,
            # add a feature for horizontal in vertical direction:
            np.sign(nearest_other[0] - agentPosition[0]),
            np.sign(nearest_other[1] - agentPosition[1]),
            # add a feature for differences in horizontal in vertical distance:
            np.sign(np.abs(nearest_other[0] - agentPosition[0]) - np.abs(nearest_other[1] - agentPosition[1]))
            ]
    
    channels = advancedSafetyFeatures()
    channels = channels + fieldFeatures()
    channels = channels + bombPlacingFeatures()
    channels = channels + coinFeatures()
    channels = channels + createFeatures()
    channels = channels + othersFeatures()

    return np.array(channels), [distance_coin, distance_create, distance_bomb, distance_other]