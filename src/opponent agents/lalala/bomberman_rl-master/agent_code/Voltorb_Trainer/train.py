import pickle
from typing import List

import events as e
import settings
from .callbacks import state_to_features, ACTIONS

import numpy as np

# Events
MOVED_BACK_AGAIN_EVENT = "MOVED_BACK_AGAIN"

COIN_DISTANCE_SMALLER = "COIN_DISTANCE_SMALLER"
COIN_DISTANCE_BIGGER = "COIN_DISTANCE_BIGGER"
COIN_DISTANCE_SAME = "COIN_DISTANCE_SAME"

CREATE_DISTANCE_SMALLER = "CREATE_DISTANCE_SMALLER"
CREATE_DISTANCE_BIGGER = "CREATE_DISTANCE_BIGGER"
CREATE_DISTANCE_SAME = "CREATE_DISTANCE_SAME"

OTHER_DISTANCE_SMALLER = "OTHER_DISTANCE_SMALLER"
OTHER_DISTANCE_BIGGER = "OTHER_DISTANCE_BIGGER"
OTHER_DISTANCE_SAME = "OTHER_DISTANCE_SAME"

PLACED_BOMB_NEXT_TO_OTHER = "PLACED_BOMB_NEXT_TO_OTHER"
PLACED_BOMB_NEXT_TO_CREATE = "PLACED_BOMB_NEXT_TO_CREATE"
PLACED_BAD_BOMB = "PLACED_BAD_BOMB"

WAITED_AND_NO_BOMB = "WAITED_AND_NO_BOMB"

NO_ESCAPE_MOVE = "NO_ESCAPE_MOVE"
ESCAPE_MOVE = "ESCAPE_MOVE"

BOMB_BUT_NOT_POSSIBLE = "BOMB_BUT_NOT_POSSIBLE"

WALK_INTO_EXPLOSION = "WALK_INTO_EXPLOSION"

EASY_INVALID_ACTION = "EASY_INVALID_ACTION"

SURVIVED_NEXT_TO_EXPLOSION = "SURVIVED_NEXT_TO_EXPLOSION"


# Trainig Hyperparams
LEARNING_RATE = 0.1
GAMMA = 0.2

def setup_training(self):
    self.logger.debug("Start training")
    self.lastMove = 'WAIT'
    self.currentCoinDistance = 1000
    self.currentCreateDistance = 1000

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   gather gamestate:
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

    old_features, old_distances = state_to_features(self, old_game_state)
    new_features, distances = state_to_features(self, new_game_state)

    field = old_game_state['field']

    distance_to_nearest_coin = distances[0]
    distance_to_nearest_create = distances[1]
    
    old_nearest_other = old_distances[2]

    def bfs(objects, start):
        if len(objects) == 0:
            return None, 0

        explored = [start]
        queue = [(start,0)]

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

    others = old_game_state['others']
    othersPositions = list(map(lambda x: x[3], others))
    agentPosition = old_game_state['self'][3]
    neighbourFields = [(agentPosition[0] + 1, agentPosition[1]),
        (agentPosition[0] - 1, agentPosition[1]),
        (agentPosition[0], agentPosition[1] + 1),
        (agentPosition[0], agentPosition[1] - 1)]
    explos = old_game_state['explosion_map']
    
    _, new_nearest_other = bfs(othersPositions, new_game_state['self'][3])

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   add custom events:
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

    if (self_action, self.lastMove) in [('UP', 'DOWN'), ('DOWN', 'UP'), ('LEFT', 'RIGHT'), ('RIGHT', 'LEFT')]:
            events.append(MOVED_BACK_AGAIN_EVENT)

    if self_action == 'BOMB' and old_game_state['self'][2] == False:
        events.append(BOMB_BUT_NOT_POSSIBLE)

    nextToExplosion = False
    for neighbour in neighbourFields:
        if explos[neighbour[0]][neighbour[1]] != 0:
            nextToExplosion = True
    if nextToExplosion:
        events.append(SURVIVED_NEXT_TO_EXPLOSION)
    
    for event in events:
        if event == e.INVALID_ACTION:
            #self.logger.debug(f'Invalid Action with agent at {agentPosition} and others positions {othersPositions}')
            if (self_action == 'UP' and field[agentPosition[0]][agentPosition[1] - 1] != 0) \
                or (self_action == 'DOWN' and field[agentPosition[0]][agentPosition[1] + 1] != 0) \
                or (self_action == 'RIGHT' and field[agentPosition[0] + 1][agentPosition[1]] != 0) \
                or (self_action == 'LEFT' and field[agentPosition[0] - 1][agentPosition[1]] != 0):
                    events.append(EASY_INVALID_ACTION)
        if event == e.BOMB_DROPPED:
            others = old_game_state['others']
            othersPos =  list(map(lambda x: x[3], others))
            
            placeBombNextToSomething = False
            for other in othersPos:
                if other in neighbourFields:
                    events.append(PLACED_BOMB_NEXT_TO_OTHER)
                    placeBombNextToSomething = True
            for neighbour in neighbourFields:
                if field[neighbour[0]][neighbour[1]] == 1:
                    events.append(PLACED_BOMB_NEXT_TO_CREATE)
                    placeBombNextToSomething = True
            if not placeBombNextToSomething:
                events.append(PLACED_BAD_BOMB)
        #explos = old_game_state['explosion_map']
        #self.logger.debug(f'Exlosion map was: {explos} and decision was {(explos == 0).all()}')
        if event == e.WAITED and len(old_game_state['bombs']) == 0 and (old_game_state['explosion_map'] == 0).all():
            events.append(WAITED_AND_NO_BOMB)

    if distance_to_nearest_coin != 1000:
        if distance_to_nearest_coin < self.currentCoinDistance:
            events.append(COIN_DISTANCE_SMALLER)
        elif distance_to_nearest_coin > self.currentCoinDistance:
            events.append(COIN_DISTANCE_BIGGER)
        else:
            events.append(COIN_DISTANCE_SAME)

    if old_nearest_other != 1000:
        if new_nearest_other < old_nearest_other:
            events.append(OTHER_DISTANCE_SMALLER)
        elif new_nearest_other > old_nearest_other:
            events.append(OTHER_DISTANCE_BIGGER)
        else:
            events.append(OTHER_DISTANCE_SAME)

    creates = []
    for i in range(settings.COLS):
        for j in range(settings.ROWS):
            if field[i][j] == 1:
                creates.append((i,j))
    
    if len(creates) != 0:
        if distance_to_nearest_create < self.currentCreateDistance:
            events.append(CREATE_DISTANCE_SMALLER)
        elif distance_to_nearest_create > self.currentCreateDistance:
            events.append(CREATE_DISTANCE_BIGGER)
        else:
            events.append(CREATE_DISTANCE_SAME)
    
    if len(old_game_state['bombs']) > 0:
        escape_move = True
        if 1 in old_features[0:5]:
            for i in range(5):
                if old_features[i] == 1 and (self_action,i) in [('RIGHT',0), ('LEFT',1), ('DOWN',2), ('UP',3), ('WAIT',4)]:
                    events.append(NO_ESCAPE_MOVE)
                    escape_move = False
            if escape_move:
                events.append(ESCAPE_MOVE)

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   update self:
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

    self.currentCoinDistance = distance_to_nearest_coin
    self.currentCreateDistance = distance_to_nearest_create
    self.lastMove = self_action

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   training step:
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

    reward = reward_from_events(self, events)
    action = int(np.where(np.array(ACTIONS) == self_action)[0][0])
    
    if self.seenFeatures.size != 0:
        possible_old_features_indices = np.where((self.seenFeatures == old_features).all(axis=1))
        if len(possible_old_features_indices[0]) != 0:
            index_old = possible_old_features_indices[0][0]
        else:
            index_old = addNewFeatures(self, old_features)
    else:
        index_old = addNewFeatures(self, old_features)

    expectedOldReward = self.featureRewards[index_old][action]

    possible_new_features_indices = np.where((self.seenFeatures == new_features).all(axis=1))
    if len(possible_new_features_indices[0]) == 0:
        index_new = addNewFeatures(self, new_features)
    else:
        index_new = possible_new_features_indices[0][0]
    
    expectedRewardAfterStep = np.amax(self.featureRewards[index_new])

    # "Q-Function" update
    self.featureRewards[index_old][action] = expectedOldReward + LEARNING_RATE * (reward + GAMMA * expectedRewardAfterStep - expectedOldReward)

    self.logger.debug(f'Updated guess for featues {old_features} form {expectedOldReward} to {self.featureRewards[index_old][action]}')


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   add custom events:
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

    oldPosX, oldPosY = last_game_state['self'][3]
    if last_action == 'UP':
        if last_game_state['explosion_map'][oldPosX][oldPosY - 1] != 0:
            events.append(WALK_INTO_EXPLOSION)
    if last_action == 'DOWN':
        if last_game_state['explosion_map'][oldPosX][oldPosY + 1] != 0:
            events.append(WALK_INTO_EXPLOSION)
    if last_action == 'RIGHT':
        if last_game_state['explosion_map'][oldPosX + 1][oldPosY] != 0:
            events.append(WALK_INTO_EXPLOSION)
    if last_action == 'LEFT':
        if last_game_state['explosion_map'][oldPosX - 1][oldPosY] != 0:
            events.append(WALK_INTO_EXPLOSION)


    old_features, _ = state_to_features(self, last_game_state)

    if len(last_game_state['bombs']) > 0:
        escape_move = True
        if 1 in old_features:
            for i in range(5):
                if old_features[i] == 1 and (last_action,i) in [('RIGHT',0), ('LEFT',1), ('DOWN',2), ('UP',3), ('WAIT',4)]:
                    events.append(NO_ESCAPE_MOVE)
                    escape_move = False
            if escape_move:
                events.append(ESCAPE_MOVE)
    
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   training step:
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

    reward = reward_from_events(self, events)
    action = int(np.where(np.array(ACTIONS) == last_action)[0][0])
    if self.seenFeatures.size != 0:
        possible_old_features_indices = np.where((self.seenFeatures == old_features).all(axis=1))
        if len(possible_old_features_indices[0]) != 0:
            index_old = possible_old_features_indices[0][0]
        else:
            index_old = addNewFeatures(self, old_features)
    else:
        index_old = addNewFeatures(self, old_features)

    expectedOldReward = self.featureRewards[index_old][action]

    # "Q-Function" update
    self.featureRewards[index_old][action] = expectedOldReward + LEARNING_RATE * (reward - expectedOldReward)

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   Store the model:
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    
    with open("my-seen-features.pt", "wb") as file1:
        pickle.dump(self.seenFeatures, file1)
    with open("my-feature-rewards.pt", "wb") as file2:
        pickle.dump(self.featureRewards, file2)

    self.logger.debug(f'Encountered {self.seenFeatures.shape[0]} features')



def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        EASY_INVALID_ACTION: -100000,
        MOVED_BACK_AGAIN_EVENT: -10,
        COIN_DISTANCE_BIGGER: -1400,
        COIN_DISTANCE_SMALLER: 1400,
        COIN_DISTANCE_SAME: -1300,
        e.COIN_COLLECTED: 3000,
        CREATE_DISTANCE_BIGGER: -200,
        CREATE_DISTANCE_SMALLER: 200,
        CREATE_DISTANCE_SAME: -20,
        PLACED_BOMB_NEXT_TO_OTHER: 1000,
        PLACED_BOMB_NEXT_TO_CREATE: 1000,
        PLACED_BAD_BOMB: -8000,
        WAITED_AND_NO_BOMB: -10000,
        NO_ESCAPE_MOVE: -10000,
        ESCAPE_MOVE: 10000,
        BOMB_BUT_NOT_POSSIBLE: -100000,
        OTHER_DISTANCE_SMALLER: 80,
        OTHER_DISTANCE_SAME: -1,
        OTHER_DISTANCE_BIGGER: -80,
        WALK_INTO_EXPLOSION: -11000,
        SURVIVED_NEXT_TO_EXPLOSION: 5000,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def addNewFeatures(self, feature):
    self.seenFeatures = np.append(self.seenFeatures, np.expand_dims(feature, axis=0), axis = 0)
    self.featureRewards = np.append(self.featureRewards, np.expand_dims(np.zeros(6), axis=0), axis=0)
    return self.featureRewards.shape[0] - 1