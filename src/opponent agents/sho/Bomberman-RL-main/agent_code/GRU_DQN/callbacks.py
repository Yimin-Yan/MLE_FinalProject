import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os

processing_unit = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

epsilon = 0.1

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
    self.DEEP_Q_Agent = DEEPNetwork().to(processing_unit)
    self.DEEP_Q_Agent.double()
#load model parameters
    if os.path.isfile("model_parameters.pt"):
        self.DEEP_Q_Agent.load_state_dict(torch.load("model_parameters.pt",map_location=processing_unit))


def act(self,game_state: dict):
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if game_state == None:
        pass
    else:
        if len(game_state["bombs"])>0:
            self.strategy = 'shortest'
        else:
            self.strategy = 'softmax'

        #explore
        if self.train and random.random() <= epsilon:
            self.logger.debug(f'Choosing action purely at random, epsilon value: {np.round(epsilon, 2)}')
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
#input features for model
        features = state_to_features(game_state)
        features_torch = torch.from_numpy(np.reshape(features,(1,22)))
        self.DEEP_Q_Agent.eval()
        self.model = self.DEEP_Q_Agent(features_torch.to(processing_unit)).to(processing_unit)
        self.DEEP_Q_Agent.train()

        #exploit
        if self.strategy=='shortest':
            return ACTIONS[torch.argmax(self.model)]
        else:
            return np.random.choice(ACTIONS,p=torch.softmax(self.model/2,-1).cpu().detach().numpy().flatten())

def explosion_state(explosion_location,x,y,bombs,field):
    """
    state for danger zone from explosion
    """
    for bomb in bombs:
        x_state_bomb,y_state_bomb = bomb[0]
        for i in range(1, 5):
            #explosion x up
            if field[x_state_bomb + i, y_state_bomb] == -1:
                break
            explosion_location[x_state_bomb + i, y_state_bomb] = 10-1*bomb[1]-1*i
        for i in range(1, 5):
            #explosion x down
            if field[x_state_bomb - i, y_state_bomb] == -1:
                break
            explosion_location[x_state_bomb - i, y_state_bomb] = 10-1*bomb[1]-1*i
        for i in range(1, 5):
            #explosion y up
            if field[x_state_bomb, y_state_bomb + i] == -1:
                break
            explosion_location[x_state_bomb, y_state_bomb + i] = 10-1*bomb[1]-1*i
        for i in range(1, 5):
            #explosion y down
            if field[x_state_bomb, y_state_bomb - i] == -1:
                break
            explosion_location[x_state_bomb, y_state_bomb - i] = 10-1*bomb[1]-1*i
        explosion_location[x_state_bomb,y_state_bomb] = 10-1*bomb[1]
    danger_zone = explosion_location[x-1:x+2,y-1:y+2].flatten()
    return(danger_zone)

def coin_state(coin_location,x,y):
    """
    state for coins
    """
    if len(coin_location)>0:
        coins = np.array(coin_location)
        coin_distance = np.sum(np.abs(coins - np.array([x,y])[None,:]),axis=-1)
        coin = coins[np.argmin(coin_distance)].flatten()-np.array([x,y])
    else:
        coin = np.array([0,0])
    return(coin)

def arena_state(arena_location,x,y):
    """
    state for arena
    """
    if len(arena_location[0])>0:
        arenas = np.array(list(zip(arena_location[0],arena_location[1])))
        arena_distance = np.sum(np.abs(arenas - np.array([x, y])[None, :]), axis=-1)
        arena = arenas[np.argmin(arena_distance)].flatten() - np.array([x, y])
    else:
        arena = np.array([0,0])
    return(arena)

def state_to_features(game_state: dict):
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
    x, y = game_state["self"][-1]

    field = game_state["field"]
    if game_state["self"][2]:
        field[x,y] = 1
    for others in game_state["others"]:
        field[others[-1]] = 10
    vision = field[x-1:x+2,y-1:y+2].flatten()

    explosion_location  = np.clip(np.array(game_state["explosion_map"])*100,0,10)
    bombs = game_state["bombs"]
    coin_location = game_state["coins"]
    arena_location = np.where(field == 1)

    danger_zone = explosion_state(explosion_location,x,y,bombs,field)
    coin = coin_state(coin_location,x,y)
    arena = arena_state(arena_location,x,y)
#features like enemy location, explosion danger, coin location, and arena tiles
    features = np.hstack((vision,danger_zone,coin,arena))

    return(features)

class DEEPNetwork(nn.Module):
    """
    Deep Learning model using GRU layers
    """
    def __init__(self):
        super(DEEPNetwork, self).__init__()
        self.lstm1 = nn.GRU(22,12,2,batch_first = True, dropout = 0.2)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.linear0 = nn.Linear(6,6)
        self.relu0 = nn.ReLU()
        self.linear2 = nn.Linear(6, len(ACTIONS))

    def forward(self, x):
        out, states = self.lstm1(x)
        out = self.pool1(out)
        out = self.linear0(out)
        out = self.relu0(out)
        out = self.linear2(out)
        return(out)
