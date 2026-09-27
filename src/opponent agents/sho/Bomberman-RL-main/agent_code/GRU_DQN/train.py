import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import events as e
from collections import namedtuple, deque
from .callbacks import state_to_features, ACTIONS, DEEPNetwork
from . import callbacks
from typing import List
import numpy as np
import os

#cuda or cpu
processing_unit = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path=os.path.dirname(os.path.abspath(__file__))
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
DECREASED_DISTANCE = "DIST_DECREASED"
INCREASED_DISTANCE = "DIST_INCREASED"
BOMB_DROPPED_CORNER = "BOMB_CORNER"
BOMB_NEAR_CRATE = 'BOMB_NEAR_CRATE'
BOMB_NOT_NEAR_CRATE = 'BOMB_NOT_NEAR_CRATE'
MOVED_AWAY_FROM_BOMB = 'MOVED_AWAY_FROM_BOMB'
MOVED_TO_BOMB = 'MOVED_TO_BOMB'
DEAD_END = 'DEAD_END'
NOT_DEAD_END = 'NOT_DEAD_END'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #initialize values optimizer and loss functions
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.optimizer = optim.Adam(self.DEEP_Q_Agent.parameters(), lr=0.00005)
    self.sl1_loss = nn.SmoothL1Loss().to(processing_unit)
    self.store_list = []
    self.reward_total  = 0.0
    self.reward_total1 = []
    self.loss_total   = 0.0
    self.loss_total1 = []
    self.total_epsilon = []
    self.gamma = 0.99
    self.decay = 0.9999
    self.min_epsilon = 0.001
    self.total_epsilon = []
    self.update_iteration_counter = 0
    self.target_DEEP_Q_Agent = DEEPNetwork().to(processing_unit)
    self.target_DEEP_Q_Agent.double()
    self.target_DEEP_Q_Agent.load_state_dict(self.DEEP_Q_Agent.state_dict())
    self.target_DEEP_Q_Agent.eval()

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
    if old_game_state is None and self_action == 'BOMB':
        events.append(BOMB_DROPPED_CORNER)

    if old_game_state == None:
        pass
    else:
        if new_game_state is not None:
            # custom events begin
            events = bomb_check(new_game_state, self_action, events)
            if np.array(new_game_state['coins']).T.size > 0:
                events = coin_dist_check(old_game_state, new_game_state, events)
            if np.array(new_game_state['bombs']).T.size > 0:
                events = bomb_dist_check(old_game_state, new_game_state, events)
                events = dead_end_check2(old_game_state, new_game_state, events)
        # custom events end
        self.update_iteration_counter += 1
        self.optimizer.zero_grad()
        reward = reward_from_events(self,old_game_state,new_game_state,events)
        index = ACTIONS.index(self_action)
        self.reward_total += reward
        #storing experience
        if len(self.store_list)>500:
            self.store_list.pop(0)
        if new_game_state != None:
            self.store_list.append([torch.from_numpy(state_to_features(old_game_state)).to(processing_unit),
                                torch.from_numpy(state_to_features(new_game_state)).to(processing_unit),
                                index,torch.tensor([reward]).to(processing_unit).double(),True,10.0])
        elif new_game_state == None:
            self.store_list.append([torch.from_numpy(state_to_features(old_game_state)).to(processing_unit),
                                torch.from_numpy(state_to_features(old_game_state)).to(processing_unit),
                                index,torch.tensor([reward]).to(processing_unit).double(),False,10.0])
        if len(self.store_list)>1:
            top = np.array([i[5] for i in self.store_list])
            top_avg = top/np.sum(top)
            rand = np.random.choice(len(self.store_list),size=np.clip(len(self.store_list),None,5),replace=False,p=top_avg)
            mask_terminal_states = np.ones(len(rand))

            for i,j in enumerate(rand):
                if self.store_list[j][4] == False: mask_terminal_states[i] = 0
#mask states
            mask = torch.from_numpy(mask_terminal_states).double().to(processing_unit)
            previous_batch = torch.stack([self.store_list[i][0] for i in rand],0)
            current_batch = torch.stack([self.store_list[i][1] for i in rand],0)
            batch_reward = torch.stack([self.store_list[i][3] for i in rand])
            batch_action = torch.tensor([self.store_list[i][2] for i in rand]).to(processing_unit)
#calculate loss
            pred_value = self.DEEP_Q_Agent(previous_batch).gather(1,batch_action.view(-1,1))
            true_value = batch_reward + (self.gamma * mask * self.target_DEEP_Q_Agent(current_batch).detach().max(1)[0]).unsqueeze(1)
            loss = self.sl1_loss(pred_value,true_value)
            self.loss_total += loss.item()

            for i in rand: self.store_list[i][5]=loss.item()

            loss.backward()
            self.optimizer.step()

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
    game_events_occurred(self, last_game_state, last_action, None, events)
#save model parameters
    torch.save(self.DEEP_Q_Agent.state_dict(),path+"\model_parameters.pt")

    if self.update_iteration_counter>100:
        self.target_DEEP_Q_Agent.load_state_dict(self.DEEP_Q_Agent.state_dict())
        self.target_DEEP_Q_Agent.eval()
        self.update_iteration_counter = 0
#store reward and loss
    self.reward_total1.append(self.reward_total/last_game_state["step"])
    np.save(path+"/model_reward.npy",self.reward_total1)

    self.loss_total1.append(self.loss_total/last_game_state["step"])
    np.save(path+"/model_loss.npy",self.loss_total1)

    self.reward_total=0.0
    self.loss_total=0.0
#update epsilon
    callbacks.epsilon = max(self.min_epsilon, callbacks.epsilon*self.decay)
#store epsilon value
    self.total_epsilon.append(callbacks.epsilon/last_game_state["step"])
    np.save(path+"/model_epsilon.npy",self.total_epsilon)

def reward_from_events(self,old,new,events):
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
        e.INVALID_ACTION: -0.8,
        e.BOMB_EXPLODED: 0.14,
        e.BOMB_DROPPED: -0.1,
        # e.CRATE_DESTROYED: 0.02,
        # e.COIN_FOUND: 0.02,
        e.COIN_COLLECTED: 0.5,
        e.KILLED_OPPONENT: 1,
        e.KILLED_SELF: -1,
        e.GOT_KILLED: -1,
        e.OPPONENT_ELIMINATED: 2,
        e.SURVIVED_ROUND: 0.2,
        # custom events
        e.DECREASED_DISTANCE: 0.3,
        e.INCREASED_DISTANCE: -0.2,
        e.BOMB_DROPPED_CORNER: -0.8,
        e.BOMB_NEAR_CRATE: 0.6,
        e.BOMB_NOT_NEAR_CRATE: -0.5,
        e.MOVED_AWAY_FROM_BOMB: 0.5,
        e.MOVED_TO_BOMB: -0.6,
        e.DEAD_END: -0.7,
        e.NOT_DEAD_END: 0.6
    }
    reward = 0
    for event in events:
        if event in game_rewards:
            reward += game_rewards[event]
    self.logger.info(f"Awarded {reward} for events {', '.join(events)}")
    return reward

def coin_dist_check(old_game_state, new_game_state, events):
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
