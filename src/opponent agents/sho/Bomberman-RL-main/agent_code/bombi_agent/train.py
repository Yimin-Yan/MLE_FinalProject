import random
from typing import List
import numpy as np
import events as e
from agent_code.bombi_agent.constants import *
from agent_code.bombi_agent.feat_ex import FeatureExtraction


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    np.random.seed()
    self.accumulated_reward = 0
    self.current_round = 1

    # Learning methods hyper-parameters
    self.discount = 0.95
    self.alpha = 0.01

    # Parameters for diminished epsilon policy.
    self.accumulated_reward_generation = 0
    self.generation_current = 1
    self.generation_nrounds = 10
    self.generation_total = max(1, int(10 / self.generation_nrounds))
    self.game_ratio = self.generation_current / self.generation_total

    # Hyper-parameters
    self.replay_buffer = []
    self.replay_buffer_max_steps = 200
    self.replay_buffer_update_after_nrounds = 10
    self.replay_buffer_sample_size = 50
    self.replay_buffer_every_ngenerations = 1


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, events will contain a list of all game
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
    # At least one action must be completed before an interim reward can be computed.
    if self.game_state['step'] == 1:
        return None

    reward = new_reward(events)
    self.logger.info('Given reward of %s', reward)
    self.accumulated_reward += reward

    # Getting features for the new game state
    self.F = FeatureExtraction(new_game_state)

    # Previous step action.
    self.prev_action = self_action

    # Keep a record of all experiences of this for later learning.
    if self.game_state['step'] <= self.replay_buffer_max_steps + 1:
        self.replay_buffer.append((self.F_prev, self.prev_action, reward, self.F, False))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # adding experience
    self.F = FeatureExtraction(last_game_state)
    self.prev_action = last_action
    reward = new_reward(events)

    self.replay_buffer.append((self.F_prev, self.prev_action, reward, self.F, True))
    self.logger.info('Given end reward of %s', reward)
    self.accumulated_reward += reward
    self.accumulated_reward_generation += self.accumulated_reward
    self.accumulated_reward = 0
    self.current_round += 1

    experience_replay(self)

    np.save(weights_file, self.weights)


def experience_replay(self):
    if (self.current_round % self.generation_nrounds) == 0:
        experience_mini_batch = random.sample(self.replay_buffer, self.replay_buffer_sample_size)
        print("REPLAY BUFFER SIZE:", len(self.replay_buffer))
        print("MINI BATCH SIZE:", len(experience_mini_batch))
        self.replay_buffer_sample_size += 20

        # Resetting replay buffer
        if self.generation_current % self.replay_buffer_every_ngenerations == 0:
            print("RESETTING REPLAY BUFFER")
            self.replay_buffer = []

        weights_batch_update = batch_gradient_descent(experience_mini_batch, self)

        self.weights = self.weights + weights_batch_update
        self.plot_weights.append(self.weights)

        self.generation_current += 1
        self.game_ratio = self.generation_current / self.generation_total

        self.plot_rewards.append(self.accumulated_reward_generation)
        np.save(t_training_id, self.plot_rewards)
        self.accumulated_reward_generation = 0


def batch_gradient_descent(experience_mini_batch, self):
    weights_batch_update = np.zeros(len(self.weights))
    for X, A, R, Y, terminal in experience_mini_batch:
        if A is not None:
            x_a = X.state_action(A)
            q_max, a_max = Y.maximum_qval(self.weights)
            td_error = R + (self.discount * q_max) - np.dot(x_a, self.weights)

            weights_batch_update = weights_batch_update + self.alpha / self.replay_buffer_sample_size * td_error * x_a
    return weights_batch_update


def new_reward(events):
    reward = 0
    reward_map = {
        e.BOMB_DROPPED: 0.1,
        e.COIN_COLLECTED: 3,
        e.KILLED_SELF: -10,
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 1,
        e.KILLED_OPPONENT: 7,
        e.GOT_KILLED: -5,
        e.SURVIVED_ROUND: 2,
        e.INVALID_ACTION: -0.5

    }
    for event in events:
        reward += reward_map.get(event)
    return reward
