import os
import random

from agent_code.bombi_agent.constants import *
from agent_code.bombi_agent.feat_ex import *


def weight_initialize(trained=True):
    best_guess = np.asarray([1, 1.7, -7, 3.5, 1.5, 1, 0.5, -0.8, 0.5])
    trained_values = [1., 2.13864661, -7.53564286, 3.73366143, 2.61421682, 1.00162857, 0.60588571, -1.17647312, 0.50014563]

    return trained_values if trained else best_guess


def select_policy(greedy_action, policy, policy_epsilon, game_ratio, logger=None):
    # exploration vs exploitation
    random_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'])

    if policy == 'greedy':
        logger.info('Pick greedy action %s', greedy_action)
        return greedy_action
    elif policy == 'epsilon_greedy':
        logger.info('Picking greedy action at probability %s', 1 - policy_epsilon)
        return np.random.choice([greedy_action, random_action],
                                p=[1 - policy_epsilon, policy_epsilon])
    elif policy == 'diminishing':
        policy_epsilon_dimimishing = max(0.05, (1 - game_ratio) * policy_epsilon)
        logger.info('Picking greedy action at probability %s', 1 - policy_epsilon_dimimishing)
        return np.random.choice([greedy_action, random_action],
                                p=[1 - policy_epsilon_dimimishing, policy_epsilon_dimimishing])
    else:
        return None


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
    if self.train or not os.path.isfile(weights_file):
        self.logger.info("Setting up model from scratch.")
        self.weights = weight_initialize()
        print("INITIALIZED WEIGHTS", self.weights)
    else:
        self.logger.info("Loading model from saved state.")
        self.weights = np.load(weights_file)
        print("LOADED WEIGHTS", self.weights)


    self.plot_rewards = []
    self.plot_weights = [self.weights]


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.game_state = game_state
    self.F_prev = FeatureExtraction(self.game_state)

    # Shuffling the greedy actions to avoid bias from getting the same reward
    Q_maximum, A_maximum = self.F_prev.maximum_qval(self.weights)
    random.shuffle(A_maximum)
    self.game_ratio = game_state['round'] / 10
    self.next_action = select_policy(A_maximum[0], t_policy, t_policy_epsilon,
                                     self.game_ratio, self.logger)
    return self.next_action
