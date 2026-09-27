import os
import pickle
import random

import numpy as np

from q_tabular_agent import TabularQAgent

from q_helpers import get_legal_actions, ACTS, N_ACTIONS, N_STATES, state_to_tabular_features, describe_tabular_state

config = {
    "n_episode"           : 50000,  # Number of training episodes
    "n_eval"              : 100,    # Number of evaluation episodes every eval_freq training episodes
    "eval_freq"           : 1000,
    "train_freq"          : 1,      # Train models every train_freq training episodes
    "discount"            : 0.95,    # Discount in all Q learning algorithms
    "exploration"         : 0.0,    # Initial exploration rate
    "exploration_decay"   : 1e-5,   # Decrease of exploration rate for every action
    "exploration_min"     : 0.0,
    "learning_rate_mode"  : "adaptive",
    "learning_rate"       : 1,
    "learning_rate_decay" : .9999,
    "randomise_order"     : False,  # Randomise starting order of agents for every game
    "only_legal_actions"  : True,   # Have agents only take legal actions
    "debug"               : False,  # Print loss and evaluation information during training
    "initial_q"           : 0.0,      # Initial Q value for tabular Q learning
}

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

    self.agent = TabularQAgent(
            agent_id=0,
            n_actions=N_ACTIONS,
            n_states=N_STATES,
            config=config,
    )


    filepath = "q_table.npz"

    if os.path.isfile(filepath):
        print("Loading model from saved state.")
        self.logger.info("Loading model from saved state.")

        data     = np.load(filepath, allow_pickle=True)
        self.agent.q           = data["q"].item()
        self.agent.q_visits    = data["q_visits"].item()




def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    features = state_to_tabular_features(game_state)

    if not self.train:
        print(" | ".join(
            f"{a[:1]}: {self.agent.q[features][i]:.2f}" +
            (f" ({self.agent.q_visits[features][i]:.1e}x)" if features in self.agent.q_visits else "")
            for i, a in enumerate(ACTS)
        ) if features in self.agent.q else "Not in table")


    return ACTS[self.agent.act(features, actions=get_legal_actions(game_state=game_state))]

