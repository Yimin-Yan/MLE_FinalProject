import os
import pickle
import random

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from q_deep_agent import DeepQAgent

from q_helpers import get_legal_actions, ACTS, N_ACTIONS, N_STATES, state_to_tabular_features, describe_tabular_state

base_config = {
    "n_episode"           : 2000,
    "n_eval"              : 100,    # Number of evaluation episodes every eval_freq training episodes
    "eval_freq"           : 100,
    "train_freq"          : 1,      # Train models every train_freq training episodes
    "grad_steps"          : 4,      # Number of gradient updates per training step
    "discount"            : 0.8,    # Discount in all Q learning algorithms
    "learning_rate_decay" : 1,
    "exploration"         : 1.0,    # Initial exploration rate
    "exploration_decay"   : 1e-3,   # Decrease of exploration rate for every action
    "exploration_min"     : 0.1,
    "learning_rate"       : 3e-4,
    "debug"               : False,  # Print loss and evaluation information during training
    "plot_debug"          : False,  # Plot game outcomes
    "batch_size"          : 64,    # Batch size for DQN algorithm
    "board_encoding"      : "encoding_cnn",
    "replay_buffer_size"  : 100000,  # Replay buffer for DQN algorithm
    "replay_buffer_min"   : 10000,   # minimum size before we start training
    "target_update_tau"   : 0.1,    # Weight for update in dual DQN architecture target = (1 - tau) * target + tau * online
    "target_update_freq"  : 10,     # Update target network every n episodes
    "target_update_mode"  : "hard", # "hard": update every target_update freq or "soft": update using Polyakov rule with target_update_tau
    "prb_beta_steps"      : 2e5
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
    def build_cnn(input_shape, num_actions, lr=1e-4, clipnorm=10.0):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Flatten()(x)                 # keep spatial info
        x = layers.Dense(128, activation="relu")(x)

        # value
        v = layers.Dense(128, activation="relu")(x)
        v = layers.Dense(1)(v)

        # advantage
        a = layers.Dense(128, activation="relu")(x)
        a = layers.Dense(num_actions)(a)

        # dueling combine (use Lambda to stay on-graph)
        a_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(a)
        q = layers.Add()([v, layers.Subtract()([a, a_mean])])

        model = tf.keras.Model(inputs, q)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm),
            loss=tf.keras.losses.Huber()
        )
        return model


    self.agent = DeepQAgent(
            agent_id=0,
            n_actions=N_ACTIONS,
            n_states=N_STATES,
            config=base_config,
    )

    self.agent.online_model = build_cnn(self.agent.input_shape, self.agent.n_actions, lr=base_config["learning_rate"])
    self.agent.target_model = build_cnn(self.agent.input_shape, self.agent.n_actions, lr=base_config["learning_rate"])

    if not self.train:
        # Load everything back
        self.agent.load("./snapshots", base_name="default")






def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    return ACTS[self.agent.act(game_state, actions=get_legal_actions(game_state=game_state))]

