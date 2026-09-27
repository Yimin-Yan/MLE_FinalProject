import os
import pickle
import random

import numpy as np

from q_tabular_agent import TabularQAgent

from q_helpers import get_legal_actions, ACTS, N_ACTIONS, N_STATES, state_to_tabular_features, describe_tabular_state, DIRS, OCCS, OBJ_BITS

config = {
    "n_eval"              : 100,    # Number of evaluation episodes every eval_freq training episodes
    "eval_freq"           : 1000,
    "train_freq"          : 1,      # Train models every train_freq training episodes
    "discount"            : 0.8,    # Discount in all Q learning algorithms
    "exploration"         : 0.05,    # Initial exploration rate
    "exploration_decay"   : 0,   # Decrease of exploration rate for every action
    "exploration_min"     : 0.05,
    "learning_rate_mode"  : "fixed",
    "learning_rate"       : 1e-4,
    "learning_rate_decay" : 1,
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
    pass


def extract_legal_actions(features: int) -> list[str]:
    """
    Build legal actions directly from neighbour_bits.
    A direction is legal if the tile is not a WALL, BOMB, ENEMY, or DANGER.
    Bomb and Wait are added if their safety bits are set.
    """
    legal = []

    wait_bit   = (features >> 17) & 0b1
    bomb_bit   = (features >> 16) & 0b1
    neighbour_bits = (features >> 0) & 0xFFF

    for shift, dir_name in zip((9,6,3,0), DIRS):
        occ_code = (neighbour_bits >> shift) & 0b111
        occ_name = OCCS[occ_code]

        # allow movement if tile is passable
        if occ_name not in ("WALL", "ENEMY", "BOMB", "DANGER", "CRATE"):
            legal.append(dir_name)

    # safe bomb?
    if bomb_bit:
        legal.append("BOMB")

    # safe wait?
    if wait_bit:
        legal.append("WAIT")

    return legal

def act(self, game_state: dict) -> str:
    features = state_to_tabular_features(game_state)

    # legal actions derived from state bits
    legal = extract_legal_actions(features)

    # decode high-level info
    obj_bits   = (features >> 14) & 0b11
    dir_bits   = (features >> 12) & 0b11
    neighbour_bits = (features >> 0) & 0xFFF

    # --- 1. If bomb is legal & useful
    if "BOMB" in legal:
        for shift in (9,6,3,0):
            occ_code = (neighbour_bits >> shift) & 0b111
            if OCCS[occ_code] in ("CRATE", "ENEMY"):
                return "BOMB"

    # --- 2. Move towards object of interest
    if obj_bits != OBJ_BITS["NONE"]:
        preferred = DIRS[dir_bits]
        if preferred in legal:
            return preferred

    # --- 3. Random safe move
    walk_moves = [a for a in legal if a in DIRS]
    if walk_moves:
        return random.choice(walk_moves)

    # --- 4. Safe WAIT as last resort
    if "WAIT" in legal:
        return "WAIT"

    # fallback (shouldn’t happen)
    return random.choice(ACTS)
