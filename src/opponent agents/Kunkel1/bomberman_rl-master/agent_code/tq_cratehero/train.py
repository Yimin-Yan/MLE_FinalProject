import pickle
from typing import List

import events as e
from q_helpers import get_legal_actions, state_to_tabular_features, reward_from_events, ACTS, N_ACTIONS, N_STATES, ACT_BITS
import numpy as np
import os


def save_snapshot(agent, out_dir, base_name, chunk_idx=None, compressed=True):
    """
    Writes a single NPZ containing all dicts + metadata.
    Dicts are saved as 0-D object arrays so they round-trip cleanly.
    """
    out_dir2 = os.path.join(out_dir, "dicts")
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(out_dir2, exist_ok=True)
    fname = f"{base_name}{'' if chunk_idx is None else f'_chunk_{chunk_idx:04d}'}"
    path = os.path.join(out_dir2, f"{fname}.npz")
    base_fname = f"{base_name}"
    bpath = os.path.join(out_dir, f"{base_fname}.npz")

    payload = {
        "q":            np.array(agent.q,            dtype=object),
        "q_visits":     np.array(agent.q_visits,     dtype=object),
        "q_td_error":   np.array(agent.q_td_error,   dtype=object),
        "q_update_mag": np.array(agent.q_update_mag, dtype=object),
        "meta": np.array({
            "chunk_idx":          chunk_idx,
            "train_step":         getattr(agent, "_train_step", None),
            "learning_rate":      getattr(agent, "learning_rate", None),
            "learning_rate_mode": getattr(agent, "learning_rate_mode", None),
            "discount":           getattr(agent, "discount", None),
            "epsilon":            getattr(agent, "epsilon", None),
            "notes": "All dicts stored as 0-D object arrays; load with allow_pickle=True and .item().",
        }, dtype=object),
    }

    if compressed:
        np.savez_compressed(path, **payload)
        np.savez_compressed(bpath, **payload)
    else:
        np.savez(path, **payload)
        np.savez(bpath, **payload)

    print(f"[Chunk {chunk_idx}] Saved single-archive snapshot to {path}")
    return path

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.agent.start_game(is_training=True)
    self.iteration = 0
    self.game = 0

    self.reward = 0
    self.cumreward = 0



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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(self, events)

    self.cumreward += reward

    # state_to_features is defined in callbacks.py
    self.agent.update(iteration = self.iteration,
                      state = state_to_tabular_features(old_game_state),
                      legal_actions = get_legal_actions(old_game_state),
                      action = ACT_BITS[self_action],
                      reward = reward,
                      done = False)
    self.iteration += 1

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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(self, events)

    self.cumreward += reward

    self.agent.update(iteration = self.iteration,
                      state = state_to_tabular_features(last_game_state),
                      legal_actions = get_legal_actions(last_game_state),
                      action = ACT_BITS[last_action],
                      reward = reward,
                      done = True)
    self.agent.final_update(reward = 0) # All the final rewards are handed out before, no additional reward is necessary
    self.agent.train()

    if self.game % 100 == 0:
        chunk_idx = int(self.game/100)
        save_snapshot(self.agent, "./", "online", chunk_idx=chunk_idx)
        print("Saving snapshot: {chunk_idx}")

    print("Game {self.game}: Reward = {self.cumreward}")

    self.iteration += 1
    self.game += 1


