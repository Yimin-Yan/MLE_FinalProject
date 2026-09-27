import pickle
from typing import List

import events as e
from q_helpers import get_legal_actions, state_to_tabular_features, reward_from_events, ACTS, N_ACTIONS, N_STATES, ACT_BITS
import numpy as np

import os
import numpy as np
import tensorflow as tf
import subprocess

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
    self.chunk_idx = 0
    self.cumulative_reward = 0



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

    reward = reward_from_events(events)
    self.cumulative_reward += reward

    # state_to_features is defined in callbacks.py
    self.agent.update(iteration = self.iteration,
                      state = old_game_state,
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

    reward = reward_from_events(events)
    self.cumulative_reward += reward

    self.agent.update(iteration = self.iteration,
                      state = last_game_state,
                      legal_actions = get_legal_actions(last_game_state),
                      action = ACT_BITS[last_action],
                      reward = reward,
                      done = True)
    self.agent.final_update(reward = 0) # All the final rewards are handed out before, no additional reward is necessary
    self.agent.train()

    if self.game % 250  == 0:
        print(f"Reward: {self.cumulative_reward} Exploration: {self.agent.exploration:.2f}")

    if self.game % 1999  == 0:
        # Save agent state, q_visits, and models
        self.agent.save("./snapshots", base_name=f"experiment_{self.chunk_idx:02d}")
        self.agent.save("./snapshots", base_name="default")

        evaluate_agent(self.chunk_idx, "./")
        self.chunk_idx += 1


    self.cumulative_reward = 0
    self.iteration += 1
    self.game += 1



def evaluate_agent(chunk_idx, out_dir):
    """Run main.py play with the trained agent and save stats."""
    dicts_dir = os.path.join(out_dir, "dicts")
    stats_file = os.path.join(dicts_dir, f"eval_chunk_{chunk_idx:04d}.json")

    cmd = [
        "python3", "../../main.py",
        "play",
        "--agents", "cnn_allstar_duel", "rule_based_agent", "peaceful_agent", "tq_representator",
        "--n-rounds", "100",
        "--train", "0",
        "--no-gui",
        "--save-stats", stats_file,
        "--match-name", f"chunk_{chunk_idx}"
    ]

    print(f"[Chunk {chunk_idx}] Evaluating agent with 50 rounds...")
    subprocess.run(cmd, check=True)
    print(f"[Chunk {chunk_idx}] Evaluation stats saved to {stats_file}")
