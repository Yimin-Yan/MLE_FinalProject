import numpy as np

from collections import defaultdict, deque
from q_agent_parent import Agent
from q_helpers import TransitionFields
import pickle

class TabularQAgent(Agent):
    def __init__(self, agent_id: int, n_actions: int, n_states: int, config: dict):
        """
        Initialize a Q-learning agent with a Q-table.

        :param agent_id: The ID of the agent.
        :param n_actions: The number of available actions.
        :param n_states: The number of states in the environment.
        :param config: A dictionary containing agent configuration parameters.
        """
        super().__init__(agent_id, n_actions)
        self.n_states              = n_states
        self.initial_q             = config["initial_q"]
        # instead of a big array, use a defaultdict that creates a vector of size n_actions
        self.q                     = {}  # state → np.array of shape (n_actions,)
        self.q_visits              = {}  # state → np.array of ints (n_actions,)
        self.q_td_error            = {}  # state -> np.array (n_actions,), latest delta per action
        self.q_update_mag          = {}  # state -> np.array (n_actions,), latest |alpha*delta|

        self.train_freq            = config["train_freq"]
        self.discount              = config["discount"]
        self.learning_rate_mode    = config["learning_rate_mode"]
        self.learning_rate         = config["learning_rate"]
        self.learning_rate_decay   = config["learning_rate_decay"]
        self.exploration           = config["exploration"]
        self.exploration_decay     = config["exploration_decay"]
        self.exploration_min       = config["exploration_min"]
        self.debug                 = config["debug"]
        self.name                  = f"table-q agent {agent_id}"
        self.training_data         = []
        self.training_episodes     = []
        self.n_training_episodes   = 0
        self.all_training_episodes = []
        self.cumulative_reward     = 0
        self.cumulative_rewards    = []

    def _ensure_state(self, state):
        """Create Q & visits arrays for a new state if needed."""
        if state not in self.q:
            self.q[state]            = np.ones(self.n_actions) * self.initial_q
            self.q_visits[state]     = np.zeros(self.n_actions, dtype=int)
            self.q_td_error[state]   = np.zeros_like(self.q[state], dtype=float)
            self.q_update_mag[state] = np.zeros_like(self.q[state], dtype=float)


    def start_game(self, is_training: bool):
        """
        Set whether the agent is in training mode and reset cumulative rewards.

        :param do_training: Set training mode of agent.
        """
        super().start_game(is_training)
        self.training_data = []
        self.cumulative_reward = 0


    def act(self, state, actions):
        """
        Select an action using an epsilon-greedy policy.

        :param state: The current state.
        :param actions: List of available actions.
        :return: The selected action.
        """
        # make sure the arrays exist before we read them
        self._ensure_state(state)

        # Explore
        if np.random.uniform(0, 1) < self.exploration and self.is_training:
            action = np.random.choice(actions)
        # Exploit
        else:
            # Disable q-values of illegal actions
            illegal_actions = np.setdiff1d(np.arange(self.n_actions), actions)

            # work on a copy, leave the table intact
            q_row = self.q[state].copy()
            q_row[illegal_actions] = -np.inf          # mask only in the copy

            best = np.flatnonzero(q_row == q_row.max())
            action = np.random.choice(best)

        # Decrease exploration rate
        self.exploration = np.max([self.exploration * (1-self.exploration_decay), self.exploration_min])

        if self.debug:
            print(f"Pick action {action} in state {state} with q-values {self.q[state]}")

        return action

    def update(self, iteration, state, legal_actions, action, reward, done):
        """
        Update the Q-values based on the Q-learning update rule.

        :param iteration: The current iteration number.
        :param state: The current state.
        :param legal_actions: List of legal actions.
        :param action: The selected action.
        :param reward: The observed reward.
        :param done: True if the episode is done, False otherwise.
        """
        super().update(iteration, state, legal_actions, action, reward, done)

        self.cumulative_reward += reward
        if self.is_training:
            self.training_data.append([iteration, state, legal_actions, action, reward, done])

    def final_update(self, reward : float):
        """
        Update the Q-values at the end of an episode. Sets the last training input to done, adds final rewards, validates training data and refreshes training buffer.

        :param reward: The observed reward.
        """
        super().final_update(reward)

        self.cumulative_reward += reward

        if self.is_training:
            self.training_data[-1][TransitionFields.DONE]    = True
            self.training_data[-1][TransitionFields.REWARD] += reward

            self.validate_training_data()

            # Second pass – enrich with next_state and next_legal_actions
            for i in range(len(self.training_data) - 1):
                next_state         = self.training_data[i + 1][TransitionFields.STATE]
                next_legal_actions = self.training_data[i + 1][TransitionFields.LEGAL_ACTIONS]
                self.training_data[i].extend([next_state, next_legal_actions])

            # No next states and next legal actions in terminal state
            self.training_data[-1].extend([None, None])


            self.training_episodes.append(self.training_data)
            self.training_data = []

            self.cumulative_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0

    def validate_training_data(self):
        """
        Validate the integrity of training data, checking for missing iterations and incomplete episodes.
        """

        for i in range(len(self.training_data) - 1):
            # Validate iteration number
            i1 = self.training_data[i][TransitionFields.ITERATION]
            i2 = self.training_data[i+1][TransitionFields.ITERATION]
            if (i1 + 1 != i2):
                raise ValueError(f"Missing iteration between iterations {i1} and {i2} in training data")

    def train(self, num_episodes=None, num_transitions=None):
        """
        Train the agent by updating Q-values based on the collected training data.

        Parameters:
        - num_episodes (int, optional): Number of episodes to use from the training_episodes buffer.
        If None, use all available episodes.
        """
        if not self.is_training or not self.training_episodes:
            return


        # Use all episodes for deterministic training and remove them from the buffer
        if num_episodes is None and num_transitions is None:
            # Determine how many episodes to train on
            n = len(self.training_episodes)
            indices            = np.arange(n) # Deterministic training
            episodes_to_use    = [self.training_episodes[i] for i in indices]
            transitions        = [transition for episode in reversed(episodes_to_use) for transition in episode]

            remaining_episodes = []

        # Use the given number of episodes for determistic training and remove them from the buffer
        elif num_transitions is None:
            # Determine how many episodes to train on
            n = len(self.training_episodes)
            indices            = np.arange(n) # Deterministic training
            episodes_to_use    = [self.training_episodes[i] for i in indices[:num_episodes]]
            remaining_episodes = [self.training_episodes[i] for i in indices[num_episodes:]]
            transitions        = [transition for episode in reversed(episodes_to_use) for transition in episode]
        # Use a given number of transitions for randomised training and do not remove them from the buffer
        else:
            remaining_episodes     = self.training_episodes
            all_transitions        = [transition for episode in self.training_episodes for transition in episode]
            # Determine how many transitions to train on
            n = len(all_transitions)
            indices = np.random.permutation(n)
            print(f"Picking {num_transitions} from {n} transistions")
            transitions = [all_transitions[i] for i in indices[:num_transitions]]





        for iteration, state, legal_actions, action, reward, done, next_state, next_legal_actions in transitions:

            # make sure the arrays exist before we read them
            self._ensure_state(state)

            if self.learning_rate_mode == "adaptive":
                alpha = self.learning_rate / (1 + self.q_visits[state][action])
            elif self.learning_rate_mode == "fixed":
                alpha = self.learning_rate
            else:
                raise ValueError("Unknown learning_rate_mode")

            # Increment visit count
            self.q_visits[state][action] += 1

            # Q-learning update rule
            if done:
                target = reward
            else:
                self._ensure_state(next_state)  # make sure it's initialized

                # Ensure next_legal_actions are valid integer indices
                nla = np.asarray(next_legal_actions, dtype=int)
                next_max = np.max(self.q[next_state][nla])

                target = reward + self.discount * next_max

            # --- compute TD error and apply update ---
            delta  = target - self.q[state][action] # TD error
            update = alpha * delta                  # actual Q-step (signed)

            self.q[state][action] += update  # Q update

            # --- log metrics (latest values) ---
            self.q_td_error[state][action]   = float(delta)        # signed TD error
            self.q_update_mag[state][action] = float(abs(update))  # magnitude of update

        # Keep only remaining episodes in some training modes
        self.training_episodes = remaining_episodes


    def save_transitions(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.all_training_episodes), f)

    def load_transitions(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            self.training_episodes = pickle.load(f)
