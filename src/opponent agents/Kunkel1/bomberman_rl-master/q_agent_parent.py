class Agent:

    def __init__(self, agent_id: int, n_actions: int):
        """
        Initialize an agent.

        :param agent_id:  The ID of the agent.
        :param n_actions: The number of available actions.
        """
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.is_training = False
        self.name = f"agent {agent_id}"
        self.cumulative_reward = 0

    def start_game(self, is_training: bool):
        """
        Set whether the agent is in training mode and reset cumulative rewards.

        :param do_training: Set training mode of the agent.
        """
        self.is_training = is_training
        self.cumulative_reward = 0

    def act(self, state, actions):
        """
        Select an action based on the current state.

        :param state: The current state.
        :param actions: List of available actions.
        :return: The selected action.
        """
        raise NotImplementedError()

    def update(self, iteration: int, state, legal_actions, action, reward, done):
        """
        Update the agent's training data based on the observed transition.

        :param iteration: The current iteration number.
        :param state: The current state.
        :param legal_actions: List of legal actions.
        :param action: The selected action.
        :param reward: The observed reward.
        :param done: True if the episode is done, False otherwise.
        """
        self.cumulative_reward += reward
        pass

    def final_update(self, reward):
        """
        Update the agent's training data at the end of an episode.

        :param reward: The observed reward.
        """
        self.cumulative_reward += reward
        pass

    def train(self):
        """
        Update the agent's Q-values based on the training data.
        """
        pass
