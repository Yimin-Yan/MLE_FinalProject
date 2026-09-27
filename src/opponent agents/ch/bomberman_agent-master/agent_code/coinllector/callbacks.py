import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from random import shuffle

from settings import s
from settings import e

# Actions
actions = ('LEFT', 'RIGHT', 'UP', 'DOWN')
action_space = len(actions)
last_actions = np.zeros((0,1), dtype=np.int8)

# Hyperparameters in [0,1]
alpha = 0.75     # learning rate
gamma = 0.95     # discount factor
epsilon = 0.05   # randomness in policy

# State representation: Relative coordinates
# [left, right, top, bottom, next coin x, next coin y]
n_features = 6

# Regression
observations = np.zeros((0, n_features))
rewards = np.zeros((0, action_space))
Q = np.zeros((0, action_space))
regr = MultiOutputRegressor(LGBMRegressor(zero_as_missing=True, use_missing=False))

# True: reset weights (overwrite on HDD)
# False: use saved weights
reset = False

# True: clean duplicates from the Q-table before fitting the regression model
# False: fit model to all collected data
clean = True


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of closest target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    return best


def find_best_action(observation, logger=None):
    """Predict the best possible action in the current state.

    This functions predicts the Q-values of the current state representation
    (given by a list of features) using the fitted regression model and returns
    the index of the action reaching the highest score.

    Args:
        observation: list of features of current state.
        logger: optional logger object for debugging.
    Returns:
        index of action with the highest score.
    """

    global regr, actions
    current_rewards = regr.predict(np.asarray(observation).reshape(1, -1))
    if logger:
        logger.debug(f'Observation: {np.asarray(observation).reshape(1, -1)}')
        logger.debug(f'Predicted Rewards: {current_rewards}')
    return np.argmax(current_rewards)


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    global observations, rewards, last_actions, Q, regr, reset, clean
    if not reset:
        self.logger.debug('Loading weights')
        observations = np.load('observations.npy')
        rewards = np.load('rewards.npy')
        Q = np.load('Q.npy')
        last_actions = np.load('last_actions.npy')
        self.logger.debug(f'Data: {Q.shape}')
        self.logger.debug('Weights loaded. Training regression model...')
        if clean:
            # Remove duplicates where s=s', Q(s,a)=Q(s',a')
            unique_indices = np.unique(np.hstack((observations, Q)), axis=0, return_index=True)[1]
            relevant_Q = np.take(Q, unique_indices, axis=0)
            corresponding_states = np.take(observations, unique_indices, axis=0)
            self.logger.debug(f'Cleaned: {relevant_Q.shape}')
            regr.fit(corresponding_states, relevant_Q)
        else:
            regr.fit(observations, Q)
        self.logger.debug('Model trained')
    np.random.seed()


def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """

    global epsilon, observations, regr, actions, last_actions

    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    coins = self.game_state['coins']

    # Find absolute coordinates of the most immediately coin
    free_space = arena == 0
    targets = []
    targets.append(look_for_targets(free_space, (x,y), coins))

    # Find the adjacent fields
    left = arena[x-1, y]
    right = arena[x+1, y]
    top = arena[x, y+1]
    bottom = arena[x, y-1]
    observation = [left, right, top, bottom]

    # Find relative coordinates of targets or (0,0) if target non-existent
    for target in targets:
        if target != None:
            target = (target[0]-x, target[1]-y)
        else:
            target = (0,0)

        observation.append(target[0])
        observation.append(target[1])

    # Append the current state representation to the global list
    observations = np.vstack((observations, np.asarray(observation)))

    # Execute a random action or find the best action depending on epsilon
    if np.random.random_sample() < epsilon or reset == 1:
        last_actions = np.vstack((last_actions, np.random.randint(len(actions))))
    else:
        last_actions = np.vstack((last_actions, find_best_action(observation, self.logger)))
    self.next_action = actions[last_actions[-1][0]]

def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """

    global rewards, action_space, last_actions

    reward = 0

    # Positive rewards for good actions, negative rewards for bad actions
    for event in self.events:
        if event == e.INVALID_ACTION:
            reward = reward - 5
        elif event == e.COIN_COLLECTED:
            reward = reward + 100
        else:
            reward = reward - 1

    # Save reward in rewards list in the corresponding column for the executed action
    current_reward = np.zeros((1, action_space))
    current_reward[0][last_actions[-1]] = reward
    rewards = np.vstack((rewards, current_reward))

def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """

    global regr, observations, rewards, Q, alpha, gamma, last_actions, action_space, reset, clean

    # Final rewards, copied from reward_update()
    reward = 0

    for event in self.events:
        if event == e.INVALID_ACTION:
            reward = reward - 5
        elif event == e.COIN_COLLECTED:
            reward = reward + 100
        else:
            reward = reward - 1

    # Save reward in rewards list in the corresponding column for the executed action
    current_reward = np.zeros((1, action_space))
    current_reward[0][last_actions[-1]] = reward
    rewards = np.vstack((rewards, current_reward))

    # Learning according to the update rule given in our report
    Q = np.zeros((0, action_space))
    if not reset:
        Q_values = np.amax(regr.predict(observations), axis=1)  # Predict Q(s,a) with the current fitted model
        for i in range(len(Q_values)):
            if i < len(Q_values)-1:
                last_action = last_actions[i][0]
                current_reward = rewards[i][last_action]    # Find r
                td = np.subtract(np.add(current_reward, np.multiply(gamma, Q_values[i+1])), Q_values[i])    # Find TD-error of current (s,a) tuple
                Q_s_a = np.zeros((1, action_space))         # Find new Q(s,a)
                Q_s_a[0][last_action] = np.add(Q_values[i], np.multiply(alpha, td))
                Q = np.vstack((Q, Q_s_a))
            else:
                Q = np.vstack((Q, rewards[i])) # Final state
    else:
        Q = rewards
        reset = 0

    # Fit the regression model
    if clean:
        # Remove duplicates where s=s', Q(s,a)=Q(s',a')
        unique_indices = np.unique(np.hstack((observations, Q)), axis=0, return_index=True)[1]
        relevant_Q = np.take(Q, unique_indices, axis=0)
        corresponding_states = np.take(observations, unique_indices, axis=0)
        regr.fit(corresponding_states, relevant_Q)
    else:
        regr.fit(observations, Q)

    # Save the weights in the same folder as main.py
    np.save('observations.npy', observations)
    np.save('rewards.npy', rewards)
    np.save('Q.npy', Q)
    np.save('last_actions.npy', last_actions)
