import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from random import shuffle

from settings import s
from settings import e

# Actions
actions = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT')
action_space = len(actions)
last_actions = np.zeros((0,1), dtype=np.int8)

# Hyperparameters in [0,1]
alpha = 0.75     # learning rate
gamma = 0.95     # discount factor
epsilon = 0.0  # randomness in policy
epsilon_ = epsilon
# State representation: Relative coordinates
# [top, bottom, left, right, coin x, coin y, crate x, crate y, dead end x, dead end y, bomb x, bomb y, opp x, opp y, opp_in_env, bomb flag, crates_in_env, bomb_time]
n_features = 18

# Regression
observations = np.zeros((0, n_features))
rewards = np.zeros((0, action_space))
Q = np.zeros((0, action_space))
regr = MultiOutputRegressor(LGBMRegressor(zero_as_missing=True, use_missing=False))

# 1: reset weights (overwrite on HDD)
# 0: use saved weights
reset = 0

move = 0
# 1: clean duplicates from the Q-table before fitting the regression model
# 0: use all recorded episodes
clean = 1


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
    if reset == 0:
        self.logger.debug('Loading weights')
        observations = np.load('observations.npy')
        rewards = np.load('rewards.npy')
        Q = np.load('Q.npy')
        last_actions = np.load('last_actions.npy')
        self.logger.debug(f'Data: {observations.shape}\t{Q.shape}')
        self.logger.debug('Weights loaded. Training regression model...')
        if clean == 0:
            regr.fit(observations, Q)
        else:
            # Remove duplicates where s=s', Q(s,a)=Q(s',a')
            unique_indices = np.unique(np.hstack((observations, Q)), axis=0, return_index=True)[1]
            relevant_Q = np.take(Q, unique_indices, axis=0)
            corresponding_states = np.take(observations, unique_indices, axis=0)
            self.logger.debug(f'Cleaned: {relevant_Q.shape}')
            regr.fit(corresponding_states, relevant_Q)
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

    global epsilon, observations, regr, actions, last_actions, move

    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    coins = self.game_state['coins']
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]

    free_space = arena == 0
    targets = []

    observation = []

    # Find absolute coordinates of the most immediately targets
    targets.append(look_for_targets(free_space, (x,y), coins))
    targets.append(look_for_targets(free_space, (x,y), crates))
    targets.append(look_for_targets(free_space, (x,y), dead_ends))
    targets.append(look_for_targets(free_space, (x,y), bomb_xys))
    targets.append(look_for_targets(free_space, (x,y), others))

    # Find the adjacent fields (top, bottom, left, right)
    # 0 if free, 1 if wall/crate/explosion

    directions = ((x, y-1), (x, y+1), (x-1, y), (x+1, y))
    for d in directions:
        observation.append(arena[d] in (-1, 1) or self.game_state['explosions'][d] > 0 or d in bomb_xys)
        self.logger.debug(f'{d}\t{observation[-1]}')


    # Find relative coordinates of targets or (0,0) if target non-existent and append
    for target in targets:
        if target != None:
            target = (target[0]-x, target[1]-y)
        else:
            target = (0,0)

        observation.append(target[0])
        observation.append(target[1])

    # Check if other is in neighbor
    observation.append(0)
    directions = ((x, y-1), (x, y+1), (x-1, y), (x+1, y))
    for d in directions:
        if d in others:
            observation[-1] = 1

    # Check if other is in the same row/ column
    #observation.append(0)
    #other_pos = look_for_targets(free_space, (x,y), others)
    #if other_pos != None:
    #    if ((other_pos[0] == x and abs(other_pos[0]-x) < 5) or 
    #        (other_pos[1] == y and abs(other_pos[1]-y) < 5) ):
    #        observation[-1] = 1
    #else:
    #    observation[-1] = 0

    # Append the bomb flag
    observation.append(not bombs_left)

    # Check if crate is in neighbor
    observation.append(0)
    directions = ((x, y-1), (x, y+1), (x-1, y), (x+1, y))
    for d in directions:
        if arena[d] == 1:
            observation[-1] = 1
  
    # Append timer for bomb
    bomb_pos = look_for_targets(free_space, (x,y), bomb_xys)
    # -1 to distingush explosion after t=0 and no expl. at all t=-1
    observation.append(-1)
    for (x_,y_,t) in bombs:
        if x_ == bomb_pos[0] and y_ == bomb_pos[1]:
            observation[-1] = t 
            break

    # Append the current state representation to the global list
    observations = np.vstack((observations, np.asarray(observation)))

    # Change epsilon if agent is in a loop (Comment out for training reset)
    if np.array_equal(observations[-1][:-1], observations[-3][:-1]) and np.array_equal(observations[-3][:-1], observations[-5][:-1]):

        move = 1       
        epsilon = 0.6

    # Execute a random action or find the best action depending on epsilon
    if np.random.random_sample() < epsilon or reset == 1:

        if move == 0:
            last_actions = np.vstack((last_actions, np.random.randint(len(actions))))
        else: 
            last_actions = np.vstack((last_actions, np.random.randint(len(actions)-2)))
            move = 0

    else:
        last_actions = np.vstack((last_actions, find_best_action(observation, self.logger)))
    self.next_action = actions[last_actions[-1][0]]

    #Reset the randomness in policy
    epsilon = epsilon_


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """

    global rewards, action_space, last_actions, observations

    reward = 0

    # Positive rewards for good actions, negative rewards for bad actions
    for event in self.events:
        if event == e.INVALID_ACTION:
            reward = reward - 100
        elif event == e.CRATE_DESTROYED:
            reward = reward + 100
        elif event == e.COIN_COLLECTED:
            reward = reward + 400
            if e.BOMB_EXPLODED in self.events:
                reward = reward -300
        elif event == e.WAITED:
            reward = reward - 15
        elif event == e.BOMB_DROPPED:
            reward = reward - 20

            #Search for crates to lay bomb / no double bomb
            if ( (observations[-1][6], observations[-1][7]) == (0, 0) and
                  observations[-1][-2] == 1 and observations[-1][-3] == 0):
                reward = reward + 95
                self.logger.debug(f'First IF_Loop')

            #Search for an opponent to lay a bomb / no double bomb
            if ( observations[-1][14] == 1 and observations[-1][-3] == 0):
                reward = reward + 95
                self.logger.debug(f'Second IF_Loop')

            #No double bomb
            if observations[-1][-3] == 1:
                reward = reward -30

            #Minus if no crates and opponent / no double bomb
            if ( (observations[-1][6], observations[-1][7]) == (0, 0) and
                  observations[-1][-2] == 0 and observations[-1][14] == 0 
              and observations[-1][-3] == 0 ):
                reward = reward - 95
                self.logger.debug(f'Third IF_Loop')




        elif event == e.BOMB_EXPLODED:
            reward = reward + 0#10
        elif event == e.KILLED_SELF:
            reward = reward - 500
        elif event == e.KILLED_OPPONENT:
            reward = reward + 400
        elif event == e.GOT_KILLED:
            reward = reward - 500

        else:
            reward = reward - 1

    
    self.logger.debug(f'EVENTS: {self.events}')
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

    # Final rewards
    reward = 0

    for event in self.events:
        if event == e.SURVIVED_ROUND:
            reward = reward + 300
        elif event == e.KILLED_SELF:
            reward = reward - 500#50
        else:
            reward = reward - 1

    current_reward = np.zeros((1, action_space))
    current_reward[0][last_actions[-1]] = reward
    rewards = np.vstack((rewards, current_reward))

    # Learning according to the update rule given in our report
    Q = np.zeros((0, action_space))
    if reset == 0:
        # Predict Q(s,a) with the current fitted model
        Q_values = np.amax(regr.predict(observations), axis=1)  
        for i in range(len(Q_values)):
            if i < len(Q_values)-1:
                last_action = last_actions[i][0]
                # Find r
                current_reward = rewards[i][last_action]  
                # Find TD-error of current (s,a) tuple
                td = np.subtract(np.add(current_reward, np.multiply(gamma, Q_values[i+1])), Q_values[i])    
                Q_s_a = np.zeros((1, action_space))
                # Find new Q(s,a)
                Q_s_a[0][last_action] = np.add(Q_values[i], np.multiply(alpha, td))                         
                Q = np.vstack((Q, Q_s_a))
            else:
                # Final state
                Q = np.vstack((Q, rewards[i])) 
    else:
        Q = rewards
        reset = 0

    # if reset == 1, the agent will blow itself up during the first rounds
    # an exception will be thrown until there is sufficient data
    try:
        # Fit the regression model
        if clean == 0:
            regr.fit(observations, Q)
        else:
            # Remove duplicates where s=s', Q(s,a)=Q(s',a')
            unique_indices = np.unique(np.hstack((observations, Q)), axis=0, return_index=True)[1]
            relevant_Q = np.take(Q, unique_indices, axis=0)
            corresponding_states = np.take(observations, unique_indices, axis=0)
            regr.fit(corresponding_states, relevant_Q)
    except:
        reset = 1
    finally:
        # Strange bug that sometimes only returns the Q of the most recent episode
        # (Probably fixed already)
        if observations.shape[0] == Q.shape[0]:
            np.save('observations.npy', observations)
            np.save('rewards.npy', rewards)
            np.save('Q.npy', Q)
            np.save('last_actions.npy', last_actions)
