import copy
from random import shuffle

import numpy as np

from settings import BOMB_POWER


class FeatureExtraction:
    def __init__(self, game_state, bias=0):
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

        # Defining amount of features / weights
        self.bias = bias
        self.dim = 9

        # Environment data
        self.arena = game_state['field']
        self.bombs = game_state['bombs']
        self.coins = game_state['coins']
        self.explosions = game_state['explosion_map']
        self.others = game_state['others']
        self.name, self.score, self.bombs_left, self.xy = game_state['self']
        self.x = self.xy[0]
        self.y = self.xy[1]

        self.bombs = [(xy[0], xy[1], t) for (xy, t) in self.bombs]
        self.bombs_xy = [(x, y) for (x, y, t) in self.bombs]
        self.others_xy = [(xy[0], xy[1]) for (name, score, b_left, xy) in self.others]

        self.agent = (self.x, self.y)
        self.directions = {
            'UP': (self.x, self.y - 1),
            'DOWN': (self.x, self.y + 1),
            'LEFT': (self.x - 1, self.y),
            'RIGHT': (self.x + 1, self.y),
            'BOMB': self.agent,
            'WAIT': self.agent
        }

        # Look for free tiles in the arena and compare
        self.free_space = self.arena == 0

        # Agents should not be seen as obstacles because they will most likely move in the next round.
        for x_bomb, y_bomb, _ in self.bombs:
            self.free_space[x_bomb, y_bomb] = False

        # computing the blast range manually.
        self.danger_zone = []
        if len(self.bombs) != 0:
            for x_bomb, y_bomb, _ in self.bombs:
                self.danger_zone += get_blast_coords(self.arena, x_bomb, y_bomb)

        # Safe-Zone defined
        self.safe_zone = [(x, y) for x in range(1, 16) for y in range(1, 16)
                          if (self.arena[x, y] == 0)
                          and (x, y) not in self.danger_zone]

        # In the arena, list of all crates
        self.crates = [(x, y) for x in range(1, 16) for y in range(1, 16)
                       if (self.arena[x, y] == 1)]

        # Dead ends, or tiles with only one surrounding, free tile, are computed
        self.dead_end = [(x, y) for x in range(1, 16) for y in range(1, 16)
                         if (self.arena[x, y] == 0)
                         and ([self.arena[x + 1, y],
                               self.arena[x - 1, y],
                               self.arena[x, y + 1],
                               self.arena[x, y - 1]].count(0) == 1)]

        # If walls are not taken into account, bomb map returns the maximum blast range of a bomb.
        self.bomb_map = np.ones(self.arena.shape) * 5
        for x_bomb, y_bomb, t in self.bombs:
            for (i, j) in [(x_bomb + h, y_bomb) for h in range(-3, 4)] + [(x_bomb, y_bomb + h) for h in range(-3, 4)]:
                if (0 < i < self.bomb_map.shape[0]) and (0 < j < self.bomb_map.shape[1]):
                    self.bomb_map[i, j] = min(self.bomb_map[i, j], t)

        self.feature = np.vstack(
            (self.feature0(),
             self.feature1(),
             self.feature2(),
             self.feature3(),
             self.feature4(),
             self.feature5(),
             self.feature6(),
             self.feature7(),
             self.feature8(),
             )).T

    def state(self):
        """
        The feature matrix F is returned, with each column representing a feature F i(S,A)
        and each row representing an action A.
        """
        return self.feature

    def state_action(self, action):
        """
       Return the feature's column vector
        """
        return self.feature[self.actions.index(action), :]

    def maximum_qval(self, weights):
        """
        It can be used to update weights in the midst of training or to adopt a greedy policy.
        The required weights are taken as a parameter and considered to be known.
        Returns the greatest Q-value for all conceivable actions, as well as the action that corresponds to it.
        """
        #  For every action computing the dot product (w, F_i(S,A))
        qval_lfa = np.dot(self.feature, weights)
        qval_max = np.max(qval_lfa)
        aval_max = np.where(qval_lfa == qval_max)[0]

        return qval_max, [self.actions[a] for a in aval_max]

    def feature0(self):
        return [self.bias] * len(self.actions)

    def feature1(self):
        """
        If the agent moves towards the nearest coin- Reward it!
        """
        feature = []

        # Check to see whether there are any coins in the arena and if the agent can access them directly.
        optimal_direction = search_targets_strict(self.free_space, self.agent, self.coins)
        if optimal_direction == None:
            return [0] * len(self.actions)

        # Examine whether the next move action corresponds to the direction of the coin closest to you.
        for act in self.actions:
            dir = self.directions[act]

            if dir == self.agent:
                feature.append(0)
            elif dir == optimal_direction:
                feature.append(1)
            else:
                feature.append(0)

        return feature

    def feature2(self):
        """
        Actions that leads to to agents' death in a location- Penalize it!
        """
        feature = []

        for action in self.actions:
            dir = self.directions[action]

            # To check if the tile in the next action is occupied by an object
            # (including opponents even if they move away because there is a possibility they might wait)
            if (self.arena[dir] != 0) or (dir in self.others_xy) or (dir in self.bombs_xy):
                dir = self.agent

            # To check if the agent moves into a blast range or even into an ongoing explosion
            # Such a movement in both cases, causes certain death for the agent (that is, we set F_i(s, a) = 1).
            if ((dir in self.danger_zone) and (self.bomb_map[dir] == 0)) or (self.explosions[dir] > 1):
                feature.append(1)
            else:
                feature.append(0)

        return feature

    def feature3(self):
        """
        Action that leads the agent to move towards the shortest direction outside of any blast range- Reward it!

        """
        feature = []

        # To search the arena for bombs with an explosion radius that could harm our agent.
        if len(self.bombs) == 0 or (self.agent not in self.danger_zone):
            return [0] * len(self.actions)

        # To see if the agent is able to go to a safe zone
        optimal_direction = search_targets_strict(self.free_space, self.agent, self.safe_zone)
        if optimal_direction == None:
            return [0] * len(self.actions)

        for action in self.actions:
            dir = self.directions[action]

            # We don't drop a bomb when an agent is moving away from one.
            if action == 'BOMB':
                feature.append(0)
            elif dir == optimal_direction:
                feature.append(1)
            else:
                feature.append(0)

        return feature

    def feature4(self):
        """
        Action for collecting a coin- Reward it!
        """
        feature = []

        for act in self.actions:
            dir = self.directions[act]

            if dir == self.agent:
                feature.append(0)
            elif dir in self.coins:
                feature.append(1)
            else:
                feature.append(0)

        return feature

    def feature5(self):
        """
        Action of placing a bomb next to the crate only if it can outrun the blast radius later- Reward it!

        """
        feature = []

        for act in self.actions:
            if act == 'BOMB' and self.bombs_left > 0:
                CHECK_FOR_CRATE = False
                for dir in self.directions.values():
                    if self.arena[dir] == 1:
                        CHECK_FOR_CRATE = True
                        break

                # Not rewarding the agent if it cannot outrun the blast radius later
                danger_zone = copy.deepcopy(self.danger_zone)
                danger_zone += get_blast_coords(self.arena, self.x, self.y)

                safe_zone = [(x, y) for x in range(1, 16) for y in range(1, 16)
                             if self.arena[x, y] == 0 and (x, y) not in danger_zone]
                best_coord = search_targets_strict(self.free_space, self.agent, safe_zone)

                if CHECK_FOR_CRATE and best_coord != None:
                    feature.append(1)
                else:
                    feature.append(0)
            else:
                feature.append(0)

        return feature

    def feature6(self):
        """
        If the agent moves towards the next crate- Reward it!
        """
        feature = []

        free_space = copy.deepcopy(self.free_space)
        for x_crate, y_crate in self.crates:
            free_space[x_crate, y_crate] = True

        # We don't distinguish between crates that can be reached by the agent and those that can't.
        optimal_direction = search_targets(free_space, self.agent, self.crates)
        if optimal_direction == None:
            return [0] * len(self.actions)

        for action in self.actions:
            d = self.directions[action]

            # To get the agent towards a crate by a move action.
            if d == self.agent:
                feature.append(0)
            else:
                # No reward given if the agent is already next to a crate.
                if d in self.crates:
                    return [0] * len(self.actions)
                if d == optimal_direction:
                    feature.append(1)
                else:
                    feature.append(0)

        return feature

    def feature7(self):
        """
       Action that leads the agent into getting trapped by it's own bomb- Penalize it!
        """
        feature = []

        for action in self.actions:
            d = self.directions[action]

            if d == self.agent:
                feature.append(0)
            elif d in self.dead_end and self.agent in self.bombs_xy:
                feature.append(1)
            else:
                feature.append(0)

        return feature

    def feature8(self):
        """
        If our agent can escape from it, the act of placing a bomb to kill an agent- Reward it!
        """

        feature = []

        danger_zone = copy.deepcopy(self.danger_zone)
        my_bomb_zone = get_blast_coords(self.arena, self.x, self.y)
        danger_zone += my_bomb_zone
        safe_zone = [(x, y) for x in range(1, 16) for y in range(1, 16)
                     if (self.arena[x, y] == 0)
                     and (x, y) not in danger_zone]

        optimal_path = search_targets_path(self.free_space, self.agent, safe_zone)

        for action in self.actions:
            if action == 'BOMB':
                CHECK_COND = False
                for others in self.others_xy:
                    if len(optimal_path) == 0:
                        return [0] * len(self.actions)
                    if (others in my_bomb_zone) and (optimal_path[-1] in safe_zone):
                        CHECK_COND = True
                        break
                if CHECK_COND and (self.bombs_left > 0):
                    feature.append(1)
                else:
                    feature.append(0)
            else:
                feature.append(0)

        return feature


def search_targets_path(free_space, start, targets, logger=None):
    """ Determine the direction of the nearest target that can be reached using only free tiles.
    Searches the attainable free tiles in a breadth-first manner until a target is found.
    If no target can be reached, the agent closest to a target is chosen.

    Arguments:
        start: the location where the search should begin.
        free_space is a numpy array with a Boolean value. True in the case of free tiles and False in the case of barriers.
        logger: a debugging logger object that is optional.
        targets: a list or array containing all target tile coordinates.
    Returns: the path to the nearest target or the tile closest to any target, starting at the next step.
        """
    if len(targets) == 0:
        return []

    starting_val = [start]
    parent_dict = {start: start}
    distance_now = {start: 0}
    optimal = start
    optimal_dis = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(starting_val) > 0:
        current_location = starting_val.pop(0)

        dist = np.sum(np.abs(np.subtract(targets, current_location)), axis=1).min()
        if dist + distance_now[current_location] <= optimal_dis:
            optimal = current_location
            optimal_dis = dist + distance_now[current_location]
        if dist == 0:
            optimal = current_location
            break

        x_cur, y_cur = current_location
        neighbors = [(x_cur, y_cur) for (x_cur, y_cur) in
                     [(x_cur + 1, y_cur), (x_cur - 1, y_cur), (x_cur, y_cur + 1), (x_cur, y_cur - 1)] if
                     free_space[x_cur, y_cur]]
        shuffle(neighbors)

        for n in neighbors:
            if n not in parent_dict:
                starting_val.append(n)
                parent_dict[n] = current_location
                distance_now[n] = distance_now[current_location] + 1

    if logger:
        logger.debug(f'Suitable target found at {optimal}')

    current_location = optimal
    path = []
    while True:
        path.insert(0, current_location)
        if parent_dict[current_location] == start:
            return path
        current_location = parent_dict[current_location]


def search_targets(free_space, start, targets, logger=None):
    """
    Returns the co-ordinate of the first step towards the nearest target or the tile that is closest to any target
    """
    optimal_path = search_targets_path(free_space, start, targets, logger)

    if len(optimal_path) != 0:
        return optimal_path[0]
    else:
        return None


def search_targets_strict(free_space, start, targets, logger=None):
    """Similar to search targets, but this function only returns a direction
        if a target can be reached from the starting point.
    """
    optimal_path = search_targets_path(free_space, start, targets, logger)

    if (len(optimal_path) != 0) and (optimal_path[-1] in targets):
        return optimal_path[0]
    else:
        return None


def get_blast_coords(arena, x, y):
    """ For a bomb, get the blast range.

    Parameters:
    * arena:  a two-dimensional array defining the game's arena
    * x, y:   Co-ordinates of the bomb.

    Return Value:
    * Array holding each bomb's blast range coordinate.
    """
    bomb_power = BOMB_POWER
    blast_position = [(x, y)]

    for i in range(1, bomb_power + 1):
        if arena[x + i, y] == -1: break
        blast_position.append((x + i, y))
    for i in range(1, bomb_power + 1):
        if arena[x - i, y] == -1: break
        blast_position.append((x - i, y))
    for i in range(1, bomb_power + 1):
        if arena[x, y + i] == -1: break
        blast_position.append((x, y + i))
    for i in range(1, bomb_power + 1):
        if arena[x, y - i] == -1: break
        blast_position.append((x, y - i))

    return blast_position
