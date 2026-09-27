import os
import pickle
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
STATE = 17
EXPLORATION_RATE = 0.1
EXPLORE_COUNT = 0
EXPLOIT_COUNT = 0


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    Loading the Q-table is done.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Loading Q_sa_inc_3 function from saved state.")
    with open('Q_sa_upload.npy', 'rb') as f:
        self.q_sa = np.load(f, allow_pickle=True)
    self.q_sa = self.q_sa.tolist()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    global EXPLORE_COUNT
    global EXPLOIT_COUNT
    q = self.q_sa
    current_idx = features_to_index(state_to_features(game_state))

    p = np.random.random()
    if p < EXPLORATION_RATE:
        self.logger.debug("Exploring")
        EXPLORE_COUNT += 1
        # action = np.random.choice(ACTIONS)
        # Do the second best action sometimes in case of the agent stuck in a loop
        second_best = list(q[current_idx])[np.argsort(list(q[current_idx].values()))[-2]]
        if q[current_idx][second_best] > 0.1:
            action = second_best
        else:
            action = ACTIONS[np.argmax(list(q[current_idx].values()))]
    else:
        self.logger.debug("Exploiting (predict actions)")
        EXPLOIT_COUNT += 1
        action = ACTIONS[np.argmax(list(q[current_idx].values()))]

    self.logger.debug(f"Explore count: {EXPLORE_COUNT}, Exploit count: {EXPLOIT_COUNT}")
    self.logger.debug(f"Q: {list(q[features_to_index(state_to_features(game_state))].values())}")
    self.logger.debug(f"Index: {features_to_index(state_to_features(game_state))}")
    self.logger.debug(f"Took action {action}")
    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    self_cor_channel = np.array(game_state["self"][3])
    down_channel = [self_cor_channel[1] + 1, self_cor_channel[0]]
    up_channel = [self_cor_channel[1] - 1, self_cor_channel[0]]
    left_channel = [self_cor_channel[1], self_cor_channel[0] - 1]
    right_channel = [self_cor_channel[1], self_cor_channel[0] + 1]

    left = cor_states(game_state, left_channel, 'left')
    right = cor_states(game_state, right_channel, 'right')
    up = cor_states(game_state, up_channel, 'up')
    down = cor_states(game_state, down_channel, 'down')
    # self_ch = cor_states(game_state, self_cor_channel) # no need to check current position

    stacked_channels = np.array([left] + [right] + [up] + [down] + [game_state["self"][2]])
    stacked_channels = np.concatenate(stacked_channels, axis=None)
    # and return them as a vector
    return stacked_channels


def cor_states(game_state, coordinates, direction_str):
    """
    Checks the given coordinates if there is a coin, if there is a wall or crate or free, and if there is a danger.
    Danger also contains walls as well.

    :param game_state:  A dictionary describing the current game board.
    :param coordinates:  The given coordinates.
    :param direction_str:  The direction that is currently checking.
    :return: np.array of size 4 -> (1 (if coin), 1 (if crate), 1 (if danger), 1 (if dead end))
    """
    field_channel = np.array(game_state["field"]).T
    coins = np.array(game_state["coins"]).T
    bombs = np.array(game_state["bombs"]).T
    explosion = np.array(game_state["explosion_map"])

    # Below changes the field map's some free-tile values (0) to crate (1). Those changed free-tile values coordinates
    # are same as the opponent coordinates. Since both crates and opponents are the things that need to be destroyed, we
    # decided to make this change.
    for other_idx in range(len(np.array(game_state["others"]))):
        other_cor_channel = np.array(game_state["others"][other_idx][3])
        field_channel[other_cor_channel[1], other_cor_channel[0]] = 1

    # We create 4 bits for each directions' states. First bit is for coins, second bit is for crates, third bit is for
    # checking danger, and the last bit is for checking dead ends.
    state_bits = np.zeros(4)

    # Below checks if the given coordinate is free, crate or a wall. If it's free, second element of the state_bits
    # array becomes 0, if it's a crate then the same element becomes 1, and if it's a wall, third element of the
    # state_bits array becomes 1.
    if field_channel[coordinates[0], coordinates[1]] == 1:
        state_bits[1] = 1
    elif field_channel[coordinates[0], coordinates[1]] == 0:
        state_bits[1] = 0
    else:
        state_bits[2] = 1

    # Below checks if there exists a coin in the given coordinate. If there is a coin in the given direction within the
    # 3 blocks range, the coin state becomes 1.
    if coins.size != 0:
        for ind in range(len(coins[0])):
            if coins[0, ind] == coordinates[1] and coins[1, ind] == coordinates[0]:
                state_bits[0] = 1
            elif direction_str == 'left' and coins[1, ind] == coordinates[0] and (coins[0, ind] == coordinates[1] - 1 or coins[0, ind] == coordinates[1] - 2 or coins[0, ind] == coordinates[1] - 3):
                state_bits[0] = 1
            elif direction_str == 'right' and coins[1, ind] == coordinates[0] and (coins[0, ind] == coordinates[1] + 1 or coins[0, ind] == coordinates[1] + 2 or coins[0, ind] == coordinates[1] + 3):
                state_bits[0] = 1
            elif direction_str == 'down' and coins[0, ind] == coordinates[1] and (coins[1, ind] == coordinates[0] + 1 or coins[1, ind] == coordinates[0] + 2 or coins[1, ind] == coordinates[0] + 3):
                state_bits[0] = 1
            elif direction_str == 'up' and coins[0, ind] == coordinates[1] and (coins[1, ind] == coordinates[0] - 1 or coins[1, ind] == coordinates[0] - 2 or coins[1, ind] == coordinates[0] - 3):
                state_bits[0] = 1

    # Below checks if the given coordinate is dangerous or not. If there is an explosion or a bomb that is about to
    # explode danger state is given. Since we do not want our agent to go towards a wall, all walls are seen as
    # danger as well.
    if bombs.size != 0:
        for idx in range(len(bombs[0])):
            check = list(bombs[0][idx])
            if bombs[1][idx] < 1:
                if (check[0] == coordinates[1] and np.abs(check[1] - coordinates[0]) < 4) or (
                        check[1] == coordinates[0] and np.abs(check[0] - coordinates[1]) < 4):
                    state_bits[2] = 1
            elif bombs[1][idx] == 1:
                if (check[0] == coordinates[1] and np.abs(check[1] - coordinates[0]) < 3 and (field_channel[coordinates[0], coordinates[1] - 1] != 0 and field_channel[coordinates[0], coordinates[1] + 1] != 0)) or (
                        check[1] == coordinates[0] and np.abs(check[0] - coordinates[1]) < 3 and (field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[coordinates[0] + 1, coordinates[1]] != 0)):
                    state_bits[2] = 1
            if check[0] == coordinates[1] and check[1] == coordinates[0]:
                state_bits[2] = 1
    if explosion[coordinates[1], coordinates[0]] != 0:
        state_bits[2] = 1

    # Below checks if the given coordinate is a dead end or not. If the 3 further blocks among the given direction are
    # free, not dead end state is given.
    if bombs.size != 0:
        for idx in range(len(bombs[0])):
            check = list(bombs[0][idx])
            dist_bomb = np.sum(np.abs(np.array(check) - np.array(coordinates[::-1])))
            if dist_bomb < 4:
                if direction_str == 'left' and (coordinates[1] > 2) and (check[0] == coordinates[1] + 1 or check[1] == coordinates[0]) and (
                        (field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                            coordinates[0], coordinates[1] - 1] != 0 and field_channel[
                             coordinates[0] + 1, coordinates[1]] != 0) or (
                                field_channel[coordinates[0] - 1, coordinates[1]] != 0 and
                                field_channel[
                                    coordinates[0] + 1, coordinates[1]] != 0 and
                                field_channel[coordinates[0] - 1, coordinates[1] - 1] != 0 and
                                field_channel[coordinates[0], coordinates[1] - 2] != 0 and field_channel[
                                    coordinates[0] + 1, coordinates[1] - 1] != 0)
                    or (
                                field_channel[coordinates[0] - 1, coordinates[1]] != 0 and
                                field_channel[
                                    coordinates[0] + 1, coordinates[1]] != 0 and
                                field_channel[coordinates[0] - 1, coordinates[1] - 1] != 0 and field_channel[
                                    coordinates[0] + 1, coordinates[1] - 1] != 0 and
                                field_channel[coordinates[0] - 1, coordinates[1] - 2] != 0 and
                                field_channel[coordinates[0], coordinates[1] - 3] != 0 and field_channel[
                                    coordinates[0] + 1, coordinates[1] - 2] != 0)):
                    state_bits[3] = 1
                elif direction_str == 'left' and (coordinates[1] == 1 or coordinates[1] == 2) and (
                        field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                    coordinates[0], coordinates[1] - 1] != 0 and field_channel[
                            coordinates[0] + 1, coordinates[1]] != 0):
                    state_bits[3] = 1
                elif direction_str == 'right' and (coordinates[1] < 14) and (check[0] == coordinates[1] - 1 or check[1] == coordinates[0]) and (
                        (field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                            coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                             coordinates[0] + 1, coordinates[1]] != 0) or (
                                field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                            coordinates[0] + 1, coordinates[1]] != 0 and
                                field_channel[coordinates[0] - 1, coordinates[1] + 1] != 0 and
                                field_channel[coordinates[0], coordinates[1] + 2] != 0 and field_channel[
                                    coordinates[0] + 1, coordinates[1] + 1] != 0)
                    or (
                                field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                            coordinates[0] + 1, coordinates[1]] != 0 and
                                field_channel[coordinates[0] - 1, coordinates[1] + 1] != 0 and field_channel[
                                    coordinates[0] + 1, coordinates[1] + 1] != 0 and
                                field_channel[coordinates[0] - 1, coordinates[1] + 2] != 0 and
                                field_channel[coordinates[0], coordinates[1] + 3] != 0 and field_channel[
                                    coordinates[0] + 1, coordinates[1] + 2] != 0)):
                    state_bits[3] = 1
                elif direction_str == 'right' and (coordinates[1] == 15 or coordinates[1] == 14) and (
                        field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                    coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                            coordinates[0] + 1, coordinates[1]] != 0):
                    state_bits[3] = 1
                elif direction_str == 'up' and (coordinates[0] > 2) and (check[0] == coordinates[1] or check[1] == coordinates[0] - 1) and (
                        (field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                            coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                             coordinates[0], coordinates[1] - 1] != 0) or (field_channel[
                                                                               coordinates[0], coordinates[1] + 1] != 0 and
                                                                           field_channel[
                                                                               coordinates[0], coordinates[1] - 1] != 0 and
                                                                           field_channel[
                                                                               coordinates[0] - 1, coordinates[1] + 1] != 0 and
                                                                           field_channel[
                                                                               coordinates[0] - 2, coordinates[1]] != 0 and
                                                                           field_channel[
                                                                               coordinates[0] - 1, coordinates[1] - 1] != 0)
                    or (
                                field_channel[
                                    coordinates[0], coordinates[1] + 1] != 0 and
                                field_channel[
                                    coordinates[0], coordinates[1] - 1] != 0 and
                                field_channel[
                                    coordinates[0] - 1, coordinates[1] + 1] != 0 and
                                field_channel[
                                    coordinates[0] - 1, coordinates[1] - 1] != 0 and
                                field_channel[coordinates[0] - 2, coordinates[1] + 1] != 0 and
                                field_channel[coordinates[0] - 3, coordinates[1]] != 0 and field_channel[
                                    coordinates[0] - 2, coordinates[1] - 1] != 0)):
                    state_bits[3] = 1
                elif direction_str == 'up' and (coordinates[0] == 1 or coordinates[0] == 2) and (
                        field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                    coordinates[0], coordinates[1] + 1] != 0 and field_channel[coordinates[0], coordinates[1] - 1] != 0):
                    state_bits[3] = 1
                elif direction_str == 'down' and (coordinates[0] < 14) and (check[0] == coordinates[1] or check[1] == coordinates[0] + 1) and (
                        (field_channel[coordinates[0] + 1, coordinates[1]] != 0 and field_channel[
                            coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                             coordinates[0], coordinates[1] - 1] != 0) or (field_channel[
                                                                               coordinates[0], coordinates[1] + 1] != 0 and
                                                                           field_channel[
                                                                               coordinates[0], coordinates[1] - 1] != 0 and
                                                                           field_channel[
                                                                               coordinates[0] + 1, coordinates[1] + 1] != 0 and
                                                                           field_channel[
                                                                               coordinates[0] + 2, coordinates[1]] != 0 and
                                                                           field_channel[
                                                                               coordinates[0] + 1, coordinates[1] - 1] != 0)
                    or (field_channel[
                            coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                             coordinates[0], coordinates[1] - 1] != 0 and
                                field_channel[coordinates[0] + 1, coordinates[1] + 1] != 0 and field_channel[
                                    coordinates[0] + 1, coordinates[1] - 1] != 0 and
                                field_channel[coordinates[0] + 2, coordinates[1] + 1] != 0 and
                                field_channel[coordinates[0] + 3, coordinates[1]] != 0 and field_channel[
                                    coordinates[0] + 2, coordinates[1] - 1] != 0)):
                    state_bits[3] = 1
                elif direction_str == 'down' and (coordinates[0] == 15 or coordinates[0] == 14) and (
                        field_channel[coordinates[0] + 1, coordinates[1]] != 0 and field_channel[
                    coordinates[0], coordinates[1] + 1] != 0 and field_channel[coordinates[0], coordinates[1] - 1] != 0):
                    state_bits[3] = 1

    return state_bits


def features_to_index(features):
    """
    Converts the given feature array to an index which will be used as the q-table index.

    :param features:  Each direction has 4 elements of features so 16 from there and 1 element for checking to
    drop a bomb.
    :return: np.array of size 4 -> (1 (if coin), 1 (if crate), 1 (if danger), 1 (if dead end))
    """
    state_no = STATE
    index = 0
    for n in range(state_no):
        index += 2 ** (state_no - n - 1) * features[n]
    return index
