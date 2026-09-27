
import os
import random
import numpy as np
from collections import deque
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint  # Import ModelCheckpoint callback
from tensorflow.keras import layers, models

from q_agent_parent import Agent
from q_prioritised_experience_replay import PrioritizedReplayBuffer
from q_helpers import TransitionFields, state_to_tabular_features, state_to_cnn_features
from settings import BOMB_TIMER, EXPLOSION_TIMER, BOMB_POWER

# Define a directory to save checkpoints and logs
checkpoint_dir = 'checkpoints'
log_dir = 'logs'

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Base class for sampling buffer (either uniform buffer or prioritised experience replay buffer)
class ReplaySampler:
    def sample(self, buffer, batch_size):
        raise NotImplementedError

    def add(self, buffer, transition):
        raise NotImplementedError

    def update_priorities(self, buffer, idxs, td_errors):
        pass  # No-op for uniform replay

class UniformReplaySampler(ReplaySampler):
    def sample(self, buffer, batch_size):
        minibatch = random.sample(buffer, batch_size)
        idxs = None
        weights = np.ones(batch_size, dtype=np.float32)  # uniform weights
        return minibatch, idxs, weights

    def add(self, buffer, transition):
        buffer.append(transition)

class PrioritizedReplaySampler(ReplaySampler):
    td_error_list = []

    def sample(self, buffer, batch_size):
        minibatch, idxs, weights = buffer.sample(batch_size)
        return minibatch, idxs, weights

    def update_priorities(self, buffer, idxs, td_errors):
        buffer.update_priorities(idxs, td_errors)
        self.td_error_list.append(td_errors)

    def add(self, buffer, transition):
        buffer.add(transition)


class WeightedTrainer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE  # we need per-sample losses
        )

    @tf.function(jit_compile=True)
    def train_step(self, x, y, sample_weights=None):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            per_sample_losses = self.loss_fn(y, y_pred)   # shape (B,)
            if sample_weights is not None:
                per_sample_losses *= tf.cast(sample_weights, per_sample_losses.dtype)
            loss = tf.reduce_mean(per_sample_losses)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss




class DeepQAgent(Agent):
    # encoding of feature vector
    ENCODING_BINARY   = 0 # convert 18-bit simplified state representation to base 2-number and pass to NN as vector
    ENCODING_CNN      = 1 # one-hot encoding for CNN
    INPUT_ENCODINGS = {
        "encoding_binary": ENCODING_BINARY,
        "encoding_cnn": ENCODING_CNN
    }
    LARGE_NEGATIVE_NUMBER = -1e6

    def __init__(self, agent_id, n_actions, n_states, config):
        """
        Initialize a Deep Q-Network (DQN) agent for reinforcement learning.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing agent configuration parameters.
                                    - 'board_size': The size of the game board.
                                    - 'n_episode': The total number of episodes for training.
                                    - 'discount': The discount factor (gamma).
                                    - 'learning_rate': The learning rate (alpha).
                                    - 'learning_rate_decay': The learning rate decay factor.
                                    - 'exploration': The exploration rate (epsilon).
                                    - 'exploration_decay': The exploration rate decay factor.
                                    - 'exploration_min': Minimum exploration rate.
                                    - 'batch_size': The batch size for training.
                                    - 'replay_buffer_size': The size of the replay buffer.
                                    - 'target_update_tau':  The weighting between online and target network for updating the target network.
        """
        super().__init__(agent_id, n_actions)
        self.n_states            = n_states

        self.name                = f"deep-q agent {agent_id}"
        self.n_episode           = config["n_episode"]
        self.grad_steps          = config["grad_steps"]
        self.discount            = config["discount"]
        self.learning_rate       = config["learning_rate"]
        self.learning_rate_decay = config["learning_rate_decay"]
        self.exploration         = config["exploration"]
        self.exploration_decay   = config["exploration_decay"]
        self.exploration_min     = config["exploration_min"]
        self.batch_size          = config["batch_size"]
        self.replay_buffer_size  = config["replay_buffer_size"]
        self.replay_buffer_min   = config["replay_buffer_min"]
        self.target_update_mode  = config["target_update_mode"]
        self.target_update_freq  = config["target_update_freq"]
        self.target_update_tau   = config["target_update_tau"]
        self.debug               = config["debug"]

        self.training_round      = 0
        self.training_data       = []
        self.training_log        = []

        self.enable_double_dqn   = config.get("enable_double_dqn", True)
        if self.enable_double_dqn:
            print("Double DQN enabled!")
        self.enable_prio_exp_rep = config.get("enable_prioritised_experience_replay", True)
        if self.enable_prio_exp_rep:
            print("Prioritised Experience Replay enabled!")

        if not self.enable_prio_exp_rep:
            self.replay_buffer  = deque(maxlen=self.replay_buffer_size)
            self.replay_sampler = UniformReplaySampler()
        else:
            self.replay_buffer  = PrioritizedReplayBuffer(
                capacity=self.replay_buffer_size,
                alpha      = config.get("prb_alpha", 0.6),
                beta0      = config.get("prb_beta0", 0.4),
                # anneal beta to 1 over the course of the training, in practive, we reach 1 a little sooner than at the end because of buffer filling up in the beginning
                beta_steps = config.get("prb_beta_steps", 2e5),
                epsilon    = config.get("prb_epsilon", 1e-6),
            )

            self.replay_sampler = PrioritizedReplaySampler()


        self._input_cache       = {}

        # choose encoding of feature vector
        self.board_encoding      = self.INPUT_ENCODINGS[config["board_encoding"]]
        if self.board_encoding == self.ENCODING_BINARY:
            self.input_shape     = 18
        elif self.board_encoding == self.ENCODING_CNN:
            self.input_shape     = (9, 9, 11)
        else:
            raise ValueError("Unknown board encoding")


        # Models need to be defined and compiled in derived classes
        self.online_model        = None
        self.target_model        = None


    def update(self, iteration, state, legal_actions, action, reward, done):
        """
        Update the Q-values based on the Q-learning update rule.

        :param iteration:      The current iteration.
        :param state:          The current state.
        :param legal_actions:  List of legal actions.
        :param action:         The selected action.
        :param reward:         The observed reward.
        :param done:           Boolean indicating whether the episode is done.
        """
        super().update(iteration, state, legal_actions, action, reward, done)
        if self.is_training:
            self.training_data.append([iteration, self.state_to_input(state), legal_actions, action, reward, done])


    def final_update(self, reward):
        """
        Update the training data after the final step of an episode.

        :param reward: The final observed reward.
        """
        super().final_update(reward)

        if self.is_training:
            self.training_data[-1][TransitionFields.DONE]    = True
            self.training_data[-1][TransitionFields.REWARD] += reward

            self.validate_training_data()
            self.move_training_data_to_replay_buffer()

            # Decrease exploration rate
            self.exploration = np.max([self.exploration * (1-self.exploration_decay), self.exploration_min])


    def encode_binary(self, game_state: dict) -> np.ndarray:
        """Convert a decimal number to a base-N vector of fixed length."""
        n = state_to_tabular_features(game_state)
        base    = 2
        padding = 18 # 18 bit for state representation
        digits = []

        # Convert to base
        while n > 0:
            digits.append(n % base)
            n //= base

        # Pad with zeros and reverse
        while len(digits) < padding:
            digits.append(0)
        digits.reverse()

        return np.array(digits, dtype=np.float32)

    def encode_cnn_onehot(self, game_state: dict) -> np.ndarray:
        return state_to_cnn_features(game_state)



    def state_to_input(self, game_state: dict):
        """
        Convert the state into an input representation suitable for the neural network.

        :param state: The current state.
        :return:      The input representation of the state.
        """
        if game_state is None:
            representation = np.zeros(self.input_shape, dtype=np.float32)
        elif self.board_encoding == self.ENCODING_BINARY:
            representation = self.encode_binary(game_state)
        elif self.board_encoding == self.ENCODING_CNN:
            representation = self.encode_cnn_onehot(game_state)
        else:
            raise ValueError("Unsupported input mode")


        tensor = tf.convert_to_tensor(representation[np.newaxis, :], dtype=tf.float32)
        return tensor

    def validate_training_data(self):
        """
        Validate the integrity of training data, checking for missing iterations and incomplete episodes.
        """
        # Check integrity of training data
        if self.training_data[-1][TransitionFields.DONE] is not True:
            raise ValueError("Last training datum not done")

        # Validate iteration number
        for i in range(len(self.training_data) - 1):
            i1 = self.training_data[i  ][TransitionFields.ITERATION]
            i2 = self.training_data[i+1][TransitionFields.ITERATION]
            if (i1 + 1 != i2):
                raise ValueError(f"Missing iteration between iterations {i1} and {i2} in training data")



    def move_training_data_to_replay_buffer(self):
        """
        Move training data to the replay buffer, connecting states with their subsequent states and converting game dicts to states
        """
        # Connect state with next state and move to replay buffer
        for i in range(len(self.training_data)):
            iteration, state, legal_actions, action, reward, done = self.training_data[i]
            if not done:
                next_state          = self.training_data[i+1][TransitionFields.STATE]
                next_legal_actions  = self.training_data[i+1][TransitionFields.LEGAL_ACTIONS]
            else:
                next_state          = self.state_to_input(None)
                next_legal_actions  = None

            self.replay_sampler.add(self.replay_buffer, [state, legal_actions, action, next_state, next_legal_actions, reward, done])

        self.training_data = []

    def minibatch_to_arrays(self, minibatch):
        """
        Convert a minibatch of experiences into arrays for training.

        :param minibatch: A minibatch of experiences.
        :return:          Tensors containing states, actions, next_states, rewards, and not_terminal flags.
        """
        B = len(minibatch)

        # 1) Build the legal-action and scalar arrays in NumPy
        legal_actions       = np.zeros((B,  self.n_actions  ), dtype=np.float32)
        actions             = np.zeros( B,                     dtype=np.int32  )
        next_legal_actions  = np.zeros((B,  self.n_actions),   dtype=np.float32)
        rewards             = np.zeros( B,                     dtype=np.float32)
        not_terminal        = np.zeros( B,                     dtype=np.float32)

        # 2) Build lists of per-sample state tensors
        state_tensors      = []
        next_state_tensors = []

        for i, (s, l, a, s_, l_, r, d) in enumerate(minibatch):
            legal_actions [i, l]       = 1
            actions       [i]          = a
            next_legal_actions [i, l_] = 1
            rewards       [i]          = r
            not_terminal  [i]          = 1 - d

            state_tensors.append(s)
            next_state_tensors.append(s_ if not d else tf.zeros_like(s)) # Handle s_ for terminal states

        # 3) Concatenate into batch tensors [B, …]
        states      = tf.concat(state_tensors,      axis=0)  # [B, shape...]
        next_states = tf.concat(next_state_tensors, axis=0)  # [B, shape...]

        # return tensors
        return (
            states,
            tf.convert_to_tensor(legal_actions),
            tf.convert_to_tensor(actions),
            next_states,
            tf.convert_to_tensor(next_legal_actions),
            tf.convert_to_tensor(rewards),
            tf.convert_to_tensor(not_terminal)
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, 9, 9, 11], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
        ]
    )
    def _graph_act(self, s, legal_idxs):
        # get q-values
        q = self.online_model(s, training=False)             # shape [1, n_actions]
        # build a mask inside TF
        mask = tf.scatter_nd(
            tf.expand_dims(legal_idxs, 1),                   # [[i0], [i1], …]
            tf.ones_like(legal_idxs, dtype=tf.float32),      # [1,1,…]
            [self.n_actions]                                 # output shape
        )                                                     # shape [n_actions]
        mask = tf.reshape(mask, [1, -1])                     # [1, n_actions]
        # apply mask + LARGE_NEGATIVE_NUMBER trick
        neg_inf = tf.constant(self.LARGE_NEGATIVE_NUMBER, tf.float32)
        masked_q = mask * q + (1 - mask) * neg_inf           # still [1, n_actions]
        # pick best
        return tf.argmax(masked_q, axis=1)[0]                # a scalar tf.Tensor[int32]


    def act(self, game_state : dict, actions):
        """
        Select an action using an epsilon-greedy policy.

        :param state:   The current state.
        :param actions: List of available actions.
        :return:        The selected action.
        """

        # explore
        if np.random.uniform(0, 1) < self.exploration and self.is_training:
            action = np.random.choice(actions)
        # exploit
        else:
            s = self.state_to_input(game_state)
            # one synchronous graph call, no py-side masking or .numpy() inside TF internals:
            action = int(self._graph_act(s, tf.constant(actions, tf.int32)))

        return action


    def update_target_weights(self):
        """
        Update the weights of the target network according to
        target_weight = (1 - tau) * target_weight + tau * online_weight for soft update
        and
        target_weight = online_weight for hard update
        """

        if self.target_update_mode == "soft":
            online_weights = self.online_model.get_weights()
            target_weights = self.target_model.get_weights()

            new_target_weights = [
                (1 - self.target_update_tau) * target_weight + self.target_update_tau * online_weight
                for online_weight, target_weight in zip(online_weights, target_weights)
            ]
            self.target_model.set_weights(new_target_weights)

        else:
            if self.training_round % self.target_update_freq == 0:
                self.target_model.set_weights(self.online_model.get_weights())




    def train(self):
        """
        Train the agent's Q-network using experiences from the replay buffer.
        """

        if len(self.replay_buffer) < self.replay_buffer_min:
            return                          # still warming up

        # Sample a random minibatch from the replay replay_buffer
        for gradient_step in range(self.grad_steps):

            # ───────── 1)  sample  ─────────
            minibatch, idxs, weights = self.replay_sampler.sample(
                self.replay_buffer, self.batch_size
            )

            # unpack as **tensors** (dtype=float32 unless noted)
            (states, legal_s, actions, next_states,
            next_legal_s, rewards, not_terminal) = self.minibatch_to_arrays(minibatch)

            # ───────── 2)  forward pass ─────────
            q_current = self.online_model(states, training=False)              # (B, A)
            q_next_t  = self.target_model(next_states, training=False)         # (B, A), used to evaluate Q_target
            q_next_o  = self.online_model(next_states, training=False) if self.enable_double_dqn else q_next_t # (B, A), used to evaluate best action

            # ───────── 3)  mask + argmax on next-state ─────────
            masked_next = tf.where(next_legal_s > 0, q_next_o, self.LARGE_NEGATIVE_NUMBER)      # (B, A)
            best_actions = tf.argmax(masked_next, axis=1, output_type=tf.int32)  # (B,)

            # if **all** actions were illegal in a row, give q=0
            has_legal = tf.reduce_any(next_legal_s > 0, axis=1)                # (B,)
            batch_ids = tf.range(tf.shape(states)[0], dtype=tf.int32)
            gather_nd = tf.stack([batch_ids, best_actions], axis=1)            # (B,2)

            next_q_max = tf.where(
                has_legal,
                tf.gather_nd(q_next_t, gather_nd),     # Q_target(s′, a*)
                tf.zeros_like(rewards)                 # no legal → 0
            )                                          # shape (B,)

            # ───────── 4)  Bellman target ─────────
            true_targets = rewards + not_terminal * self.discount * next_q_max  # (B,)

            # ───────── 5)  TD-error for PER ─────────)   # |δ
            cur_q_sa   = tf.gather_nd(q_current, tf.stack([batch_ids, actions], axis=1))             # (B,)
            td_errors  = true_targets - cur_q_sa                                                     # (B,)
            self.replay_sampler.update_priorities(
                self.replay_buffer, idxs, td_errors.numpy())   # |δ|+ε recommended

            # ───────── 6)  replace Q(s,a) by target in a tensor-safe way ─────────
            q_target_batch = tf.tensor_scatter_nd_update(
                q_current,                                           # base tensor
                tf.stack([batch_ids, actions], axis=1),              # indices of (s,a)
                true_targets                                         # new values
            )                                                        # still (B, A)

            # ───────── 7)  SGD step ─────────
            sample_weights = tf.convert_to_tensor(weights, dtype=tf.float32)

            loss = self.online_model.train_on_batch(
                states, q_target_batch, sample_weight=sample_weights
            )

            self.training_log.append({
                "training_round": self.training_round,
                "gradient_step": int(gradient_step),
                "loss": float(loss)
            })

            # update target network
            self.update_target_weights()

        self.training_round+= 1


    def save(self, out_dir, base_name="dqn_agent", compressed=True):
        os.makedirs(out_dir, exist_ok=True)
        model_dir = os.path.join(out_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        # Save metadata
        payload = {
            "meta": np.array({
                "learning_rate": self.learning_rate,
                "discount": self.discount,
            }, dtype=object),
        }

        npz_path = os.path.join(out_dir, f"{base_name}.npz")
        if compressed:
            np.savez_compressed(npz_path, **payload)
        else:
            np.savez(npz_path, **payload)

        # Save models
        self.online_model.save(os.path.join(model_dir, f"{base_name}_online_model.keras"))
        self.target_model.save(os.path.join(model_dir, f"{base_name}_target_model.keras"))


        log_path = os.path.join(out_dir, f"{base_name}_training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

        print(f"[SAVE] Agent state saved in {out_dir}")

    def load(self, in_dir, base_name="dqn_agent"):
        model_dir = os.path.join(in_dir, "models")

        # Load metadata
        npz_path = os.path.join(in_dir, f"{base_name}.npz")
        metadata = np.load(npz_path, allow_pickle=True)["meta"].item()
        self.discount = metadata["discount"]

        # Load models
        online_model_path = os.path.join(model_dir, f"{base_name}_online_model.keras")
        target_model_path = os.path.join(model_dir, f"{base_name}_target_model.keras")
        print("Load online model")
        self.online_model = tf.keras.models.load_model(online_model_path, safe_mode=False)
        print("Load target model")
        self.target_model = tf.keras.models.load_model(target_model_path, safe_mode=False)

        log_path = os.path.join(in_dir, f"{base_name}_training_log.json")
        with open(log_path, "r") as f:
            training_log = json.load(f)
        print(f"[LOAD] Agent state loaded from {in_dir}")
