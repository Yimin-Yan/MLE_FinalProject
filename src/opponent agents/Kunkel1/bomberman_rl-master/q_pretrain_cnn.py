#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
import subprocess
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


import q_helpers
from q_deep_agent import DeepQAgent

from q_helpers import get_legal_actions, ACTS, N_ACTIONS, N_STATES, state_to_tabular_features, describe_tabular_state



# --- Argument Parsing for CNN Agent ---
def parse_args():
    p = argparse.ArgumentParser(
        description="Train a CNN-based DeepQAgent from saved transitions."
    )


    # I/O
    p.add_argument("--transitions-dir", "-i", required=True,
                   help="Path(s) to one or more transitions pickle files (list of episodes).")
    p.add_argument("--output-dir", "-o", required=True,
                   help="Path to output .npz file where NNs will be saved.")

    # Training control
    p.add_argument("--n-episode", type=int, default=1000,
                   help="Number of training episodes.")
    p.add_argument("--grad-steps", type=int, default=1000,
                   help="Number of gradient updates per training step.")


    # Evaluation options
    p.add_argument("--evaluate", action="store_true",
                   help="Run evaluation after each training chunk.")
    p.add_argument("--agents", nargs="+", default=["rule_based_agent"] * 4,
                   help="Agent names to use during evaluation.")
    p.add_argument("--scenario", default="classic",
                   help="Scenario to use during evaluation.")
    p.add_argument("--main-py", default="main.py",
                   help="Path to main.py script.")
    p.add_argument("--eval-rounds", type=int, default=100,
                   help="Number of rounds for evaluation.")

    # Core hyperparameters
    p.add_argument("--discount", type=float, default=0.8,
                   help="Discount factor γ.")
    p.add_argument("--learning-rate", "-l", type=float, default=1e-4,
                   help="Learning rate.")
    p.add_argument("--learning-rate-decay", "-d", type=float, default=1.0,
                   help="Learning rate decay factor.")

    # Exploration
    p.add_argument("--exploration", type=float, default=1.0,
                   help="Initial exploration rate (ε).")
    p.add_argument("--exploration-decay", type=float, default=1e-3,
                   help="Exploration rate decay per step.")
    p.add_argument("--exploration-min", type=float, default=0.1,
                   help="Minimum exploration rate.")

    # Replay buffer
    p.add_argument("--replay-buffer-size", type=int, default=1000000,
                   help="Replay buffer capacity.")
    p.add_argument("--replay-buffer-min", type=int, default=100000,
                   help="Minimum buffer size before training starts.")

    # Target network
    p.add_argument("--target-update-tau", type=float, default=0.1,
                   help="Soft update weight for target network.")
    p.add_argument("--target-update-freq", type=int, default=30,
                   help="Hard update frequency (in episodes).")
    p.add_argument("--target-update-mode", choices=["hard", "soft"], default="hard",
                   help="Target network update strategy.")

    # Misc
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for training.")
    p.add_argument("--board-encoding", default="encoding_cnn",
                   help="State encoding method (default CNN).")
    p.add_argument("--debug", action="store_true",
                   help="Enable debug logging.")
    p.add_argument("--plot-debug", action="store_true",
                   help="Plot training results during debug.")

    return p.parse_args()


def evaluate_agent(chunk_idx, out_dir, args):
    """Run main.py play with the trained agent and save stats."""
    dicts_dir = os.path.join(out_dir, "dicts")
    stats_file = os.path.join(dicts_dir, f"eval_chunk_{chunk_idx:04d}.json")

    cmd = [
        "python3", args.main_py,
        "play",
        "--agents", *args.agents,
        "--scenario", args.scenario,
        "--n-rounds", str(args.eval_rounds),
        "--train", "0",
        "--no-gui",
        "--save-stats", stats_file,
        "--match-name", f"chunk_{chunk_idx}"
    ]

    print(f"[Chunk {chunk_idx}] Evaluating agent with {args.eval_rounds} rounds...")
    subprocess.run(cmd, check=True)
    print(f"[Chunk {chunk_idx}] Evaluation stats saved to {stats_file}")

import os, gzip, pickle, gc, glob

def iter_shards(shards_dir):
    # supports both .pkl and .pkl.gz
    files = sorted(glob.glob(os.path.join(shards_dir, "episodes-shard-*.pkl*")))
    for fp in files:
        if fp.endswith(".gz"):
            with gzip.open(fp, "rb") as f:
                yield pickle.load(f)  # list[episode]
        else:
            with open(fp, "rb") as f:
                yield pickle.load(f)

def train_from_shards(args, agent, shards_dir):
    print("Streaming shards & training")
    len_episodes = 0
    agent.start_game(is_training=True)

    out_dir = os.path.dirname(args.output_dir)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    chunk_idx = 0
    shard_i   = 1

    for shard in iter_shards(shards_dir):

        print(f"Ingesting shard {shard_i}")
        # ingest shard
        for episode in shard:
            for (idx, state, legal_actions, action, reward, done, next_state, next_legal_actions) in episode:
                agent.replay_sampler.add(
                    agent.replay_buffer,
                    [state, legal_actions, action, next_state, next_legal_actions, reward, done]
                )
            len_episodes += 1

        print(f"Training {args.grad_steps} iterations")
        agent.train()

        if shard_i % 10 == 0:

            print(f"Saving checkpoint")

            # checkpoints / eval
            agent.save(out_dir, base_name=f"chunk_{chunk_idx:02d}")
            agent.save(out_dir, base_name="default")

            if args.evaluate:
                evaluate_agent(chunk_idx, out_dir, args)

            chunk_idx += 1
        shard_i += 1
        # free memory ASAP
        del shard
        gc.collect()

    print(f"Ingested {len_episodes} episodes from shards in {shards_dir}")


def main():
    args = parse_args()


    # Build config dictionary from args
    config = {
        "n_episode": args.n_episode,
        "grad_steps": args.grad_steps,
        "discount": args.discount,
        "learning_rate": args.learning_rate,
        "learning_rate_decay": args.learning_rate_decay,
        "exploration": args.exploration,
        "exploration_decay": args.exploration_decay,
        "exploration_min": args.exploration_min,
        "batch_size": args.batch_size,
        "board_encoding": args.board_encoding,
        "replay_buffer_size": args.replay_buffer_size,
        "replay_buffer_min": args.replay_buffer_min,
        "target_update_tau": args.target_update_tau,
        "target_update_freq": args.target_update_freq,
        "target_update_mode": args.target_update_mode,
        "debug": args.debug,
        "plot_debug": args.plot_debug,
    }



    agent = DeepQAgent(
            agent_id=0,
            n_actions=N_ACTIONS,
            n_states=N_STATES,
            config=config
    )



    def build_cnn_dqn_model(input_shape, num_actions, lr=1e-4, clipnorm=10.0):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

        x = layers.Flatten()(x)                 # keep spatial info
        x = layers.Dense(128, activation="relu")(x)

        # value
        v = layers.Dense(128, activation="relu")(x)
        v = layers.Dense(1)(v)

        # advantage
        a = layers.Dense(128, activation="relu")(x)
        a = layers.Dense(num_actions)(a)

        # dueling combine (use Lambda to stay on-graph)
        a_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(a)
        q = layers.Add()([v, layers.Subtract()([a, a_mean])])

        model = tf.keras.Model(inputs, q)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm),
            loss=tf.keras.losses.Huber()
        )
        return model

    agent.online_model = build_cnn_dqn_model(agent.input_shape, agent.n_actions)
    agent.online_model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss="mse")
    agent.target_model = build_cnn_dqn_model(agent.input_shape, agent.n_actions)

    train_from_shards(args, agent, shards_dir=args.transitions_dir)




if __name__ == "__main__":
    main()
