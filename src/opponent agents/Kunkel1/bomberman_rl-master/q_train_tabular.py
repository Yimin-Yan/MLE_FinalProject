#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
import subprocess
import os

import q_helpers
import q_tabular_agent

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a TabularQAgent from new transitions and dump Q-tables."
    )

    # I/O
    p.add_argument("--output-q-file", "-o", required=True,
                   help="Path to output .npz file where q and q_visits will be saved.")

    # Hyperparameters
    p.add_argument("--discount", type=float, default=0.8, help="Discount factor γ")
    p.add_argument("--learning-rate-mode", "-m", choices=["fixed", "adaptive"], default="adaptive")
    p.add_argument("--learning-rate", "-l", type=float, default=1e-0)
    p.add_argument("--learning-rate-decay", "-d", type=float, default=0.9999)
    p.add_argument("--initial-q", "-q", type=float, default=0.0)
    p.add_argument("--num-chunks", type=int, default=10, help="Number of training chunks to run.")
    p.add_argument("--debug", action="store_true")

    # Agent config (additional)
    p.add_argument("--train-freq", type=int, default=1, help="Training frequency.")
    p.add_argument("--exploration", type=float, default=0.1)
    p.add_argument("--exploration-decay", type=float, default=0.0)
    p.add_argument("--exploration-min", type=float, default=0.1)

    # Evaluation options
    p.add_argument("--evaluate", action="store_true",
                   help="Run evaluation after each training chunk.")
    p.add_argument("--agents", nargs="+", default=["rule_based_agent"] * 4,
                   help="Agent names to use during evaluation.")
    p.add_argument("--scenario", default="classic",
                   help="Scenario to use during evaluation.")
    p.add_argument("--main-py", default="main.py",
                   help="Path to main.py script.")
    p.add_argument("--eval-rounds", type=int, default=10,
                   help="Number of rounds for evaluation.")
    p.add_argument("--train-rounds", type=int, default=100,
                   help="Number of rounds for training")

    return p.parse_args()


def save_snapshot(agent, out_dir, base_name, chunk_idx):
    """Save intermediate .npz and pickle snapshots."""
    np.savez(
        os.path.join(out_dir, base_name + ".npz"),
        q=agent.q,
        q_visits=agent.q_visits
    )

    dicts_dir = os.path.join(out_dir, "dicts")
    os.makedirs(dicts_dir, exist_ok=True)

    with open(os.path.join(dicts_dir, f"q_chunk_{chunk_idx:04d}.pkl"), "wb") as f_q:
        pickle.dump(agent.q, f_q)
    with open(os.path.join(dicts_dir, f"q_visits_chunk_{chunk_idx:04d}.pkl"), "wb") as f_qv:
        pickle.dump(agent.q_visits, f_qv)

    print(f"[Chunk {chunk_idx}] Saved snapshot to {out_dir}")


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



def train_agent(chunk_idx, out_dir, args):
    """Run main.py play with the trained agent and save stats."""
    dicts_dir = os.path.join(out_dir, "dicts")
    stats_file = os.path.join(dicts_dir, f"eval_chunk_{chunk_idx:04d}.json")

    cmd = [
        "python3", args.main_py,
        "play",
        "--agents", *args.agents,
        "--scenario", args.scenario,
        "--n-rounds", str(args.train_rounds),
        "--train", "1",
        "--no-gui",
        "--save-stats", stats_file,
        "--match-name", f"chunk_{chunk_idx}"
    ]

    print(f"[Chunk {chunk_idx}] Training agent with {args.eval_rounds} rounds...")
    subprocess.run(cmd, check=True)
    print(f"[Chunk {chunk_idx}] Evaluation stats saved to {stats_file}")



def main():
    args = parse_args()

    config = {
        "discount": args.discount,
        "learning_rate_mode": args.learning_rate_mode,
        "learning_rate": args.learning_rate,
        "learning_rate_decay": args.learning_rate_decay,
        "initial_q": args.initial_q,
        "debug": args.debug,
        "train_freq": args.train_freq,
        "exploration": args.exploration,
        "exploration_decay": args.exploration_decay,
        "exploration_min": args.exploration_min,
    }

    agent = q_tabular_agent.TabularQAgent(
        agent_id=0,
        n_actions=q_helpers.N_ACTIONS,
        n_states=q_helpers.N_STATES,
        config=config,
    )

    agent.start_game(is_training=True)

    out_dir = os.path.dirname(args.output_q_file)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.output_q_file))[0]

    chunk_idx = 1

    print("Starting free training")
    while chunk_idx <= args.num_chunks:
        print(f"Visited {len(agent.q)} states")

        train_agent(chunk_idx, out_dir, args)

        save_snapshot(agent, out_dir, base_name, chunk_idx)

        if args.evaluate:
            evaluate_agent(chunk_idx, out_dir, args)

        chunk_idx += 1


    print("Training complete.")


if __name__ == "__main__":
    main()
