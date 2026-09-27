#!/usr/bin/env python3
import optuna
import subprocess
import os
import json
import pickle
import shutil
import tempfile
import numpy as np

DICT_FOLDER = "optuna"
MAIN_SCRIPT = "q_pretrain_tabular.py"

def run_training(params):
    output_q_file = "agent_code/tq_optuna/q_table.npz"

    cmd = [
        "python3", MAIN_SCRIPT,
        "--transitions-file", "agent_code/tq_demonstrator/runs/three_rule_based_peaceful_50k_18bit/transitions.pkl",  # Change as needed
        "--output-q-file", output_q_file,
        "--training-episodes", "50000",
        "--num-chunks", "1",
        "--evaluate",
        "--agents", "tq_optuna", "rule_based_agent", "rule_based_agent", "peaceful_agent",
        "--scenario", "classic",
        "--eval-rounds", "30",
        "--initial-q", str(params["initial_q"]),
        "--learning-rate", str(params["learning_rate"]),
        "--learning-rate-mode", params["learning_rate_mode"],
        "--discount", str(params["discount"])
    ]

    subprocess.run(cmd, check=True)
    return

def evaluate_last_json(dict_folder, agent_name="tq_optuna"):
    json_files = sorted(
        [f for f in os.listdir(dict_folder) if f.startswith("eval_chunk_") and f.endswith(".json")]
    )
    if not json_files:
        raise FileNotFoundError("No evaluation JSONs found.")

    last_json_path = os.path.join(dict_folder, json_files[-1])
    with open(last_json_path, "r") as f:
        data = json.load(f)

    agent_data = data["by_agent"].get(agent_name)
    if agent_data is None:
        raise ValueError(f"Agent '{agent_name}' not found in {last_json_path}")


    score = agent_data.get("score", 0)
    rounds = agent_data.get("rounds", 1)
    avg_score = score / rounds
    return avg_score


def objective(trial):
    # Define search space
    params = {
        "initial_q": trial.suggest_float("initial_q", -1.0, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "learning_rate_mode": trial.suggest_categorical("learning_rate_mode", ["fixed", "adaptive"]),
        "discount": trial.suggest_float("discount", 0.5, 0.99)
    }

    try:
        dict_folder = run_training(params)
        score = evaluate_last_json("agent_code/tq_optuna/dicts")
        return score
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float("-inf")

def main():
    os.makedirs(DICT_FOLDER, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    print(study.best_trial)

    print("\nBest params:")
    for key, val in study.best_trial.params.items():
        print(f"{key}: {val}")
    import pickle

    with open("optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

if __name__ == "__main__":
    main()
