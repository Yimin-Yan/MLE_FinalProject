# Reinforcement Learning for Bomberman

Course project for **Machine Learning Essentials**, Heidelberg University, Summer Semester 2026.

This repository contains code developed for the course's Bomberman reinforcement-learning project. The game is played by up to four agents on a discrete grid. An agent can move, wait, or place a bomb; it must collect coins, clear crates, eliminate opponents, and avoid destroying itself.

The current public checkout contains the packaged inference version of **Siegfried** in [`src/model_1`](src/model_1). Siegfried combines a trained Dueling Double DQN with engineered state features, deterministic tactical rules, legal-action masking, and time-aware survival checks.

## Course objectives

The project divides the learning problem into four progressively harder tasks:

1. collect visible coins efficiently without crates or opponents;
2. destroy crates, reveal hidden coins, and escape from bombs safely;
3. hunt the provided peaceful and coin-collecting agents; and
4. compete for score against strong agents in the full game.

The course emphasizes systematic model design, controlled experiments, meaningful performance metrics, and comparisons between model variants and provided baselines.

## Current repository status

| Path | Status |
| --- | --- |
| [`src/model_1/`](src/model_1/) | Complete Siegfried inference package, including trained weights and a delivery verifier |
| `src/model_2/` | Placeholder; no model implementation is currently published here |
| `src/model_3/` | Placeholder; no model implementation is currently published here |
| `src/opponent agents/` | External opponent material used during development; it is not part of Siegfried |

The course report is submitted separately, as required by the project instructions.

## Siegfried at a glance

| Component | Implementation |
| --- | --- |
| Actions | `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, `BOMB` |
| Board representation | 10 engineered channels on the 17 x 17 board |
| Auxiliary representation | 32 global/spatial features and 7 tactical features per action, 74 values in total |
| Q-network | Convolutional encoder, two 256-unit hidden layers, and dueling value/advantage heads |
| Additional network component | Action-aligned tactical residual |
| Training lineage | Double DQN with a dueling architecture and 5-step returns |
| Runtime policy | Neural Q-values with stage routing, tactical overrides, action masks, and exact survival filtering |
| Parameters | 608,878 |
| Default runtime stage | `task4` |

For a detailed description of the feature representation, decision process, stage routing, and individual files, see [`src/model_1/README.md`](src/model_1/README.md).

## Requirements

The packaged agent uses the following additional libraries:

- NumPy 2.5.3
- PyTorch 2.11.0, CPU build

The Bomberman environment itself is provided by the [official course framework](https://github.com/ukoethe/bomberman_rl). Its dependencies must also be installed according to that repository's instructions.

## Install and verify Siegfried

Clone this repository and create an isolated environment:

```bash
git clone https://github.com/Yimin-Yan/MLE_FinalProject.git
cd MLE_FinalProject
python3 -m venv .model1-venv
source .model1-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --requirement src/model_1/requirements.txt
python src/model_1/verify_delivery.py
```

On Windows PowerShell:

```powershell
git clone https://github.com/Yimin-Yan/MLE_FinalProject.git
Set-Location MLE_FinalProject
py -m venv .model1-venv
.\.model1-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --requirement .\src\model_1\requirements.txt
python .\src\model_1\verify_delivery.py
```

A successful verification prints `"status": "PASS"`. The verifier checks:

- the required source files;
- the SHA-256 digest of the delivered checkpoint;
- the expected NumPy and PyTorch versions;
- the checkpoint structure and its 19 policy tensors;
- finite network parameters; and
- a CPU forward pass with output shape `[1, 6]`.

This verifies package integrity and inference compatibility. It does not constitute an evaluation of game performance.

## Run in the official framework

Copy the complete [`src/model_1`](src/model_1/) directory into the official framework as:

```text
bomberman_rl/
└── agent_code/
    └── Siegfried/
```

Keep `model.pt` in the same directory as the Python modules. From the official framework directory, run a CPU game with:

```bash
M2_DEVICE=cpu M2_TRAINING_STAGE=task4 \
python main.py play \
  --agents Siegfried rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --n-rounds 10 --no-gui
```

Windows PowerShell:

```powershell
$env:M2_DEVICE = "cpu"
$env:M2_TRAINING_STAGE = "task4"
python .\main.py play --agents Siegfried rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 10 --no-gui
```

Supported runtime stages are `task1`, `task2`, `task3a`, `task3b`, and `task4`. If `M2_TRAINING_STAGE` is not set, the agent uses `task4`. The `M2_` prefix is retained from the agent's internal development lineage.

## Reproducibility boundary

The published `src/model_1` package is an inference delivery. It contains the trained policy tensors required to run Siegfried, but it does not contain optimizer state, target-network state, replay buffers, training logs, or training callbacks.

Any reported performance result should identify the scenario, opponent lineup, number of rounds, random seeds, metrics, checkpoint hash, and evaluation script. The delivery verifier alone must not be interpreted as evidence that a course task or performance threshold was passed.

## Attribution

The game environment and provided baseline agents come from the [Heidelberg Bomberman RL framework](https://github.com/ukoethe/bomberman_rl). Material under `src/opponent agents/` belongs to its respective external authors and is separate from the Siegfried implementation.
