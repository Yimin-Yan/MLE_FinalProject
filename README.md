# Bomberman reinforcement learning

This repository contains the code submitted for the Machine Learning Essentials
Bomberman project at Heidelberg University in the 2026 summer semester.

The inventory below reflects the files tracked on the `main` branch. It
includes one runnable project model: Siegfried in
[`src/model_1`](src/model_1/). The other numbered model directories are empty
placeholders. Code under [`src/opponent agents`](src/opponent%20agents/) comes
from external projects used as development references or opponents.

## Model inventory

| Path | Contents | Status |
| --- | --- | --- |
| [`src/model_1/`](src/model_1/) | Siegfried source code, trained checkpoint, dependencies, documentation, and verifier | Complete inference package |
| [`src/model_2/`](src/model_2/) | `.gitkeep` only | No model implementation |
| [`src/model_3/`](src/model_3/) | `.gitkeep` only | No model implementation |
| [`src/opponent agents/`](src/opponent%20agents/) | External Bomberman agents and reference repositories | Separate from the submitted models |

Model 2 and Model 3 contain no Python source, checkpoint, dependency file, or
model documentation. They should not be treated as runnable agents.

## Model 1: Siegfried

Siegfried is a stage-conditioned hybrid agent. A Dueling Double DQN estimates
action values from engineered board and tactical features. Stage routing,
rule-policy proposals, tactical overrides, legal-action masks, and time-aware
survival checks determine which actions may be used.

| Component | Implementation |
| --- | --- |
| Actions | `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, `BOMB` |
| Board input | 10 channels on a 17 x 17 board |
| Auxiliary input | 32 base features and 7 tactical features per action, 74 values in total |
| Q-network | Convolutional encoder, two 256-unit hidden layers, dueling value and advantage heads, and an action-aligned tactical residual |
| Training lineage | Double DQN with a dueling architecture and 5-step returns |
| Runtime policy | Neural Q-values combined with stage routing, rules, masks, and survival filtering |
| Parameters | 608,878 |
| Default stage | `task4` |

The delivered checkpoint contains 19 policy tensors and has this SHA-256
digest:

```text
B2148629D65E5C6E5866214884BD015D9ECD2D7F84EA121C425BE0EBD71748D6
```

The package is intended for inference. It has `setup` and `act` entry points,
but no training callbacks, optimizer state, target-network state, replay
buffer, or training logs. See
[`src/model_1/README.md`](src/model_1/README.md) for the file-level description
and stage behavior.

## Install and verify

The pinned runtime dependencies are NumPy 2.5.3 and the CPU build of PyTorch
2.11.0. On Linux:

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

A valid package prints `"status": "PASS"`. The verifier checks the required
files, checkpoint digest, dependency versions, checkpoint structure, finite
parameters, and a CPU forward pass with output shape `[1, 6]`. This is an
integrity and compatibility check; it does not measure game performance.

## Run Siegfried

Install the [official Bomberman RL framework](https://github.com/ukoethe/bomberman_rl),
then copy the complete `src/model_1` directory into the framework as
`agent_code/Siegfried`. Keep `model.pt` beside the Python modules.

From the framework directory, run a Task 4 game on CPU:

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

Supported stages are `task1`, `task2`, `task3a`, `task3b`, and `task4`. The
agent uses `task4` when `M2_TRAINING_STAGE` is unset. `M2_MODEL_PATH` can point
to another compatible checkpoint. The `M2_` prefix remains from the internal
development name.

Do not start this package with `--train 1`; the published files do not include
the training implementation.

## Evidence boundary

This repository does not publish evaluation tables, random-seed records, or
training logs. The package verifier therefore supports claims about file
integrity and inference compatibility only. Performance claims require the
scenario, opponents, round count, seeds, metrics, checkpoint digest, and
evaluation command used to produce them.

## External code

The game environment and provided baselines are maintained in the
[Heidelberg Bomberman RL framework](https://github.com/ukoethe/bomberman_rl).
The material collected under `src/opponent agents/` comes from external
sources. It is not required to verify or run Siegfried. Review the README and
license included with each source before reuse.
