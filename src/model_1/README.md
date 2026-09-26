# Model 1 — Siegfried

Siegfried is the inference-only Model 1 agent for the Bomberman reinforcement-
learning project. It is a **stage-conditioned hybrid policy**: a trained
**Dueling Double DQN** supplies action values, while deterministic rules,
tactical overrides, and exact survival checks constrain the final action.
It is therefore neither a purely rule-based agent nor a raw-state end-to-end
neural policy.

## Model summary

| Component | Implementation |
| --- | --- |
| Action space | `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, `BOMB` |
| Board input | 10 engineered channels on the 17 × 17 board |
| Auxiliary input | 32 global/spatial features plus 7 tactical features for each action (74 total) |
| Neural network | Convolutional encoder, two-layer 256-unit trunk, dueling value/advantage heads, and an action-aligned tactical residual |
| Training lineage | Double DQN with a dueling architecture and 5-step returns |
| Deployed policy | Neural Q-values combined with stage routing, tactical rules, legal-action masks, and exact survival filtering |
| Parameters | 608,878 |
| Default stage | `task4` |

The checkpoint contains only `format_version` and 19 policy tensors. Optimizer
state, target-network state, replay data, training logs, and training callbacks
are intentionally excluded.

## Decision process

For each game state, Siegfried:

1. builds the 10-channel board tensor and 74 auxiliary features;
2. evaluates physical legality and time-aware bomb survival;
3. applies stage-specific rule routing and tactical overrides;
4. masks unsafe or illegal actions;
5. uses the Dueling Q-network for the remaining action values; and
6. resolves equal best actions with a seeded random generator.

The tactical layer includes coin pursuit, conditional crate farming, opponent
pressure, loop avoidance, trap-aware bombing, and a veto for bombs that would
hit neither a crate nor a nearby opponent.

### Stage routing

| Stage | Runtime behaviour |
| --- | --- |
| `task1` | Disables `BOMB`, applies exact legal/survival filtering, and selects among movement or waiting actions. |
| `task3a` | Starts from the frozen rule-policy proposal. A physically legal `BOMB` proposal is accepted directly; non-bomb proposals must also pass the exact survival check. Invalid proposals fall back to the exact hybrid policy. |
| `task2`, `task3b`, `task4` | Use the exact legal/survival mask, tactical overrides, and neural Q-values. |

`task4` is used when no stage environment variable is supplied.

## Repository contents

| File | Purpose |
| --- | --- |
| [`callbacks.py`](callbacks.py) | Bomberman `setup` and `act` entry points plus stage routing |
| [`config.py`](config.py) | Frozen dimensions, seed, stage names, and runtime environment variables |
| [`features.py`](features.py) | Board channels, auxiliary features, path guidance, and survival features |
| [`model.py`](model.py) | Dueling convolutional Q-network and checkpoint loader |
| [`planner.py`](planner.py) | Exact bomb timing, survival search, tactical actions, and safety masks |
| [`rule_policy.py`](rule_policy.py) | Frozen rule-policy component used by `task3a` |
| `model.pt` | Stripped inference-only checkpoint |
| [`requirements.txt`](requirements.txt) | Pinned CPU runtime dependencies |
| [`verify_delivery.py`](verify_delivery.py) | Fail-closed integrity and inference smoke test |

## Install and verify

From the repository root:

```bash
python3 -m venv .model1-venv
source .model1-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --requirement src/model_1/requirements.txt
python src/model_1/verify_delivery.py
```

On Windows PowerShell, activate the environment with:

```powershell
py -m venv .model1-venv
.\.model1-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --requirement .\src\model_1\requirements.txt
python .\src\model_1\verify_delivery.py
```

A valid run reports `"status": "PASS"`, 8 required source files, 19 policy
tensors, and a Q-output shape of `[1, 6]`. On Linux, the verifier additionally
checks that `libtorch_cpu.so` does not request an executable stack.

## Run in the official framework

Copy the complete `src/model_1` directory into the official framework as
`agent_code/Siegfried`. Keep all files together because the default checkpoint
path is resolved relative to the agent directory.

From the framework directory, a Task 4 CPU run can then be started with:

```bash
M2_DEVICE=cpu M2_TRAINING_STAGE=task4 \
python main.py play \
  --agents Siegfried rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --n-rounds 10 --no-gui
```

For PowerShell:

```powershell
$env:M2_DEVICE = "cpu"
$env:M2_TRAINING_STAGE = "task4"
python .\main.py play --agents Siegfried rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 10 --no-gui
```

Supported stage values are `task1`, `task2`, `task3a`, `task3b`, and `task4`.
`M2_DEVICE` accepts `auto`, `cpu`, or `cuda`. `M2_MODEL_PATH` may point to an
alternative compatible checkpoint; if it is omitted, the bundled `model.pt`
is used. This package is inference-only, so it should not be launched with
`--train 1`.

## Integrity

- Checkpoint SHA-256:
  `B2148629D65E5C6E5866214884BD015D9ECD2D7F84EA121C425BE0EBD71748D6`
- Original `Siegfried.zip` SHA-256:
  `DE301C0C1BD26C9FA6158A69DC49FA0B4ABA78CF3D41FCA93D14F83952B7592B`

The verification script treats any checkpoint-hash mismatch, missing agent
file, incompatible dependency version, non-finite model value, or invalid
forward output as a failure.
