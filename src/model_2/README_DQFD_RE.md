# DQFD_RE branch

This part of the repository contains every agent developed in the DQFD_RE
branch of the report (Sections 4.4, 5.3 and 6.3), from the first tabular
Q-learner to the demonstration-based DQN that took part in the mixed matches.
All agents live in `agent_code/` and can be run with the unmodified course
framework.

**Requirements:** Python 3.11, `numpy`, `torch` (all neural agents). Tabular
agents need `numpy` only.

---

## Which directory is which model

The tables below follow the order of the report. Directories not listed here
are scratch copies and were not used for any reported result.

### Tabular agents (Sec. 6.3.1–6.3.2)

The tabular agent was developed in one directory; each version's Q-table was
saved to `experiments/` when the next version started.

| Directory / file | Model in the report |
|---|---|
| `agent_code/my_agent/` | Tabular Q-learning, final version (v7, compressed features) |
| `experiments/backup_v1_task1_qtable.pt` | Task 1, 5 features |
| `experiments/v4_task2_coin30.pt` | Task 2, v4 |
| `experiments/v5_lastdir.pt` | Task 2, v5 (previous-move feature) |
| `experiments/v6_opponents.pt` | Opponents, uncompressed (v6) |
| `experiments/v7_compressed.pt` | Opponents, compressed (v7) |
| `agent_code/my_agent_sarsa/` | SARSA |
| `agent_code/my_agent_esarsa/` | Expected SARSA |
| `agent_code/my_agent_esarsa_eps/` | Expected SARSA, ε_end = 0.2 |

To run an older version, copy its file into `agent_code/my_agent/` and rename
it to the model file name used in `callbacks.py`.

### DQN and Rainbow components (Sec. 6.3.3)

| Directory | Model in the report |
|---|---|
| `my_agent_dqn/` | DQN v1–v3full (versions in `experiments/dqn_*.pt`) |
| `my_agent_dqn_nstep/` | n-step DQN, n = 4 |
| `my_agent_dqn_n2/`, `my_agent_dqn_n8/` | n = 2, n = 8 |
| `my_agent_dqn_double/` | Double DQN |
| `my_agent_dqn_per/` | Prioritized replay |

### Attack features and symmetry augmentation (Sec. 6.3.4)

| Directory | Model in the report |
|---|---|
| `my_agent_dqn_attack/` | + attack features |
| `my_agent_dqn_attack2/` | + two further opponent features |
| `my_agent_dqn_attack_n8/` | + attack features, n = 8 |
| `my_agent_dqn_attack_n8_dueling/` | + attack features, n = 8, dueling head |
| `my_agent_dqn_attack_sym/` | + attack features + symmetry |
| `my_agent_dqn_attack_sym_n8/` | + attack features + symmetry, n = 8 |

### Learning from demonstrations (Sec. 6.3.5)

| Directory | Model in the report |
|---|---|
| `demo_collector/` | Records `rule_based_agent` transitions in our feature space |
| `my_agent_dqn_dqfd/` | DQfD, λ_E = 0.1, ratio 0 |
| `my_agent_dqn_dqfd_ratio010/` | DQfD, ratio 0.10 |
| `my_agent_dqn_dqfd_ratio025/` | DQfD, ratio 0.25 |
| `dqfd_re` | `dqfd_re`, the variant used in the mixed matches (Sec. 6.4) |

### Parameter sweeps (Sec. 6.3.7)

All sweeps start from `my_agent_dqn_dqfd` and change one parameter; 30k
training rounds, a snapshot every 5k rounds (`dqn-model_snap*.pt`).

| Directory | Change |
|---|---|
| `my_agent_coin30/` | control (coin reward 3.0) |
| `my_agent_coin50/`, `my_agent_coin80/` | coin reward 5.0, 8.0 |
| `my_agent_dqfd_c03/`, `my_agent_dqfd_c10/` | crate reward 0.3, 1.0 |
| `my_agent_kill10/` | suicide penalty −10 |
| `my_agent_gotkilled8/` | killed penalty −8 |
| `my_agent_stayed12/` | stayed-in-danger penalty −1.2 |
| `my_agent_gamma99/` | γ = 0.99 |
| `my_agent_combo/` | coin 5.0 + crate 0.3 |

### Unsuccessful attempts (Sec. 6.3.8)

| Directory | Attempt |
|---|---|
| `my_agent_reward_align/`, `my_agent_symn8_align/` | Rewards aligned with the official score, no shaping |
| `my_agent_symn8_safe/` | Heavier death penalties + stronger training opponents |
| `my_agent_dqn_diverse/`, `my_agent_dqn_diverse_ft/` | Training against our own earlier agents |
| `my_agent_dqn_cnn/`, `demo_collector_grid/` | Grid input + CNN (not completed) |

---

## Not included

- **`demonstrations.pkl`** (up to ~600 MB). Regenerate with
  ```
  python main.py play --no-gui --agents demo_collector rule_based_agent rule_based_agent rule_based_agent --train 1 --n-rounds 500
  ```
  and copy the file into the DQfD agent's directory. Use `demo_collector_grid`
  for the grid version.
- **Log files** (`logs/`) and per-round **training logs** (`training_stats.csv`).

---

## Reproducing the results

**Train** (from the repository root):
```
python main.py play --no-gui --agents <directory> rule_based_agent rule_based_agent rule_based_agent --train 1 --n-rounds <rounds>
```
Delete `dqn-model.pt` first to train from scratch; otherwise training continues
from the saved model.

**Evaluate** (Sec. 4.5): set `TRAIN_MODE = False` in the agent's `train.py`
(tabular agents: `ALPHA = 0`, `EPS_START = EPS_END = 0`), then run the same
command with `--n-rounds 200`. Learning, exploration and saving are switched
off; the per-round results are written to `training_stats.csv`.

**Mixed matches** (Sec. 6.4): set `LOG_GAME = logging.INFO` in `settings.py`,
play without `--train`, then summarise every agent from the game log:
```
python main.py play --no-gui --agents <agent1> <agent2> <agent3> <agent4> --n-rounds 100
python analyze_game_log.py logs/game.log
```

**Figures:** `python make_figures_en.py` reads the evaluation files in
`experiments/` and writes the report figures to `figures/`.

---

## Use of AI tools

TODO: state honestly how AI tools were used in this branch (for example:
drafting code, automating chains of training and evaluation runs, drafting
and editing report text) and what was done by the author (experiment design,
running and checking every experiment against the raw logs, analysis and
conclusions). Keep this consistent with the statement in the report.
