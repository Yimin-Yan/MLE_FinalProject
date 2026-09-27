# Q‑Learning with **BombeRLe**

This repository implements both **tabular Q‑learning** and a **Deep Q‑Network (DQN)** for the classic game *Bomberman* using the BombeRLe environment. The code is explained in the blog posts on [tabular Q-learning](https://kunkelalexander.github.io/blog/computers-learning-bomberman-tabular-q/) and [deep Q-learning](https://kunkelalexander.github.io/blog/computers-learning-bomberman-deep-q/).
<img src="figures/1_gameplay.gif" width=60%>

## Table of contents

* [Project structure](#project-structure)
* [Agents](#agents)
  * [Coingrabber](#coingrabber)
  * [Crate‑hero](#crate-hero)
  * [Allstar](#allstar)
* [Training / code layout](#training--code-layout)
* [Tips & notes](#tips--notes)
* [License](#license)


## Project structure

All **new agents** live under the `agent_code/` directory. A typical layout looks like this (truncated):

```text
.
├── post_1.ipynb                       # Generate figures for blog post on tabular Q-learning
├── post_2.ipynb                       # Generate figures for blog post on deep Q-learning
├── agent_code/
│   ├── peaceful_agent/
│   ├── rule_based_agent/
│   ├── tq_demonstrator/
│   ├── tq_coingrabber/
│   ├── tq_cratehero/
│   ├── tq_allstar/
│   └── cnn_allstar/
├── q_agent_parent.py
├── q_build_training_episodes.py
├── q_deep_agent.py                     # Core deep Q-learning implementation
├── q_optuna.py
├── q_pretrain_tabular.py
├── q_prioritised_experience_replay.py
├── q_tabular_agent.py                  # Core tabular Q-learning implementation
├── q_train_tabular.py
└── figures/
```

> **Heads‑up:** paths below assume you run commands from the repo root.

- `environment.py` was also updated to allow debugging the Q-table

<img src="figures/3_debug_state.gif" width=60%>


## Agents

* **cnn\_allstar** – CNN DQN-agent trained on multi‑opponent games.
* **tq\_coingrabber** – tabular Q-agent trained for coin‑heaven scenario.
* **tq\_cratehero** – tabular Q-agent trained for loot‑crate scenario.
* **tq\_allstar** – tabular Q-agent trained on multi‑opponent games.
* **tq\_demonstrator** – helper agent based on `rule_based_agent` that **stores transitions** from every time‑step while it plays. It is used to generate supervised/offline training data for the tabular and DQN pipelines. It creates a unique run folder (e.g. `runs/run_YYYYMMDD_HHMMSS_<id>/`) and appends all game transitions into a compressed pickle stream `transitions_all_games.pkl.gz`. In contrast to the rule-based agent, it also makes random moves in the beginning to let the agent explore more.

## Usage

Below are some minimal, end‑to‑end scripts to **reproduce the results** shown in the blog post. Adjust paths if your run folders differ.

### Tabular Q-Coingrabber

<img src="figures/5_coin_grabber.gif" width=60%>

1. **Generate and store raw game data** (game state dictionary at every time‑step):

```bash
python3 main.py play \
  --agents tq_demonstrator \
  --train 1 --n-rounds 50000 \
  --scenario coin-heaven --no-gui
```

2. **Convert the raw game data to training data** (use state representation and rewards to create tuples tuples of `(states, actions, rewards)`):

```bash
python3 q_build_training_episodes.py agent_code/tq_demonstrator/runs/coin_heaven_50k/
```

3. **Pretrain the agent** on all episodes in 10 batches:

```bash
python3 q_pretrain_tabular.py \
  -i agent_code/tq_demonstrator/runs/coin_heaven_50k/transitions.pkl \
  --training-episodes 5000 --num-chunks 10 \
  -o agent_code/tq_coingrabber/q_table.npz \
  --evaluate --agents tq_coingrabber --scenario coin-heaven
```

4. **Test the agent**:

```bash
python3 main.py play --agents tq_coingrabber --scenario coin-heaven
```


### Tabular Q-Crate‑hero

<img src="figures/6_crate_hero.gif" width=60%>

1. **Generate and store raw game data**:

```bash
python3 main.py play \
  --agents tq_demonstrator \
  --train 1 --n-rounds 50000 \
  --scenario loot-crate --no-gui
```

2. **Convert the training data**:

```bash
python3 q_build_training_episodes.py agent_code/tq_demonstrator/runs/loot_crate_50k/
```

3. **Pretrain the agent**:

```bash
python3 q_pretrain_tabular.py \
  -i agent_code/tq_demonstrator/runs/loot_crate_50k/transitions.pkl \
  --training-episodes 5000 --num-chunks 10 \
  -o agent_code/tq_cratehero/q_table.npz \
  --evaluate --agents tq_cratehero --scenario loot-crate
```

4. **Test the agent**:

```bash
python3 main.py play --agents tq_cratehero --scenario loot-crate
```

### Tabular Q-Allstar

<img src="figures/7_allstar.gif" width=60% class="center">

1. **Generate and store raw game data** (three opponents: peaceful + two rule‑based):

```bash
python3 main.py play \
  --agents tq_demonstrator peaceful_agent rule_based_agent rule_based_agent \
  --train 1 --n-rounds 50000 --no-gui
```

2. **Convert the training data**:

```bash
python3 q_build_training_episodes.py agent_code/tq_demonstrator/runs/three_rule_based_peaceful_50k/
```

3. **Pretrain the agent** (single dataset):

```bash
python3 q_pretrain_tabular.py \
  --transitions-file agent_code/tq_demonstrator/runs/three_rule_based_peaceful_50k/transitions.pkl \
  -o agent_code/tq_allstar/q_table.npz \
  --training-episodes 5000 --num-chunks 10 \
  --evaluate --agents tq_allstar peaceful_agent rule_based_agent rule_based_agent
```

4. **Test the agent**:

```bash
python3 main.py play --agents tq_allstar peaceful_agent rule_based_agent rule_based_agent
```



### CNN Allstar

<img src="figures/16_cnn_allstar.gif" width=60%>

1. **Generate and store raw game data** (game state dictionary at every time‑step):

```bash
python3 main.py play \
  --agents tq_demonstrator \
  --train 1 --n-rounds 50000 \
  --scenario coin-heaven --no-gui
```

2. **Convert the raw game data to training data** (use state representation and rewards to create tuples tuples of `(states, actions, rewards)`):

```bash
python3 q_build_training_episodes.py --cnn agent_code/tq_demonstrator/runs/coin_heaven_50k/
```

3. **Pretrain the agent** on all episodes in 10 batches:

```bash
python3 q_pretrain_cnn.py \
  -i agent_code/tq_demonstrator/runs/coin_heaven_50k/transitions_cnn.pkl \
  --training-episodes 5000 --num-chunks 10 \
  -o agent_code/tq_coingrabber/q_table.npz \
  --evaluate --agents tq_coingrabber --scenario coin-heaven
```

4. **Test the agent**:

```bash
python3 main.py play --agents tq_coingrabber --scenario coin-heaven
```

---

### CNN Q-Crate‑hero

<img src="figures/6_crate_hero.gif" width=60%>

1. **Train the agent**:

```bash
python3 main.py play --agents cnn_allstar rule_based_agent peaceful_agent tq_representator --train 1 --n-rounds 100000 --no-gui
```

2. **Test the agent**:

```bash
python3 main.py play --agents cnn_allstar rule_based_agent peaceful_agent tq_representator
```

### Profiling

```bash
py-spy record --rate 10 -o pyspy.svg --function -- python3 q_pretrain_cnn.py -i agent_code/tq_demonstrator/runs/three_rule_based_peaceful_50k/ -o agent_code/cnn_allstar/snapshots/ --evaluate --agents cnn_allstar peaceful_agent rule_based_agent rule_based_agent
```

### Performance

```bash
python3 main.py play --agents cnn_allstar_duel peaceful_agent rule_based_agent tq_representator --n-rounds 1000 --save-stats --no-gui
```

## Training / code layout

The Q‑learning training and utilities are split across the following files:

* `q_tabular_agent.py` – **Tabular Q‑learning core** (Q‑table updates, ε‑greedy policy, etc.).
* `q_helpers.py` – **Q‑learning helpers** (define state representation, set rewards, etc.).
* `q_train_tabular.py` – Training loop for the tabular agent.
* `q_pretrain_tabular.py` – Offline pre‑training from recorded transitions / episodes.
* `q_deep_agent.py` – DQN implementation.
* `q_prioritised_experience_replay.py` – PER buffer used by DQN.
* `q_agent_parent.py` – Shared functionality / base class for Q‑agents.
* `q_build_training_episodes.py` – Converts raw game dictionaries into `(state, action, reward)` tuples.
* `q_optuna.py` – Hyper‑parameter tuning utilities.

## Tips & notes

* **Run folders:** `tq_demonstrator` writes to `runs/<auto‑generated>/`. You can symlink or copy to scenario‑named folders (e.g. `coin_heaven_50k/`) to keep things tidy.
* **Decaying exploration:** The demonstrator uses ε‑greedy exploration with decay to enrich the dataset at the start of training.
* **Scenarios:** Use `--scenario coin-heaven` or `--scenario loot-crate` to reproduce the experiments.
* **GPU/CPU:** DQN code will use your default device; check `q_deep_agent.py` for toggles.

## License

![MIT License](LICENSE)