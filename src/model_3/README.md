# SEAT-Q experiments for Bomberman

Junfeng Wei's SEAT-Q contribution to the Hildebrandslied final project.
The game framework and supplied opponents come from
[ukoethe/bomberman_rl](https://github.com/ukoethe/bomberman_rl).
This repository covers the SEAT-Q branch; Siegfried and DQFD_RE are separate
team contributions.

SEAT-Q uses compact state features with dueling Double-DQN to learn navigation
and bombing decisions. Its variants examine multi-step learning, temporal
hazard features and auxiliary escape supervision, followed by curriculum and
residual adaptation. League is the retained policy; Guided and Recovery are
separate experiments in teacher supervision and risk-based action correction.

## Report-to-code map

SEAT-Q is the model family name used in the report; `seat_d3qn` is its game
entry point. The `m2_*` names are internal development identifiers used in
source directories, checkpoint metadata and experiment records. They do not
refer to the team's model numbering or the `src/model_3` repository location.

All source paths below are relative to `agent_code/seat_d3qn/variants/`.
Select a variant with `SEAT_D3QN_VARIANT`; Residual and Guided also require
`SEAT_M2_4_BRANCH`.

| Report name | Source directory | `SEAT_D3QN_VARIANT` | `SEAT_M2_4_BRANCH` |
| --- | --- | --- | --- |
| Baseline | `m2_0` | `m2_0` | — |
| MultiStep | `m2_1` | `m2_1` | — |
| Continuation Control | `m2_2_control` | `m2_2_control` | — |
| Temporal | `m2_2` | `m2_2` | — |
| Curriculum | `m2_3` | `m2_3` | — |
| Residual | `m2_4` | `m2_4` | `c1` |
| Guided | `m2_4` | `m2_4` | `c2` |
| League (retained policy) | `m2_4/c3_v2` | `m2_4_c3_v2` | — |
| Recovery (experimental gate) | `m2_5` | `m2_5` | — |

The default selector is `m2_3` (Curriculum). To evaluate League, set
`SEAT_D3QN_VARIANT=m2_4_c3_v2` explicitly. `SEAT_D3QN_MODEL_PATH` selects a
specific checkpoint when needed. Snapshot filenames use the same internal
identifiers; for example, `m2_4_c3_t0225000.pt` is the League checkpoint at
225,000 stage-local transitions. The reported League implementation is
`c3_v2`, even though its checkpoint filenames use the `m2_4_c3` prefix.

## Installation

In the team repository, first run `cd src/model_3`. All commands below run
from this package directory, which contains `main.py`. For a standalone copy,
use the directory containing that file.

Use Python 3.12 (the local verification environment), preferably in a virtual
environment, then run `python -m pip install -r requirements.txt`.
Evaluation uses CPU inference.

## Quick evaluation

Run from the package directory, without `--train`:

```sh
SEAT_D3QN_VARIANT=m2_4_c3_v2 python main.py play --no-gui --agents seat_d3qn peaceful_agent coin_collector_agent rule_based_agent --scenario classic --n-rounds 10 --seed 10001
```

For Residual, use `SEAT_D3QN_VARIANT=m2_4 SEAT_M2_4_BRANCH=c1`.
For Guided, use `SEAT_D3QN_VARIANT=m2_4 SEAT_M2_4_BRANCH=c2`.

Recovery requires an explicit path to its fitted risk checkpoint:

```sh
SEAT_D3QN_VARIANT=m2_5 SEAT_D3QN_MODEL_PATH="$PWD/experiments/seat_d3qn/m2_5/checkpoints/m2_5_risk_t0250000.pt" python main.py play --no-gui --agents seat_d3qn peaceful_agent coin_collector_agent rule_based_agent --scenario classic --n-rounds 10 --seed 61001
```

These commands run short evaluation checks. The report's results use larger,
multi-seed evaluations. The agent appears as `seat_d3qn` in result files.

## Repository structure

- `agent_code/seat_d3qn/`: model definitions, features, training callbacks,
  selection/evaluation scripts, tests and installed weights.
- Frozen/noisy opponent wrappers and official opponents: required by the
  curriculum, league schedule and Recovery data collection.
- `experiments/seat_d3qn/`: checkpoints, validation snapshots, training metrics
  and Recovery dataset shards.
- `results/`: recorded structured training, validation and evaluation results.
- Framework Python files, assets and all supplied agents, including the random
  opponent, deliberate-failure example (`fail_agent`), agent template
  (`tpl_agent`) and keyboard-controlled example (`user_agent`).

The shared M2-4 configuration references `m2_4_historical_agent` as an auxiliary
opponent, not as an additional reported SEAT-Q variant.

League requires the Residual C1 `training-checkpoint.pt`, which supplies its
100,000-transition replay buffer for rehearsal. Recovery requires League's
225k snapshot. These dependencies use the checkpoint paths in the configuration.
Training and selection commands can overwrite artifacts; use a separate copy
when retraining or repeating checkpoint selection.

## Training and checkpoint selection

All module names below start with `agent_code.seat_d3qn.variants.` and are run
from the package directory with `python -m <module>`. Training budgets, source
checkpoints and opponent schedules are defined in each variant's `config.py`.

| Variant | Training entry | Validation / selection entry |
| --- | --- | --- |
| Baseline | Game entry with `SEAT_D3QN_VARIANT=m2_0` and `--train 1` | `m2_0.select_best` |
| MultiStep | Game entry with `SEAT_D3QN_VARIANT=m2_1` and `--train 1` | `m2_1.select_best` |
| Continuation Control | Game entry with `SEAT_D3QN_VARIANT=m2_2_control` and `--train 1` | `m2_2_control.select_best` |
| Temporal | Game entry with `SEAT_D3QN_VARIANT=m2_2` and `--train 1` | `m2_2.select_best` |
| Curriculum | `m2_3.run_curriculum` | `m2_3.select_best` |
| Residual / Guided | `m2_4.run_training`, with `SEAT_M2_4_BRANCH=c1` / `c2` | `m2_4.select_best`, with the same branch |
| League | `m2_4.c3_v2.run_training` | `m2_4.c3_v2.select_best` |
| Recovery | `m2_5.run_training` (data collection and risk fitting) | `m2_5.validate` |

The game entry is `python main.py play --agents seat_d3qn peaceful_agent
coin_collector_agent rule_based_agent --scenario classic --no-gui --train 1
--n-rounds <rounds> --seed 42`, preceded by the appropriate variant selector.
Here `--train 1` trains only the first agent. The round count is a session
length, not the transition budget: Baseline and MultiStep use 500k transitions;
Temporal and Control use 250k additional transitions each.

For fresh training, use a separate working copy with empty output locations
for the target stage, while keeping its source checkpoints and opponent
weights. Check the variant's resume settings first: some trainers reject
existing snapshots, while staged runners resume or skip completed stages.
Running a pipeline in the supplied checkout therefore need not retrain it.

Baseline training and selection share the snapshot directory
`experiments/seat_d3qn/m2_0/checkpoints/snapshots/`; selection writes to
`results/m2_0/validation/`. This checkout contains Baseline's selected weights
and validation results, but not its original candidate snapshot sequence or
full training checkpoint. Repeating the original selection requires those
candidate snapshots; resuming that run requires its full training checkpoint.
The supplied Baseline policy can be evaluated directly using the variant
selector. Its recorded selection results are linked below.

Combined workflows are also available:

- `m2_3.run_experiment`: Curriculum training, selection and 30-seed testing.
- `m2_4.run_c1_c2_pipeline`: Residual and Guided training, selection and comparison.
- `m2_4.c3_v2.run_pipeline`: League training and selection; `--include-test`
  also runs its official-opponent test.
- `m2_5.run_train_validation`: Recovery collection, risk fitting and validation.

## Full evaluation and recorded results

To evaluate the selected League policy over 15 seeds and 100 games per seed:

```sh
python -m agent_code.seat_d3qn.variants.m2_4.c3_v2.run_test --rerun
```

`--rerun` replaces that script's previous test outputs. Without it, valid
cached results are reused when the model hash is unchanged.

Recovery's matched test evaluates both the frozen source and the gated policy
on seeds 61001--61015, with 100 games per seed **per condition**:

```sh
python -m agent_code.seat_d3qn.variants.m2_5.run_official_test
```

This script reuses valid per-seed results. For fresh games, use a separate copy
with an empty `results/m2_5/test/official_paired/` output directory. Curriculum's
`m2_3.run_experiment` uses seeds 8101--8130 with 50 games each by default;
`--rerun-test` reruns its test stage, but the entry also checks training and
selection first.

The following files contain the recorded official-opponent evaluations, not
the short checks in the quick-start section:

| Variant | Recorded result |
| --- | --- |
| Baseline | [1,500-game result](results/m2_0/m2_0_test_1500.json) |
| MultiStep | [1,500-game result](results/m2_1/m2_1_test_1500.json) |
| Continuation Control | [1,500-game result](results/m2_2/control/m2_2_control_test_1500.json) |
| Temporal | [1,500-game result](results/m2_2/m2_2_test_1500.json) |
| Curriculum | [30-seed summary](results/m2_3/m2_3_test_1500_multiseed.json) |
| League | [15-seed summary](results/m2_4/c3_v2/m2_4_c3_test_1500.json) |
| Recovery versus frozen League | [Matched-test summary](results/m2_5/test/m2_5_official_paired_test_1500.json) |

Checkpoint choices are recorded in `validation/selection-summary.json` under
each result directory. Control uses `results/m2_2/control/`; Residual and
Guided use `results/m2_4/c1/` and `results/m2_4/c2/`. Their
[Residual](results/m2_4/c1/validation/selection-summary.json) and
[Guided](results/m2_4/c2/validation/selection-summary.json) summaries contain
the validation comparison. Recovery's
[training summary](experiments/seat_d3qn/m2_5/metrics/training-summary.json)
documents risk-model fitting, and its
[validation summary](results/m2_5/validation/selection-summary.json)
records gate selection.

## Checks

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s agent_code/seat_d3qn/variants -t . -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python -m agent_code.seat_d3qn.variants.m2_4.c3_v2.verify_setup
PYTHONDONTWRITEBYTECODE=1 python -m agent_code.seat_d3qn.variants.m2_5.verify_setup --trained
```
