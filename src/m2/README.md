# M2: stage-conditioned hybrid Bomberman agent

## Model and method

M2 is a **stage-conditioned hybrid agent**. Its learned component is a
**Dueling Double Deep Q-Network (Dueling Double DQN)**, not a pure end-to-end
neural policy and not a pure rule-based agent.

The policy combines:

- a convolutional encoder over 10 engineered board channels;
- 32 engineered global/spatial auxiliary features;
- 7 tactical features for each of the 6 actions, added through an
  action-aligned residual head;
- rule-based action proposals and tactical overrides, including coin pursuit,
  conditional crate farming, and pointless-bomb vetoes; and
- exact legal-action and survival filtering before an action is selected.

Task 3a uses a frozen rule-policy proposal followed by exact survival
filtering. Other stages use the neural Q-values together with the frozen
tactical and safety logic. The bundled inference package defaults to Task 4.

## Contents

- `Siegfried/`: the single canonical M2 delivery, containing eight
  self-contained inference files and the stripped inference-only `model.pt`
  checkpoint.
- `requirements.txt`: the pinned runtime dependencies.
- `verify_delivery.py`: the Linux compatibility and inference smoke check.

The delivery intentionally excludes training code, optimizer state, target
network state, replay data, evaluation logs, and M1/M3 artifacts.

Verified checkpoint SHA-256:
`B2148629D65E5C6E5866214884BD015D9ECD2D7F84EA121C425BE0EBD71748D6`.

Source archive (`Siegfried.zip`) SHA-256:
`DE301C0C1BD26C9FA6158A69DC49FA0B4ABA78CF3D41FCA93D14F83952B7592B`.

## Reproducible Debian 13 setup

Create a clean virtual environment from the repository root. Do not reuse an
unversioned or cached PyTorch installation from another image.

```bash
python3 -m venv .m2-venv
. .m2-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --requirement src/m2/requirements.txt
python src/m2/verify_delivery.py
```

The requirements file pins the official CPU build of PyTorch used for the
Linux delivery check. `verify_delivery.py` fails closed if the installed
`libtorch_cpu.so` requests an executable stack, if the source inventory or
checkpoint hash changes, or if the checkpoint cannot be loaded and evaluated
on CPU.

GitLab CI repeats this check in a fresh `debian:13-slim` container whenever
the M2 delivery or its CI configuration changes.
