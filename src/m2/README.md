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
- rule-based action proposals and tactical overrides; and
- exact legal-action and survival filtering before an action is selected.

Task 3a uses a frozen rule-policy proposal followed by exact survival
filtering. Other stages use the neural Q-values together with the frozen
tactical and safety logic. The bundled inference package defaults to Task 4.

## Contents

- `m2_final_submission/`: the eight self-contained inference files, including
  the stripped inference-only `model.pt` checkpoint.
- `package/m2_final_submission.zip`: the verified submission archive created
  before this GitLab upload.
- `package/submission_manifest.json`: file inventory and SHA-256 hashes.

The package intentionally excludes training code, optimizer state, target
network state, replay data, evaluation logs, and M1/M3 artifacts.

Verified archive SHA-256:
`CDE9E97FF929061C0641431FB235809AF4ADF64397DA6C285833FF8AC0AB487A`.

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
`libtorch_cpu.so` requests an executable stack, if package hashes change, or
if the checkpoint cannot be loaded and evaluated on CPU.

GitLab CI repeats this check in a fresh `debian:13-slim` container whenever
the M2 delivery or its CI configuration changes.
