"""Check C3-v2 sources, transfer identity, replay and one in-memory update."""

from types import SimpleNamespace

import numpy as np
import torch

from ...m2_3.replay_buffer import LAPReplayBuffer
from . import callbacks, train
from .checks import verify_all
from .config import (
    ACTIONS,
    ADAM_BETAS,
    ADAM_EPS,
    FEATURE_DIM,
    LAP_ALPHA,
    LAP_PRIORITY_FLOOR,
    LEARNING_RATE,
    REPLAY_CAPACITY,
)
from .model import C3ResidualAdapter


def _silent(*_args, **_kwargs):
    return None


def main():
    verification = verify_all()
    agent = SimpleNamespace(
        train=True,
        logger=SimpleNamespace(info=_silent, error=_silent),
    )
    callbacks.setup(agent)

    probe = torch.linspace(-1.0, 1.0, steps=8 * FEATURE_DIM).reshape(8, FEATURE_DIM)
    with torch.no_grad():
        c1_values = agent.base_net(probe) + agent.c1_residual_net(probe)
        c3_values = c1_values + agent.adapter_net(probe)
        max_error = float((c3_values - c1_values).abs().max().item())
    if max_error > 1e-6:
        raise RuntimeError("C3-v2 t0 changed C1 by {:.3e}".format(max_error))
    if any(parameter.requires_grad for parameter in agent.base_net.parameters()):
        raise RuntimeError("The M2-3 base is not frozen")
    if any(parameter.requires_grad for parameter in agent.c1_residual_net.parameters()):
        raise RuntimeError("The C1 residual is not frozen")
    if not all(parameter.requires_grad for parameter in agent.adapter_net.parameters()):
        raise RuntimeError("The new C3-v2 adapter is not trainable")

    agent.target_adapter_net = C3ResidualAdapter(FEATURE_DIM, len(ACTIONS))
    agent.target_adapter_net.load_state_dict(agent.adapter_net.state_dict(), strict=True)
    agent.target_adapter_net.eval()
    for parameter in agent.target_adapter_net.parameters():
        parameter.requires_grad_(False)
    agent.optimizer = torch.optim.Adam(
        agent.adapter_net.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
    )
    agent.replay_buffer = LAPReplayBuffer(
        REPLAY_CAPACITY, FEATURE_DIM, LAP_ALPHA, LAP_PRIORITY_FLOOR
    )
    train._load_rehearsal_buffer(agent)
    sample = agent.rehearsal_buffer.sample(8, agent.rng)
    if sample["states"].shape != (8, FEATURE_DIM):
        raise RuntimeError("Frozen C1 rehearsal returned the wrong shape")

    for index in range(24):
        state = np.full(FEATURE_DIM, index / 24.0, dtype=np.float32)
        agent.replay_buffer.add(
            state,
            index % len(ACTIONS),
            float((index % 3) - 1),
            None,
            True,
            None,
            3,
            np.zeros(len(ACTIONS), dtype=np.float32),
            np.zeros(len(ACTIONS), dtype=np.bool_),
        )
    agent.metric_window = train._fresh_metric_window()
    agent.optimizer_step = 0
    agent.source_transition_count = 24
    agent.transition_count = 24
    train._learn_once(agent)
    if agent.optimizer_step != 1:
        raise RuntimeError("The in-memory C3-v2 update did not run")
    if not np.isfinite(agent.metric_window["loss_sum"]):
        raise RuntimeError("The C3-v2 update produced a non-finite loss")

    print("M2-3 source: {}".format(verification["sources"]["m2_3_sha256"]))
    print("Frozen C1 source: {}".format(verification["sources"]["c1_sha256"]))
    print("Frozen rehearsal items: {}".format(len(agent.rehearsal_buffer)))
    print("C3-v2 t0 max Q error: {:.1e}".format(max_error))
    print("C3-v2 one-update loss: {:.6f}".format(agent.metric_window["loss_sum"]))
    print("C3-v2 setup verification passed")


if __name__ == "__main__":
    main()
