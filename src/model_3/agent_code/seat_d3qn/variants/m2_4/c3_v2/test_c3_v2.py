"""Small deterministic tests for the independent C3-v2 components."""

import unittest

import numpy as np
import torch

from .config import (
    FEATURE_DIM,
    GATE_MODE,
    GATE_TARGETS,
    MAX_TRAINING_TRANSITIONS,
    STAGE_NAMES,
    epsilon_for_transition,
    stage_for_transition,
)
from .model import C3ResidualAdapter, mask_q_values
from .rehearsal_buffer import FrozenRehearsalBuffer


class C3V2Tests(unittest.TestCase):
    def test_zero_adapter_keeps_all_actions_at_zero(self):
        network = C3ResidualAdapter(FEATURE_DIM, 6)
        values = network(torch.randn(11, FEATURE_DIM))
        self.assertTrue(torch.equal(values, torch.zeros_like(values)))

    def test_schedule_covers_the_full_budget(self):
        self.assertEqual(GATE_MODE, "record_only")
        self.assertEqual(len(STAGE_NAMES), 10)
        self.assertEqual(GATE_TARGETS[-1], MAX_TRAINING_TRANSITIONS)
        self.assertEqual(stage_for_transition(0), (0, 50_000, "official"))
        self.assertEqual(
            stage_for_transition(499_999), (450_000, 500_000, "official")
        )
        self.assertIsNone(stage_for_transition(500_000))

    def test_epsilon_depends_on_effective_transitions(self):
        self.assertAlmostEqual(epsilon_for_transition(0), 0.05)
        self.assertAlmostEqual(epsilon_for_transition(50_000), 0.03)
        self.assertAlmostEqual(epsilon_for_transition(100_000), 0.01)
        self.assertAlmostEqual(epsilon_for_transition(500_000), 0.01)

    def test_mask_rejects_illegal_actions(self):
        values = torch.tensor([[10.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        mask = torch.tensor([[False, True, True, True, True, True]])
        self.assertEqual(int(mask_q_values(values, mask).argmax()), 5)

    def test_frozen_rehearsal_batch_shapes(self):
        size = 16
        state = {
            "feature_dim": FEATURE_DIM,
            "alpha": 0.6,
            "priority_floor": 1.0,
            "size": size,
            "states": np.zeros((size, FEATURE_DIM), dtype=np.float32),
            "actions": np.arange(size, dtype=np.int64) % 6,
            "rewards": np.zeros(size, dtype=np.float32),
            "next_states": np.zeros((size, FEATURE_DIM), dtype=np.float32),
            "dones": np.ones(size, dtype=np.bool_),
            "next_masks": np.zeros((size, 6), dtype=np.bool_),
            "horizons": np.ones(size, dtype=np.int64),
            "escape_labels": np.zeros((size, 6), dtype=np.float32),
            "escape_loss_masks": np.zeros((size, 6), dtype=np.bool_),
            "raw_priorities": np.ones(size, dtype=np.float64),
        }
        buffer = FrozenRehearsalBuffer(state, FEATURE_DIM, 0.6, 1.0)
        batch = buffer.sample(8, np.random.default_rng(42))
        self.assertEqual(batch["states"].shape, (8, FEATURE_DIM))
        self.assertEqual(batch["actions"].shape, (8,))


if __name__ == "__main__":
    unittest.main()
