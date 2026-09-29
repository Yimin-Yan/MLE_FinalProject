"""Read-only transfer check for the frozen M2-2 starting point."""

import unittest

import torch

from .config import ACTIONS, FEATURE_DIM, M2_2_MODEL_PATH
from .model import TemporalDuelingDQN


class TransferTests(unittest.TestCase):
    def test_m2_2_checkpoint_loads_without_changing_q_values(self):
        checkpoint = torch.load(M2_2_MODEL_PATH, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["metadata"]["variant_id"], "M2-2")
        self.assertEqual(int(checkpoint["transition_count"]), 250_000)

        source = TemporalDuelingDQN(FEATURE_DIM, len(ACTIONS))
        target = TemporalDuelingDQN(FEATURE_DIM, len(ACTIONS))
        source.load_state_dict(checkpoint["model_state_dict"], strict=True)
        target.load_state_dict(checkpoint["model_state_dict"], strict=True)
        probe = torch.linspace(-1.0, 1.0, 4 * FEATURE_DIM).reshape(4, FEATURE_DIM)
        with torch.no_grad():
            error = float((source(probe) - target(probe)).abs().max().item())
        self.assertLessEqual(error, 1e-6)


if __name__ == "__main__":
    unittest.main()
