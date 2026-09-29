"""Network and packaged-checkpoint checks for the final M2-3 agent."""

import unittest

import torch

from .callbacks import _check_model_metadata
from .config import ACTIONS, FEATURE_DIM, MODEL_PATH
from .model import TemporalDuelingDQN, mask_q_values


class ModelTests(unittest.TestCase):
    def test_network_outputs_have_expected_shapes(self):
        network = TemporalDuelingDQN()
        states = torch.zeros((4, FEATURE_DIM), dtype=torch.float32)
        q_values, escape_logits = network.forward_with_escape(states)
        self.assertEqual(tuple(q_values.shape), (4, len(ACTIONS)))
        self.assertEqual(tuple(escape_logits.shape), (4, len(ACTIONS)))

    def test_illegal_action_cannot_win_argmax(self):
        q_values = torch.tensor([[1.0, 50.0, 3.0, 4.0, 5.0, 6.0]])
        legal = torch.tensor([[True, False, True, True, True, True]])
        masked = mask_q_values(q_values, legal)
        self.assertEqual(int(masked.argmax(dim=1).item()), 5)
        self.assertTrue(torch.isneginf(masked[0, 1]))

    def test_packaged_model_is_the_selected_150k_checkpoint(self):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        _check_model_metadata(checkpoint["metadata"])
        self.assertEqual(int(checkpoint["transition_count"]), 150_000)

        network = TemporalDuelingDQN()
        network.load_state_dict(checkpoint["model_state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()
