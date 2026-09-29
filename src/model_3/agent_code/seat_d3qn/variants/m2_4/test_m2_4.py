"""Small deterministic checks for the new M2-4 components."""

from collections import Counter
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from .config import (
    BRANCH,
    C3_LEAGUE_STAGES,
    C3_LEAGUE_TARGETS,
    FEATURE_DIM,
    TEACHER_BUFFER_CAPACITY,
    teacher_weight,
)
from .model import ResidualQNetwork
from .callbacks import setup
from . import train as train_callbacks
from .teacher import RuleTeacher
from .teacher_buffer import TeacherReplayBuffer
from .select_best import _passes_c3_guardrail


class M24ComponentTests(unittest.TestCase):
    def test_formal_m2_3_transfer_is_exact(self):
        logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        agent = SimpleNamespace(train=True, logger=logger)
        setup(agent)
        self.assertEqual(agent.transfer_max_q_error, 0.0)
        self.assertTrue(all(not parameter.requires_grad for parameter in agent.base_net.parameters()))

    def test_zero_residual_is_exact(self):
        network = ResidualQNetwork()
        states = torch.linspace(-1.0, 1.0, steps=3 * FEATURE_DIM).reshape(3, FEATURE_DIM)
        output = network(states)
        self.assertEqual(float(output.abs().max().item()), 0.0)

    def test_teacher_schedule_reaches_zero(self):
        if BRANCH != "c2":
            self.skipTest("teacher schedule belongs to C2")
        self.assertAlmostEqual(teacher_weight(0), 0.10)
        self.assertAlmostEqual(teacher_weight(50_000), 0.05)
        self.assertAlmostEqual(teacher_weight(125_000), 0.02)
        self.assertAlmostEqual(teacher_weight(200_000), 0.0)
        self.assertAlmostEqual(teacher_weight(250_000), 0.0)

    def test_teacher_buffer_round_trip(self):
        replay = TeacherReplayBuffer(TEACHER_BUFFER_CAPACITY, FEATURE_DIM)
        state = np.zeros(FEATURE_DIM, dtype=np.float32)
        mask = np.ones(6, dtype=np.bool_)
        replay.add(state, mask, 5, 0.8)
        restored = TeacherReplayBuffer(TEACHER_BUFFER_CAPACITY, FEATURE_DIM)
        restored.load_state_dict(replay.state_dict())
        sample = restored.sample(1, np.random.default_rng(3))
        self.assertEqual(int(sample["actions"][0]), 5)
        self.assertAlmostEqual(float(sample["confidences"][0]), 0.8, places=6)

    def test_teacher_query_restores_global_rng(self):
        teacher = RuleTeacher(seed=7)
        game_state = {
            "round": 1,
            "self": ("seat_d3qn", 0, True, (1, 1)),
        }

        def fake_act(_clone, _state):
            random.random()
            np.random.random()
            return "WAIT"

        random.seed(19)
        np.random.seed(23)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        with patch(
            "agent_code.seat_d3qn.variants.m2_4.teacher.rule_callbacks.act",
            side_effect=fake_act,
        ):
            vote = teacher.query(game_state)
        self.assertEqual(vote.action, "WAIT")
        self.assertEqual(vote.confidence, 1.0)
        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(restored_numpy_state[1], numpy_state[1])
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])

    def test_teacher_can_abstain(self):
        teacher = RuleTeacher(seed=13)
        game_state = {
            "round": 1,
            "self": ("seat_d3qn", 0, True, (1, 1)),
        }
        with patch(
            "agent_code.seat_d3qn.variants.m2_4.teacher.rule_callbacks.act",
            return_value=None,
        ):
            vote = teacher.query(game_state)
        self.assertIsNone(vote.action)
        self.assertEqual(vote.confidence, 0.0)
        self.assertEqual(vote.votes, (0, 0, 0, 0, 0, 0))

    def test_c3_league_percentages(self):
        expected = {
            "hard_rule": 4,
            "self_history": 3,
            "attack": 3,
            "non_rule_strong": 3,
            "conservative": 2,
            "resource": 2,
            "noisy": 2,
            "mixed": 1,
        }
        self.assertEqual(len(C3_LEAGUE_TARGETS), 20)
        self.assertEqual(C3_LEAGUE_TARGETS[-1], 250_000)
        self.assertEqual(Counter(C3_LEAGUE_STAGES), expected)

    def test_c3_suicide_guardrail(self):
        baseline = {"mean_official_score": 5.0, "suicide_rate": 0.35}
        small_gain = {"mean_official_score": 5.1, "suicide_rate": 0.41}
        large_gain = {"mean_official_score": 5.31, "suicide_rate": 0.41}
        safe_gain = {"mean_official_score": 5.1, "suicide_rate": 0.39}
        self.assertFalse(_passes_c3_guardrail(small_gain, baseline))
        self.assertTrue(_passes_c3_guardrail(large_gain, baseline))
        self.assertTrue(_passes_c3_guardrail(safe_gain, baseline))

    def test_official_teacher_interface(self):
        field = np.zeros((17, 17), dtype=np.int64)
        field[0, :] = -1
        field[-1, :] = -1
        field[:, 0] = -1
        field[:, -1] = -1
        game_state = {
            "round": 1,
            "step": 1,
            "field": field,
            "self": ("seat_d3qn", 0, True, (1, 1)),
            "others": [("other", 0, True, (15, 15))],
            "bombs": [],
            "coins": [(3, 3)],
            "explosion_map": np.zeros_like(field),
        }
        vote = RuleTeacher(seed=11).query(game_state)
        self.assertIn(vote.action, ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"))
        self.assertGreaterEqual(vote.confidence, 0.2)

    def test_one_residual_update_is_finite(self):
        logger = SimpleNamespace(
            info=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        )
        agent = SimpleNamespace(train=True, logger=logger)
        setup(agent)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(train_callbacks, "MODEL_SNAPSHOT_DIR", root / "snapshots"), \
                    patch.object(train_callbacks, "METRICS_PATH", root / "metrics.jsonl"), \
                    patch.object(train_callbacks, "TRAINING_CHECKPOINT_PATH", root / "checkpoint.pt"), \
                    patch.object(train_callbacks, "MODEL_PATH", root / "model.pt"), \
                    patch.object(train_callbacks, "RESUME_TRAINING", False):
                train_callbacks.setup_training(agent)

                if BRANCH == "c3":
                    states = torch.linspace(
                        -1.0, 1.0, steps=2 * FEATURE_DIM
                    ).reshape(2, FEATURE_DIM)
                    with torch.no_grad():
                        difference = (
                            agent.residual_net(states)
                            - agent.reference_residual_net(states)
                        ).abs().max()
                    self.assertEqual(float(difference.item()), 0.0)

                legal_mask = np.ones(6, dtype=np.bool_)
                escape = np.ones(6, dtype=np.float32)
                for index in range(32):
                    state = np.full(FEATURE_DIM, index / 32.0, dtype=np.float32)
                    next_state = state + np.float32(0.01)
                    agent.replay_buffer.add(
                        state,
                        index % 6,
                        float((index % 5) - 2) / 10.0,
                        next_state,
                        False,
                        legal_mask,
                        3,
                        escape,
                        legal_mask,
                    )
                for index in range(8):
                    state = np.full(FEATURE_DIM, index / 8.0, dtype=np.float32)
                    agent.teacher_buffer.add(state, legal_mask, index % 6, 0.8)

                train_callbacks._learn_once(agent)
                self.assertEqual(agent.optimizer_step, 1)
                for parameter in agent.residual_net.parameters():
                    self.assertTrue(torch.isfinite(parameter).all())


if __name__ == "__main__":
    unittest.main()
