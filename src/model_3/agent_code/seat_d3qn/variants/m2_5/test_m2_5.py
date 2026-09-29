"""Unit checks for labels, inheritance, and the conservative action gate."""

from collections import Counter
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import events as e

from .config import (
    ACTIONS,
    BLOCK_TRANSITIONS,
    COLLECTION_BLOCKS,
    FEATURE_DIM,
    MAX_COLLECTION_TRANSITIONS,
    SOURCE_SHA256,
)
from .dataset import append_rows, empty_shard, load_shard, save_shard
from .labels import wasteful_death_labels
from .model import WastefulDeathRisk
from .policy import select_recovery_action
from .source import FrozenC3Policy, sha256, verify_source_files


def _records(*event_lists):
    return [{"events": tuple(items)} for items in event_lists]


class LabelTests(unittest.TestCase):
    def test_plain_self_death_is_positive(self):
        labels, distances = wasteful_death_labels(
            _records([], [], [e.KILLED_SELF], [])
        )
        self.assertEqual(labels[:3], [1, 1, 1])
        self.assertEqual(distances[:3], [3, 2, 1])

    def test_coin_before_self_death_is_not_wasteful(self):
        labels, _ = wasteful_death_labels(
            _records([], [e.COIN_COLLECTED], [e.KILLED_SELF])
        )
        self.assertEqual(labels[0], 0)

    def test_trade_kill_is_not_wasteful(self):
        labels, _ = wasteful_death_labels(
            _records([], [e.KILLED_OPPONENT, e.KILLED_SELF])
        )
        self.assertEqual(labels, [0, 0])


class GateTests(unittest.TestCase):
    def setUp(self):
        self.legal = np.ones(6, dtype=np.bool_)

    def test_gate_selects_only_near_value_recovery(self):
        q_values = np.asarray([1.0, 0.96, 0.4, 0.3, 0.2, 0.1])
        risks = np.asarray([0.95, 0.50, 0.1, 0.1, 0.1, 0.0])
        action, changed = select_recovery_action(
            q_values, risks, self.legal, True, 0.90, 0.25, 0.075
        )
        self.assertTrue(changed)
        self.assertEqual(action, 1)

    def test_gate_does_not_override_bomb(self):
        q_values = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        risks = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.99])
        action, changed = select_recovery_action(
            q_values, risks, self.legal, True, 0.90, 0.25, 0.075
        )
        self.assertFalse(changed)
        self.assertEqual(action, 5)

    def test_gate_never_introduces_bomb(self):
        q_values = np.asarray([1.0, 0.95, 0.0, 0.0, 0.0, 0.99])
        risks = np.asarray([0.99, 0.60, 0.0, 0.0, 0.0, 0.01])
        action, changed = select_recovery_action(
            q_values, risks, self.legal, True, 0.90, 0.25, 0.075
        )
        self.assertTrue(changed)
        self.assertEqual(action, 1)

    def test_no_hazard_is_exact_source_action(self):
        q_values = np.asarray([1.0, 0.9, 0.0, 0.0, 0.0, 0.0])
        risks = np.asarray([0.99, 0.0, 0.0, 0.0, 0.0, 0.0])
        action, changed = select_recovery_action(
            q_values, risks, self.legal, False, 0.90, 0.25, 0.075
        )
        self.assertFalse(changed)
        self.assertEqual(action, 0)


class DataAndSourceTests(unittest.TestCase):
    def test_collection_plan_is_exact(self):
        counts = Counter(domain for domain, _ in COLLECTION_BLOCKS)
        self.assertEqual(counts, Counter({"official": 4, "pressure": 3, "stochastic": 3}))
        self.assertEqual(len(COLLECTION_BLOCKS) * BLOCK_TRANSITIONS, MAX_COLLECTION_TRANSITIONS)

    def test_dataset_round_trip(self):
        rows = empty_shard()
        one = {
            "features": np.zeros((1, FEATURE_DIM), dtype=np.float32),
            "actions": np.zeros(1, dtype=np.int8),
            "labels": np.ones(1, dtype=np.uint8),
            "distances": np.ones(1, dtype=np.uint8),
            "explored": np.zeros(1, dtype=np.bool_),
            "round_ids": np.ones(1, dtype=np.int64),
            "steps": np.ones(1, dtype=np.int16),
            "domain_ids": np.zeros(1, dtype=np.int8),
        }
        rows = append_rows(rows, one, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            save_shard(path, rows)
            loaded = load_shard(path)
        self.assertEqual(int(loaded["labels"][0]), 1)

    def test_frozen_source_and_model_shapes(self):
        source_checkpoint = verify_source_files()
        from .config import SOURCE_MODEL_PATH

        self.assertEqual(sha256(SOURCE_MODEL_PATH), SOURCE_SHA256)
        self.assertEqual(int(source_checkpoint["transition_count"]), 225_000)
        source = FrozenC3Policy("cpu")
        risk = WastefulDeathRisk()
        probe = torch.zeros((3, FEATURE_DIM))
        self.assertEqual(tuple(source(probe).shape), (3, len(ACTIONS)))
        self.assertEqual(tuple(risk(probe).shape), (3, len(ACTIONS)))
        self.assertFalse(any(parameter.requires_grad for parameter in source.parameters()))


if __name__ == "__main__":
    unittest.main()
