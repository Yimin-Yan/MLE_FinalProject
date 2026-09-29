"""Regression checks for Baseline artifact paths; all writes use temporary files."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from . import callbacks, config, select_best, train


class BaselinePathTests(unittest.TestCase):
    def test_shared_repository_layout(self):
        root = Path(__file__).resolve().parents[4]
        self.assertEqual(config.REPOSITORY_ROOT, root)
        self.assertTrue((root / "main.py").is_file())
        self.assertEqual(select_best.REPOSITORY_ROOT, root)
        self.assertEqual(select_best.VALIDATION_DIR, root / "results/m2_0/validation")
        self.assertEqual(train.MODEL_SNAPSHOT_DIR, select_best.MODEL_SNAPSHOT_DIR)
        self.assertEqual(config.MODEL_SNAPSHOT_DIR,
                         root / "experiments/seat_d3qn/m2_0/checkpoints/snapshots")
        self.assertEqual(config.METRICS_PATH,
                         root / "experiments/seat_d3qn/m2_0/metrics/training-metrics.jsonl")

    def test_installed_and_best_weights_remain_compatible(self):
        for path in (config.MODEL_PATH, config.BEST_MODEL_PATH):
            agent = SimpleNamespace(train=False, logger=Mock())
            with patch.dict(callbacks.os.environ, {config.EVALUATION_MODEL_ENV: str(path)}):
                callbacks.setup(agent)
            self.assertEqual(tuple(agent.online_net(torch.zeros(1, 64)).shape), (1, 6))

    def test_atomic_save_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints/snapshots/model.pt"
            train._atomic_torch_save({"value": torch.tensor([1])}, path)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            self.assertEqual(torch.load(path, weights_only=True)["value"].item(), 1)

    def test_fresh_training_creates_metrics_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            agent = SimpleNamespace(device=torch.device("cpu"), logger=Mock())
            agent.online_net = callbacks.DuelingDQN(input_dim=64, action_dim=6)
            with patch.object(train, "MODEL_SNAPSHOT_DIR", directory / "checkpoints/snapshots"), \
                 patch.object(train, "METRICS_PATH", directory / "metrics/training.jsonl"), \
                 patch.object(train, "RESUME_TRAINING", False):
                train.setup_training(agent)
            self.assertTrue((directory / "metrics/training.jsonl").is_file())

    def test_validation_launches_from_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            snapshot = output / "m2_0_t0050000.pt"
            result = output / "m2_0_t0050000_seed1101.json"
            result.write_text(json.dumps({"by_agent": {"seat_d3qn": {"rounds": 1}}}))
            with patch.object(select_best, "VALIDATION_DIR", output), \
                 patch.object(select_best, "REPOSITORY_ROOT", output), \
                 patch.object(select_best.subprocess, "run") as run:
                record = select_best._run_one_validation(snapshot, 1101, 1)
            self.assertEqual(run.call_args.kwargs["cwd"], output)
            self.assertEqual(run.call_args.kwargs["env"]["SEAT_D3QN_VARIANT"], "m2_0")
            self.assertEqual(record["result_file"], result.name)

    def test_missing_snapshots_do_not_start_games_or_install_models(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(select_best, "MODEL_SNAPSHOT_DIR", Path(directory)), \
             patch.object(select_best, "_arguments", return_value=SimpleNamespace(
                 rounds_per_seed=100, seeds=[1101])), \
             patch.object(select_best.subprocess, "run") as run, \
             patch.object(select_best.shutil, "copy2") as copy:
            with self.assertRaisesRegex(FileNotFoundError, "candidate snapshots"):
                select_best.main()
            run.assert_not_called()
            copy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
