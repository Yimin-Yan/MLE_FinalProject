"""Checks for the interleaved M2-3 opponent schedule."""

import unittest

from .config import CURRICULUM_STAGES, CURRICULUM_TARGETS
from .run_curriculum import OPPONENTS, _current_block


class CurriculumTests(unittest.TestCase):
    def test_twenty_five_blocks_cover_exactly_250k_transitions(self):
        self.assertEqual(
            CURRICULUM_TARGETS,
            tuple(range(10_000, 250_001, 10_000)),
        )
        self.assertEqual(CURRICULUM_STAGES.count("official"), 12)
        self.assertEqual(CURRICULUM_STAGES.count("rule_heavy"), 8)
        self.assertEqual(CURRICULUM_STAGES.count("weak"), 5)

    def test_block_boundaries_and_opponents(self):
        for index, (target, stage) in enumerate(
            zip(CURRICULUM_TARGETS, CURRICULUM_STAGES)
        ):
            start = index * 10_000
            block = _current_block(start)
            self.assertEqual(block[:3], (start, target, stage))
            self.assertEqual(block[3], OPPONENTS[stage])
        self.assertIsNone(_current_block(250_000))

if __name__ == "__main__":
    unittest.main()
