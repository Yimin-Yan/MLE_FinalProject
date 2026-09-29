"""Small deterministic checks for the M2-2 temporal planner."""

import unittest

import numpy as np

from ..m2_1.features import state_to_features as m2_1_state_to_features
from .config import ACTION_TO_INDEX, FEATURE_DIM, TEMPORAL_HORIZON
from .features import state_to_features
from .planner import KnownHazardPlanner, _tile_is_free_at


def _open_field(size=9):
    field = np.zeros((size, size), dtype=np.int16)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def _state(
    field=None,
    position=(3, 3),
    bombs=(),
    explosion_cells=(),
    bombs_left=True,
    others=(),
):
    if field is None:
        field = _open_field()
    explosion_map = np.zeros_like(field, dtype=np.float32)
    for cell in explosion_cells:
        explosion_map[cell] = 1.0
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("seat_d3qn", 0, bombs_left, position),
        "others": list(others),
        "bombs": list(bombs),
        "coins": [],
        "explosion_map": explosion_map,
        "user_input": "WAIT",
    }


class PlannerRuleTests(unittest.TestCase):
    def setUp(self):
        self.planner = KnownHazardPlanner()

    def test_no_hazard_state_is_safe(self):
        analysis = self.planner.analyze(_state())
        wait = ACTION_TO_INDEX["WAIT"]
        self.assertTrue(analysis.escape_feasible[wait])
        self.assertEqual(float(analysis.danger_arrival[wait]), 1.0)

    def test_timer_zero_bomb_explodes_after_current_action(self):
        game_state = _state(bombs=(((3, 3), 0),), bombs_left=False)
        analysis = self.planner.analyze(game_state)
        wait = ACTION_TO_INDEX["WAIT"]
        self.assertEqual(float(analysis.danger_arrival[wait]), 0.0)
        self.assertFalse(analysis.escape_feasible[wait])

    def test_new_bomb_uses_the_same_step_countdown(self):
        analysis = self.planner.analyze(_state(position=(4, 4)))
        bomb = ACTION_TO_INDEX["BOMB"]
        expected = (5 - 1) / float(TEMPORAL_HORIZON)
        self.assertAlmostEqual(float(analysis.danger_arrival[bomb]), expected)

    def test_stone_wall_stops_blast(self):
        field = _open_field()
        field[4, 3] = -1
        game_state = _state(
            field=field,
            position=(5, 3),
            bombs=(((3, 3), 0),),
            bombs_left=False,
        )
        analysis = self.planner.analyze(game_state)
        self.assertEqual(float(analysis.danger_arrival[ACTION_TO_INDEX["WAIT"]]), 1.0)

    def test_crate_does_not_stop_blast(self):
        field = _open_field()
        field[4, 3] = 1
        game_state = _state(
            field=field,
            position=(5, 3),
            bombs=(((3, 3), 0),),
            bombs_left=False,
        )
        analysis = self.planner.analyze(game_state)
        self.assertEqual(float(analysis.danger_arrival[ACTION_TO_INDEX["WAIT"]]), 0.0)

    def test_bombs_do_not_chain_react(self):
        game_state = _state(
            position=(5, 5),
            bombs=(((3, 3), 0), ((5, 3), 3)),
            bombs_left=False,
        )
        analysis = self.planner.analyze(game_state)
        expected = (4 - 1) / float(TEMPORAL_HORIZON)
        self.assertAlmostEqual(
            float(analysis.danger_arrival[ACTION_TO_INDEX["WAIT"]]),
            expected,
        )

    def test_bomb_tile_cannot_be_reentered(self):
        game_state = _state(position=(4, 4))
        timeline = self.planner._make_timeline(
            game_state["field"],
            game_state["explosion_map"],
            (((4, 4), 4),),
        )
        self.assertFalse(_tile_is_free_at((4, 4), 2, timeline))

    def test_bomb_in_closed_dead_end_is_unsafe(self):
        field = np.full((7, 7), -1, dtype=np.int16)
        field[1, 1] = 0
        field[2, 1] = 0
        game_state = _state(field=field, position=(1, 1))
        analysis = self.planner.analyze(game_state)
        self.assertFalse(analysis.escape_feasible[ACTION_TO_INDEX["BOMB"]])

    def test_bomb_in_open_area_has_an_escape(self):
        analysis = self.planner.analyze(_state(position=(4, 4)))
        self.assertTrue(analysis.escape_feasible[ACTION_TO_INDEX["BOMB"]])

    def test_active_explosion_is_checked_before_labeling_safe(self):
        analysis = self.planner.analyze(
            _state(position=(4, 4), explosion_cells=((4, 4),))
        )
        self.assertFalse(analysis.escape_feasible[ACTION_TO_INDEX["WAIT"]])
        self.assertTrue(analysis.escape_feasible[ACTION_TO_INDEX["UP"]])

    def test_overlapping_blasts_keep_the_cell_unsafe(self):
        game_state = _state(
            position=(4, 4),
            bombs=(((2, 4), 0), ((4, 2), 0)),
            bombs_left=False,
        )
        analysis = self.planner.analyze(game_state)
        self.assertFalse(analysis.escape_feasible[ACTION_TO_INDEX["WAIT"]])

    def test_destroyed_crate_opens_on_the_following_step(self):
        field = _open_field()
        field[4, 3] = 1
        game_state = _state(field=field, bombs=(((3, 3), 0),), bombs_left=False)
        timeline = self.planner._make_timeline(
            field,
            game_state["explosion_map"],
            tuple(game_state["bombs"]),
        )
        self.assertFalse(_tile_is_free_at((4, 3), 1, timeline))
        self.assertTrue(_tile_is_free_at((4, 3), 2, timeline))

    def test_cache_reuses_only_identical_observation(self):
        game_state = _state()
        first = self.planner.analyze(game_state)
        second = self.planner.analyze(game_state)
        self.assertIs(first, second)
        self.assertEqual(self.planner.computations, 1)
        self.assertEqual(self.planner.cache_hits, 1)

    def test_same_step_with_changed_contents_is_recomputed(self):
        before_action = _state(position=(3, 3))
        after_action = _state(
            position=(3, 3),
            bombs=(((3, 3), 3),),
            bombs_left=False,
        )
        first = self.planner.analyze(before_action)
        second = self.planner.analyze(after_action)
        self.assertIsNot(first, second)
        self.assertEqual(self.planner.computations, 2)


class FeatureContractTests(unittest.TestCase):
    def test_first_42_features_match_m2_1(self):
        game_state = _state(
            position=(4, 4),
            bombs=(((2, 4), 2),),
            bombs_left=False,
        )
        analysis = KnownHazardPlanner().analyze(game_state)
        old_features = m2_1_state_to_features(game_state)
        new_features = state_to_features(game_state, analysis)
        np.testing.assert_array_equal(new_features[:42], old_features[:42])

    def test_feature_shape_and_range(self):
        game_state = _state(position=(4, 4), bombs=(((2, 4), 2),), bombs_left=False)
        analysis = KnownHazardPlanner().analyze(game_state)
        features = state_to_features(game_state, analysis)
        self.assertEqual(features.shape, (FEATURE_DIM,))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreaterEqual(float(features[42:64].min()), -1.0)
        self.assertLessEqual(float(features[42:64].max()), 1.0)


if __name__ == "__main__":
    unittest.main()
