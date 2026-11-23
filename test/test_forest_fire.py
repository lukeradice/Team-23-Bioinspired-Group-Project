import unittest
from unittest.mock import patch

import numpy as np

from ca_descriptions import forest_fire_model as fire
from capyle.ca import CAConfig, Grid2D


class TestFireSpread(unittest.TestCase):
    def setUp(self):
        self.config = CAConfig(fire.__file__)
        self.config.grid_dims = (5, 7)
        self.config.states = tuple(range(7))
        self.config.fill_in_defaults()
        self.config.wrap = False
        self.config.heights = np.zeros(self.config.grid_dims)
        self.config.initial_grid.fill(fire.LAKE_STATE)
        self.config.initial_grid[2, 0] = fire.BURNING_STATE
        self.config.initial_grid[2, 1:5] = fire.CHAPARRAL_STATE
        self.config.initial_grid[2, 5] = fire.TOWN_STATE

    def test_runs_with_the_same_seed_replay_the_same_draws(self):
        first = fire.Context.from_config(self.config)
        second = fire.Context.from_config(self.config)
        np.testing.assert_array_equal(first.rng.random(8), second.rng.random(8))

    def test_water_drops_once_on_the_first_frame_ending_after_the_scheduled_hour(self):
        for dt, hour, expected_step in ((0.2, 0.34, 2), (0.05, 0.34, 7), (0.05, 0.35, 7)):
            class Run(fire.Context):
                timestep_hours = dt
                water_deploy_hours = hour

            ctx = Run.from_config(self.config)
            engine = Grid2D(self.config, (fire.transition_function, ctx))
            with patch.object(fire, 'deploy_water', side_effect=lambda grid: grid) as drop:
                for _ in range(expected_step - 1):
                    engine.step()
                drop.assert_not_called()
                engine.step()
                drop.assert_called_once()
                engine.step()
                drop.assert_called_once()
