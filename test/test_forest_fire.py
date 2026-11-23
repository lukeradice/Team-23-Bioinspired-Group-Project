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

    def test_runs_with_the_same_seed_replay_burn_durations_and_spread_times(self):
        first = fire.Context.from_config(self.config)
        second = fire.Context.from_config(self.config)
        np.testing.assert_array_equal(first.burn_durations_hours, second.burn_durations_hours)
        np.testing.assert_array_equal(first.neighbour_spread_time_hours, second.neighbour_spread_time_hours)

    def test_timestep_changes_only_the_frame_of_a_known_arrival(self):
        for dt, expected_step in ((0.05, 2), (0.2, 1)):
            ctx = fire.Context.from_config(self.config)
            ctx.timestep_hours = dt
            ctx.neighbour_spread_time_hours = np.full_like(ctx.neighbour_spread_time_hours, np.inf)
            west = ctx.neighbour_offsets.index((0, -1))
            ctx.neighbour_spread_time_hours[west, 2, 1:6] = np.array([1, 1.4, 1, 1, 0]) / 60
            engine = Grid2D(self.config, (fire.transition_function, ctx))
            for _ in range(expected_step):
                engine.step()
            self.assertAlmostEqual(ctx.ignited_at_hours[2, 5], 4.4 / 60)
            self.assertEqual(ctx.town_arrival_step, expected_step)
            self.assertEqual(engine.grid[2, 5], fire.BURNING_STATE)
            engine.step()
            self.assertEqual(engine.grid[2, 5], fire.BURNT_STATE)

    def test_long_route_spreads_as_far_as_the_frame_deadline_allows(self):
        self.config.grid_dims = (3, 24)
        self.config.heights = np.zeros(self.config.grid_dims)
        self.config.initial_grid = np.full(self.config.grid_dims, fire.LAKE_STATE)
        self.config.initial_grid[1, :] = fire.CHAPARRAL_STATE
        self.config.initial_grid[1, 0] = fire.BURNING_STATE
        ctx = fire.Context.from_config(self.config)
        # 21 crossings of 1/64 hour fit exactly in the first frame.
        ctx.timestep_hours = 21 / 64
        ctx.neighbour_spread_time_hours = np.full_like(ctx.neighbour_spread_time_hours, np.inf)
        west = ctx.neighbour_offsets.index((0, -1))
        ctx.neighbour_spread_time_hours[west, 1, 1:] = 1 / 64
        engine = Grid2D(self.config, (fire.transition_function, ctx))
        engine.step()
        self.assertEqual(ctx.ignited_at_hours[1, 21], ctx.timestep_hours)
        self.assertTrue(np.isinf(ctx.ignited_at_hours[1, 22]))
        engine.step()
        self.assertEqual(ctx.ignited_at_hours[1, 22], 22 / 64)

    def test_burnout_or_water_can_interrupt_a_crossing(self):
        for water in (False, True):
            ctx = fire.Context.from_config(self.config)
            ctx.timestep_hours = 0.05
            ctx.neighbour_spread_time_hours = np.full_like(ctx.neighbour_spread_time_hours, np.inf)
            west = ctx.neighbour_offsets.index((0, -1))
            ctx.neighbour_spread_time_hours[west, 2, 1:3] = [0.02, 0.05]
            engine = Grid2D(self.config, (fire.transition_function, ctx))
            if water:
                ctx.water_deploy_timestep = 1
            else:
                ctx.burn_durations_hours = ctx.burn_durations_hours.copy()
                ctx.burn_durations_hours[2, 1] = 0.04

            def wet_source(grid: np.ndarray) -> np.ndarray:
                grid[2, 1] = fire.LAKE_STATE
                return grid

            with patch.object(fire, 'deploy_water', side_effect=wet_source):
                engine.step()
                engine.step()
            self.assertTrue(np.isinf(ctx.ignited_at_hours[2, 2]))

    def test_crossing_times_respect_distance_and_the_neighbourhood_mask(self):
        self.config.nhood_arr[0, 0] = 0
        with patch.object(fire.np.random, 'default_rng') as make_rng:
            rng = make_rng.return_value
            rng.uniform.side_effect = np.random.Generator(np.random.PCG64(67)).uniform
            rng.standard_exponential.side_effect = np.ones
            ctx = fire.Context.from_config(self.config)
        neighbour_spread_time_hours = ctx.neighbour_spread_time_hours
        self.assertTrue(np.isinf(neighbour_spread_time_hours[0]).all())
        self.assertTrue(np.isinf(neighbour_spread_time_hours[:, 0, 0]).all())
        self.assertAlmostEqual(neighbour_spread_time_hours[3, 2, 2], ctx.cell_km / 1.405)
        self.assertAlmostEqual(neighbour_spread_time_hours[2, 2, 2], np.sqrt(2) * neighbour_spread_time_hours[3, 2, 2])

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
