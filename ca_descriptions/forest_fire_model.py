# Name: FOREST FIRE MODEL CA
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')
# ---

from capyle.ca import Grid2D
import capyle.utils as utils
import numpy as np
from typing import NamedTuple

BURNING_STATE = 0
BURNT_STATE = 1
CHAPARRAL_STATE = 2
FOREST_STATE = 3
LAKE_STATE = 4
FLAMMABLE_SCRUBLAND_STATE = 5
TOWN_STATE = 6


def build_terrain_grid(grid_dims):
    """Builds the terrain grid. Should not include any burning/burnt cells."""
    
    # Chaparral
    initial_grid = np.full(grid_dims, CHAPARRAL_STATE, dtype=np.int8)

    # Forests
    initial_grid[20:30, 0:100] = FOREST_STATE 
    initial_grid[30:140, 20:50] = FOREST_STATE 
    initial_grid[120:140, 50:100] = FOREST_STATE 
    initial_grid[40:60, 130:180] = FOREST_STATE 

    # Lakes
    initial_grid[50:90, 70:90] = LAKE_STATE
    initial_grid[110:160, 150:160] = LAKE_STATE

    # Highly flammable scrubland
    initial_grid[90:100, 100:180] = FLAMMABLE_SCRUBLAND_STATE

    # Town
    initial_grid[175:185, 75:85] = TOWN_STATE

    return initial_grid


class Context(NamedTuple):
    cell_km = 0.25
    timestep_hours = 0.2
    # Direction the wind blows from in compass degrees (0 = northerly, 90 = easterly)
    wind_from_deg = 0.0
    # Wind speed in m/s
    wind_speed = 0.0

    terrain_grid: np.ndarray
    fuel_grid: np.ndarray
    # spread multiplier for each neighbour, in neighbourstates order
    wind_weights: np.ndarray

    # Must be defined after other fields otherwise it errors. Like sure whatever
    rng: np.random.Generator = np.random.default_rng(67)

    @classmethod
    def from_config(cls, config):
        """Create a new context with default parameters.

        The default wind is northerly at 5 m/s. NEEDS TO BE CALIBRATED
        """

        # Create a read-only terrain grid
        terrain_grid = build_terrain_grid(config.grid_dims)
        terrain_grid.setflags(write=False)

        # Burn durations in hours, independent of cell area.
        burn_times = np.zeros(config.grid_dims, dtype=float)
        burn_times[config.initial_grid == FOREST_STATE] = 30 * 24  # 30 days
        burn_times[config.initial_grid == CHAPARRAL_STATE] = 3 * 24  # 3 days
        burn_times[config.initial_grid == FLAMMABLE_SCRUBLAND_STATE] = 6  # 6 hours

        # Convert this to the number of simulation steps that a cell should burn for
        fuel_grid = np.ceil(burn_times / cls.timestep_hours).astype(int)
        fuel_grid[config.initial_grid == BURNING_STATE] = -1  # Initial burning cells have infinite fuel

        # Wind constants from alexandridis (2008) - wind speed in m/s
        WIND_C1 = 0.045
        WIND_C2 = 0.131

        # Compass bearing (degrees clockwise from north) of each neighbour
        # in the same order as CAPyLE's neighbourstates: NW, N, NE, W, E, SW, S, SE
        NEIGHBOUR_BEARINGS = np.array([315, 0, 45, 270, 90, 225, 180, 135])

        # angle between the wind and the direction fire travels from each neighbour
        angles = np.radians(NEIGHBOUR_BEARINGS - cls.wind_from_deg)

        wind_weights = np.exp(WIND_C1 * cls.wind_speed) * np.exp(WIND_C2 * cls.wind_speed * (np.cos(angles) - 1))

        return cls(
            terrain_grid=terrain_grid,
            fuel_grid=fuel_grid,
            wind_weights=wind_weights
        )


def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = "FOREST FIRE MODEL CA"
    config.dimensions = 2

    config.states = (
        BURNING_STATE, 
        BURNT_STATE, 
        CHAPARRAL_STATE, 
        FOREST_STATE, 
        LAKE_STATE, 
        FLAMMABLE_SCRUBLAND_STATE, 
        TOWN_STATE)
    # -------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----
    BURNING_COLOR = (1, 0.5, 0.3)
    BURNT_COLOR = (0.5, 0.5, 0.5)
    CHAPARRAL_COLOR = (0.75, 0.75, 0)
    FOREST_COLOR = (0.31, 0.38, 0.15)
    LAKE_COLOR = (0.24, 0.69, 0.94)
    FLAMMABLE_SCRUBLAND_COLOR = (1, 1, 0)
    TOWN_COLOR = (0, 0, 0)

    config.state_colors = [
        BURNING_COLOR, 
        BURNT_COLOR, 
        CHAPARRAL_COLOR, 
        FOREST_COLOR, 
        LAKE_COLOR, 
        FLAMMABLE_SCRUBLAND_COLOR, 
        TOWN_COLOR
    ]
    config.grid_dims = (200, 200)

    config.initial_grid = build_terrain_grid(config.grid_dims)

    # INCERATOR_LOCATION = (0, config.grid_dims[1] - 1) # top right corner
    # config.initial_grid[INCERATOR_LOCATION] = BURNING_STATE

    POWER_PLANT_LOCATION = (0, 0)
    config.initial_grid[POWER_PLANT_LOCATION] = BURNING_STATE

    # Stop fire crossing an edge and reappearing on the opposite side.
    config.wrap = False

    # ----------------------------------------------------------------------

    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()
    return config

def get_fire_spread_probability(states: np.ndarray, fire_exposure: np.ndarray, ctx: Context) -> np.ndarray:
    """Calculates the probability that each cell will catch fire from its neighbours."""

    # Values in km/h
    spread_rates = np.zeros(states.shape, dtype=float)

    # <0.3km/h - Natural Resources Canada, Fire Behaviour.
    spread_rates[states == FOREST_STATE] = 0.2

    # 0.79–2.02km/h - Stephens et al. (2008), Measuring the Rate of Spread of Chaparral Prescribed Fires in Northern California.
    spread_rates[states == CHAPARRAL_STATE] = 1.405  # Mean of the range

    # ~3km/h - Cruz & Alexander (2019)
    spread_rates[states == FLAMMABLE_SCRUBLAND_STATE] = 3.0

    spread = spread_rates * fire_exposure * ctx.timestep_hours / ctx.cell_km

    return 1 - np.exp(-spread)


def get_wind_weights(wind_from_deg: float, wind_speed: float) -> np.ndarray:
    """Returns one spread multiplier per neighbour (in neighbourstates order).

    Fire from a neighbour on the side the wind is coming from is blown
    onto the cell, so it gets a weight above 1. Fire from a neighbour
    downwind has to burn against the wind, so it gets a weight below 1.
    With no wind every weight is 1.
    """


def get_fire_exposure(neighbourstates: np.ndarray, ctx: Context) -> np.ndarray:
    """Like counting burning neighbours, but each one counts as its wind weight."""
    burning = neighbourstates == BURNING_STATE  # shape (8, rows, cols)

    # multiply each of the 8 neighbour grids by its wind weight then add them up
    return np.sum(ctx.wind_weights[:, None, None] * burning, axis=0)


def transition_function(grid, neighbourstates, neighbourcounts, ctx: Context):
    """Function to apply the transition rules
    and return the new grid"""

    # -- Consume fuel/burn out fires --

    burning = grid == BURNING_STATE

    # Negative fuel means infinite, so don't consume from negative fuel sources
    consuming_fuel = burning & (ctx.fuel_grid > 0)
    ctx.fuel_grid[consuming_fuel] -= 1

    # If burning and fuel has been exhausted, it is now fully burnt.
    grid[burning & (ctx.fuel_grid == 0)] = BURNT_STATE

    # --- Spread fires ---

    fire_exposure = get_fire_exposure(neighbourstates, ctx)
    ps = get_fire_spread_probability(grid, fire_exposure, ctx)
    igniting = ctx.rng.random(grid.shape) < ps

    grid[igniting] = BURNING_STATE

    return grid


def main():
    """ Main function that sets up, runs and saves CA"""
    # Get the config object from set up
    config = setup(sys.argv[1:])

    # Create grid object using parameters from config + transition function
    grid = Grid2D(config, (transition_function, Context.from_config(config)))

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # Save updated config to file
    config.save()
    # Save timeline to file
    utils.save(timeline, config.timeline_path)

if __name__ == "__main__":
    main()
