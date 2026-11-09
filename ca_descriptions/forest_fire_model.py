# Name: NAME
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
    timestep_hours = 1.0

    terrain_grid: np.ndarray
    fuel_grid: np.ndarray

    @classmethod
    def from_config(cls, config):
        """Create a new context with default parameters."""

        # Create a read-only terrain grid
        terrain_grid = build_terrain_grid(config.grid_dims)
        terrain_grid.setflags(write=False)

        # Create a fuel grid. Each value represents the remaining burn time steps
        fuel_grid = np.zeros(config.grid_dims, dtype=np.int16)
        fuel_grid[config.initial_grid == BURNING_STATE] = -1  # Initial burning cells have infinite fuel

        return cls(
            terrain_grid=terrain_grid,
            fuel_grid=fuel_grid
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

    POWER_PLANT_LOCATION = (0, 0)
    INCERATOR_LOCATION = (0, config.grid_dims[1] - 1) # top right corner
    config.initial_grid = build_terrain_grid(config.grid_dims)
    config.initial_grid[INCERATOR_LOCATION] = BURNING_STATE

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



def get_fuel_timesteps(states: np.ndarray, ctx: Context) -> np.ndarray:
    fuel = np.zeros(states.shape, dtype=float)
    fuel[states == FOREST_STATE] = 30 * 24  # 1 month/km2
    fuel[states == CHAPARRAL_STATE] = 3 * 24  # 3 days/km2
    fuel[states == FLAMMABLE_SCRUBLAND_STATE] = 6  # 6 hours/km2

    fuel_steps = np.ceil(fuel * ctx.cell_km ** 2 / ctx.timestep_hours)

    return fuel_steps.astype(np.int16)

def get_fire_spread_probability(states: np.ndarray, burning_neighbours: np.ndarray, ctx: Context) -> np.ndarray:
    spread_rates = np.zeros(states.shape, dtype=float)

    # These values are arbitrary and can be changed to affect how quickly fire will spread
    spread_rates[states == FOREST_STATE] = 0.035
    spread_rates[states == CHAPARRAL_STATE] = 0.1
    spread_rates[states == FLAMMABLE_SCRUBLAND_STATE] = 0.4

    spread = spread_rates * burning_neighbours / ctx.cell_km

    return 1 - np.exp(-spread * ctx.timestep_hours)


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

    ps = get_fire_spread_probability(grid, neighbourcounts[BURNING_STATE], ctx)
    igniting = np.random.random(grid.shape) < ps

    ctx.fuel_grid[igniting] = get_fuel_timesteps(grid[igniting], ctx)
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
