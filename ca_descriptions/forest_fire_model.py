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

from capyle.ca import Grid2D, CAConfig
import capyle.utils as utils
import numpy as np
from dataclasses import dataclass
from math import ceil
from typing import Optional, Sequence, Tuple

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


NUMBER_OF_HILLS = 8


def build_height_grid(grid_dims: Tuple[int, int], seed: int = 42) -> np.ndarray:
    """
    Builds a height map in metres made of a few cone-shaped hills.

    Each hill slopes down linearly from its peak, so high and low ground is
    grouped together and changes gradually instead of jumping between cells.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.indices(grid_dims)
    heights = np.zeros(grid_dims)

    for _ in range(NUMBER_OF_HILLS):
        peak_row = rng.uniform(0, grid_dims[0])
        peak_col = rng.uniform(0, grid_dims[1])
        peak_height = rng.uniform(100, 400)  # metres
        radius = rng.uniform(20, 60)  # cells (5-15km)

        distance = np.hypot(rows - peak_row, cols - peak_col)
        heights += peak_height * np.maximum(0, 1 - distance / radius)

    heights.setflags(write=False)
    return heights


@dataclass
class Context:
    # --- Non-default arguments must be first ---
    terrain_grid: np.ndarray
    # How long each cell should burn for in hours
    burn_durations_hours: np.ndarray
    # When each cell ignited (or infinite if it hasn't ignited yet)
    ignited_at_hours: np.ndarray
    # The time it takes for fire to spread from each cell to its neighbours in hours
    neighbour_spread_time_hours: np.ndarray
    # The timestep at which water should deploy
    water_deploy_timestep: int

    # --- Default arguments ---
    #  (Context configuration)

    cell_km = 0.25
    timestep_hours = 0.2
    # Direction the wind blows from in compass degrees (0 = northerly, 90 = easterly)
    wind_from_deg = 0.0
    # Wind speed in m/s
    wind_speed = 0.0
    # The current simulation step
    step = 0
    # The step at which it arrives at the town
    town_arrival_step: Optional[int] = None
    # Temperature weight determination
    ambient_temperature = 20
    # Hour the 12.5km^2 of water is dropped
    water_deploy_hours = -1 # put to -1 if you want to disable
    # Same order as CAPyLE's neighbourstates: NW, N, NE, W, E, SW, S, SE
    # Each offset locates the source neighbour relative to a destination cell
    neighbour_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # Define random seed
    seed = 67
    # Values in km/h
    terrain_spread_rates = {
        # <0.3km/h - Natural Resources Canada, Fire Behaviour.
        FOREST_STATE: 0.2,
        # 0.79–2.02km/h - Stephens et al. (2008), Measuring the Rate of Spread of Chaparral Prescribed Fires in Northern California.
        CHAPARRAL_STATE: 1.405,  # Mean of the range
        # ~3km/h - Cruz & Alexander (2019)
        FLAMMABLE_SCRUBLAND_STATE: 3.0,
    }
    # Fraction by which travel time can vary around distance / weighted spread speed.
    # Larger variations let the fire reach town earlier by following faster routes between cells.
    spread_time_variation = 0.1
    # Burn times independent of cell area in hours
    max_burn_times = {
        FOREST_STATE: 30 * 24, # 1 month
        CHAPARRAL_STATE: 3 * 24, # 3 days
        FLAMMABLE_SCRUBLAND_STATE: 6 # 6 hours
    }
    # Burn time for a square will be at least 80% of the maximum in max_burn_times
    min_burn_time_random_multiplier = 0.8

    @classmethod
    def from_config(cls, config: CAConfig):
        """Create a new context with defaults inferred from the simulation's config."""

        # Create new random generator
        rng = np.random.default_rng(cls.seed)

        # Create a read-only terrain grid
        terrain_grid = config.initial_grid.copy()
        terrain_grid.setflags(write=False)

        # Get how long each cell should burn for in hours
        burn_durations_hours = cls.get_burn_durations_hours(terrain_grid, rng)
        burn_durations_hours.setflags(write=False)
        
        # Create a grid which will contain when each cell ignited (in hours from simulation start)
        # Sources ignite at hour zero. Infinity means it hasn't ignited yet.
        initial_fire_sources = terrain_grid == BURNING_STATE
        ignited_at_hours = np.where(initial_fire_sources, 0.0, np.inf)

        # Wind changes the spread speed according to the direction the fire travels.
        wind_weights = cls.get_wind_weights()
        wind_weights.setflags(write=False)

        # Each degree above or below the reference adjusts spread speed by 2% of the base speed.
        TEMPERATRE_REF = 20 # no temperature weighting at 20 degrees C
        temperature_weight = max(0, 1 + 0.02 * (cls.ambient_temperature - TEMPERATRE_REF))

        # For each cell, the slope from each neighbour determines its uphill or downhill multiplier.
        slope_weights = cls.get_slope_weights(config.heights)
        slope_weights.setflags(write=False)

        # Get the time it takes for fire to spread from each cell to its neighbours in hours
        neighbour_spread_time_hours = cls.get_neighbour_spread_times(
            terrain_grid,
            config.neighbourhood().neighbourhood,
            temperature_weight * wind_weights * slope_weights,
            rng
        )
        neighbour_spread_time_hours.setflags(write=False)

        # Get the timestep at which water should be deployed
        water_deploy_timestep = ceil(cls.water_deploy_hours / cls.timestep_hours)

        return cls(
            terrain_grid=terrain_grid,
            burn_durations_hours=burn_durations_hours,
            ignited_at_hours=ignited_at_hours,
            neighbour_spread_time_hours=neighbour_spread_time_hours,
            water_deploy_timestep=water_deploy_timestep,
        )

    @classmethod
    def get_burn_durations_hours(cls, terrain_grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Burn durations in hours, independent of cell area.
        burn_durations = np.zeros(terrain_grid.shape, dtype=float)
        for state, max_burn_time in cls.max_burn_times.items():
            cells = terrain_grid == state

            # Introduce some randomness for the amount of days a terrain square will burn for
            random_multipliers = rng.uniform(cls.min_burn_time_random_multiplier, 1.0, size=cells.sum())
            burn_durations[cells] = max_burn_time * random_multipliers

        burn_durations[terrain_grid == BURNING_STATE] = np.inf  # Initial burning cells have infinite fuel
        return burn_durations

    @classmethod
    def get_wind_weights(cls) -> np.ndarray:
        # Wind constants from alexandridis (2008) - wind speed in m/s
        WIND_C1 = 0.045
        WIND_C2 = 0.131

        # Compass bearing (degrees clockwise from north) of each neighbour
        # in the same order as CAPyLE's neighbourstates: NW, N, NE, W, E, SW, S, SE
        NEIGHBOUR_BEARINGS = np.array([315, 0, 45, 270, 90, 225, 180, 135])

        # angle between the wind and the direction fire travels from each neighbour
        angles = np.radians(NEIGHBOUR_BEARINGS - cls.wind_from_deg)

        wind_weights = np.exp(WIND_C1 * cls.wind_speed) * np.exp(WIND_C2 * cls.wind_speed * (np.cos(angles) - 1))
        
        # Shape (8, 1, 1) applies each direction's wind weight to every row and column.
        return wind_weights[:, None, None]

    @classmethod
    def get_slope_weights(cls, heights: np.ndarray) -> np.ndarray:
        """Returns a spread multiplier for every neighbour of every cell, shape (8, rows, cols).

        Fire climbing from a lower neighbour up to a cell gets a weight above 1,
        fire running downhill gets a weight below 1. Flat ground gives 1.
        """
        # Slope constant from alexandridis (2008), per degree of slope
        SLOPE_A = 0.078

        rows, cols = heights.shape
        padded = np.pad(heights, 1, mode='edge')

        weights = []
        for dr, dc in cls.neighbour_offsets:
            neighbour_heights = padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
            distance_m = np.hypot(dr, dc) * cls.cell_km * 1000  # diagonals are further away
            slope_deg = np.degrees(np.arctan((heights - neighbour_heights) / distance_m))
            weights.append(np.exp(SLOPE_A * slope_deg))

        return np.array(weights)

    @classmethod
    def get_neighbour_spread_times(
        cls, terrain_grid: np.ndarray, neighbourhood: np.ndarray,
        spread_multiplier: np.ndarray, rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw once per run the time in hours for fire to spread from each neighbour into each cell."""

        spread_rates = np.zeros(terrain_grid.shape, dtype=float)
        for state, rate in cls.terrain_spread_rates.items():
            spread_rates[terrain_grid == state] = rate

        # Bound the variation so the fastest route cannot string together near-instant fuel crossings.
        # Draw once per neighbour and direction so changing the timestep reuses the same fire.
        random_spread_time_factors = rng.uniform(1 - cls.spread_time_variation, 1 + cls.spread_time_variation,
                                          size=spread_multiplier.shape)

        # Infinity means this neighbour cannot ignite the cell. Only positive rates overwrite it.
        neighbour_spread_time_hours = np.full(spread_multiplier.shape, np.inf)

        # Determine for each cell for each of the 8 directions a spread time for that direction's neighbouring cell 
        # to spread to it once ignited.
        for i, offset in enumerate(cls.neighbour_offsets):
            dr, dc = offset
            if neighbourhood[1 + dr, 1 + dc]:
                distance_to_neighbour = cls.cell_km * np.hypot(dr, dc)

                # Weighted speed (km/h) divided by centre-to-centre distance (km) gives the hourly rate of spread to that neighbour.
                spread_rate_per_hour = spread_rates * spread_multiplier[i] / distance_to_neighbour

                # The reciprocal of spread rate per hour gives spread time, multiply it by the random spread factor to introduce noise.
                # For example, 0.25 km at 1 km/h takes 15 minutes on average.
                np.divide(random_spread_time_factors[i], spread_rate_per_hour, out=neighbour_spread_time_hours[i], where=spread_rate_per_hour > 0)
            neighbour_spread_time_hours[i, terrain_grid == TOWN_STATE] = 0  # Town ignites on first contact.

        return neighbour_spread_time_hours


def setup(args: Sequence[str]) -> CAConfig:
    """Set up the config object used to interact with the GUI"""
    # No args means a headless run, so there is no GUI config file to load
    config = utils.load(args[0]) if args else CAConfig(this_file_loc)
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

    if config.initial_grid is None:
        config.initial_grid = build_terrain_grid(config.grid_dims)

    # Stored on the config so the GUI can show it with the "Show height map" button
    config.heights = build_height_grid(config.grid_dims)

    INCERATOR_LOCATION = (0, config.grid_dims[1] - 1) # top right corner
    config.initial_grid[INCERATOR_LOCATION] = BURNING_STATE

    # POWER_PLANT_LOCATION = (0, 0)
    # config.initial_grid[POWER_PLANT_LOCATION] = BURNING_STATE

    # Stop fire crossing an edge and reappearing on the opposite side.
    config.wrap = False

    # ----------------------------------------------------------------------

    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.town_arrival_step = None  # Show the town arrival button before the first run.
        config.save()
        sys.exit()
    return config

CellIndices = Tuple[np.ndarray, np.ndarray]  # Matching arrays of row and column indices

def get_fire_front(grid: np.ndarray, can_ignite: np.ndarray, ctx: Context) -> CellIndices:
    """Find burning cells with at least one neighbour that can ignite."""
    rows, cols = grid.shape

    # A one-cell False border lets edge and corner cells check all eight neighbours as if off-grid cells had no fuel.
    padded = np.pad(can_ignite, 1, constant_values=False)

    # A cell is true if one of a its neighbours can ignite.
    has_ignitable_neighbour = np.zeros_like(grid, dtype=bool)
    for dr, dc in ctx.neighbour_offsets:
        has_ignitable_neighbour |= padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]

    # Return the row and column of each burning cell along that boundary.
    return np.nonzero((grid == BURNING_STATE) & has_ignitable_neighbour)


def get_neighbour_indices_in_direction(
    fire_sources: CellIndices, offset: Tuple[int, int], grid_shape: Tuple[int, int],
) -> Tuple[CellIndices, np.ndarray]:
    """Find the next cell in one direction from each burning cell.

    Return the row and column of each neighbour that exists, plus one True/False value
    per burning cell to indicate if there is a valid neighbour at the offset. 
    """
    fire_source_rows, fire_source_cols = fire_sources
    dr, dc = offset
    neighbour_rows, neighbour_cols = fire_source_rows - dr, fire_source_cols - dc
    rows, cols = grid_shape

    # ensure it only gets neighbours within the grid
    cells_with_neighbour = ((neighbour_rows >= 0) & 
                            (neighbour_rows < rows) & 
                            (neighbour_cols >= 0) & 
                            (neighbour_cols < cols))

    return ((neighbour_rows[cells_with_neighbour], neighbour_cols[cells_with_neighbour]), 
            cells_with_neighbour)


def get_newly_lit_ignited_at_hours(
    grid: np.ndarray, ctx: Context, current_step_hours: float,
) -> np.ndarray:
    """Return the earliest ignition hour for each newly reached cell, or infinity if unreached."""

    can_ignite = (grid != BURNING_STATE) & (grid != BURNT_STATE) & (grid != LAKE_STATE)
    # Infinity means no route has reached this cell yet, so the first finite arrival wins.
    newly_lit_ignited_at_hours = np.full(grid.shape, np.inf)

    # Start at the fire front. A cell already burning may have ignited several frames ago.
    fire_sources = get_fire_front(grid, can_ignite, ctx)
    fire_source_ignited_at_hours = ctx.ignited_at_hours[fire_sources]

    # Follow the fire one neighbour at a time until no earlier ignition can cause ignition in this timeframe.
    while fire_source_ignited_at_hours.size > 0:
        # The source must still be burning when its neighbour ignites.
        source_burnout_hours = fire_source_ignited_at_hours + ctx.burn_durations_hours[fire_sources]
        # Track which cells gain an earlier arrival during this pass through the neighbours.
        reached_earlier = np.zeros_like(grid, dtype=bool)

        for direction_index, offset in enumerate(ctx.neighbour_offsets):
            neighbour_indices, has_neighbour = get_neighbour_indices_in_direction(fire_sources, offset, grid.shape)
            # Each stored duration is indexed by the destination and the direction its fire comes from.
            spread_time_hours = ctx.neighbour_spread_time_hours[direction_index][neighbour_indices]

            # Calculate possible arrival time of ignition to the neighbours.
            # For example: a source lit at minute 1 with a 1.4-minute spread time can ignite its neighbour 
            # at minute 2.4.
            candidate_ignited_at_hours = fire_source_ignited_at_hours[has_neighbour] + spread_time_hours


            # The neighbours must ignite by the end of the current time step and before the source burns out.
            finishes_by_both_deadlines = ((candidate_ignited_at_hours <= current_step_hours) & 
                                          (candidate_ignited_at_hours <= source_burnout_hours[has_neighbour]))

            # Store the fastest valid ignition route (meets both deadlines) for the ignitable neighbours.
            earlier_route = candidate_ignited_at_hours < newly_lit_ignited_at_hours[neighbour_indices]
            valid_ignition_route = can_ignite[neighbour_indices] & finishes_by_both_deadlines
            is_earliest_route_so_far = valid_ignition_route & earlier_route

            # Record these ignition times without changing the live grid while other routes are still being checked.
            neighbour_rows, neighbour_cols = neighbour_indices
            reached_neighbours = (neighbour_rows[is_earliest_route_so_far], neighbour_cols[is_earliest_route_so_far])
            newly_lit_ignited_at_hours[reached_neighbours] = candidate_ignited_at_hours[is_earliest_route_so_far]
            reached_earlier[reached_neighbours] = True

        # An earlier arrival can also bring fire to the next neighbour sooner. 
        fire_sources = np.nonzero(reached_earlier)
        fire_source_ignited_at_hours = newly_lit_ignited_at_hours[fire_sources]

    return newly_lit_ignited_at_hours


def transition_function(
    grid: np.ndarray, neighbourstates: np.ndarray, neighbourcounts: np.ndarray, ctx: Context,
) -> np.ndarray:
    """Function to apply the transition rules and return the new grid"""

    ctx.step += 1
    current_step_hours = ctx.step * ctx.timestep_hours

    # --- Spread fires ---

    # One cell per frame only works if the distance a fire spreads is <= the width of a cell
    # We add up the time fire takes to spread from one cell to the next.
    # A newly lit cell can light its own neighbour if there is still time left in the frame.
    newly_lit_ignited_at_hours = get_newly_lit_ignited_at_hours(grid, ctx, current_step_hours)
    newly_lit = np.isfinite(newly_lit_ignited_at_hours)
    ctx.ignited_at_hours[newly_lit] = newly_lit_ignited_at_hours[newly_lit]
    grid[newly_lit] = BURNING_STATE

    # -- Consume fuel/burn out fires --

    burnt = ctx.ignited_at_hours + ctx.burn_durations_hours <= current_step_hours
    # Keep newly ignited town cells visible for one frame.
    town_contact = newly_lit & (ctx.terrain_grid == TOWN_STATE)
    grid[(grid == BURNING_STATE) & burnt & ~town_contact] = BURNT_STATE

    #  --- Deploy water ---

    if ctx.step == ctx.water_deploy_timestep:
        grid = deploy_water(grid)

    # --- Check if fire spread to the town ---

    town_lit = np.any((grid == BURNING_STATE) & (ctx.terrain_grid == TOWN_STATE))
    if ctx.town_arrival_step is None and town_lit:
        ctx.town_arrival_step = ctx.step

    return grid

def deploy_water(grid):
    '''Function to return the grid with 12.5km^2 of the cells replaced with the lake state to simulate water being dropped'''

    # Line 1
    grid[0:100, 100] = LAKE_STATE

    # Line 2 
    grid[100, 100:199] = LAKE_STATE

    return grid

def main():
    """ Main function that sets up, runs and saves CA"""
    # Get the config object from set up
    config = setup(sys.argv[1:])

    # Create grid object using parameters from config + transition function
    ctx = Context.from_config(config)
    grid = Grid2D(config, (transition_function, ctx))

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # Save simple values so the GUI can display the result on demand.
    config.town_arrival_step = ctx.town_arrival_step
    config.timestep_hours = ctx.timestep_hours

    # Save updated config to file
    config.save()
    # Save timeline to file
    utils.save(timeline, config.timeline_path)


def run_headless(max_steps=10_000):
    """Run without the GUI until the fire reaches the town, then print when it did."""
    config = setup([])
    config.fill_in_defaults()
    ctx = Context.from_config(config)
    grid = Grid2D(config, (transition_function, ctx))

    # Step manually, grid.run() would open a progress window
    while ctx.town_arrival_step is None and ctx.step < max_steps:
        grid.step()

    if ctx.town_arrival_step is None:
        print(f"Fire did not reach the town within {max_steps} steps")
    else:
        hours = ctx.town_arrival_step * ctx.timestep_hours
        print(f"Fire reached the town at step {ctx.town_arrival_step} ({hours:.1f} hours)")


if __name__ == "__main__":
    # The GUI always passes a config path, so no args means headless
    if len(sys.argv) > 1:
        main()
    else:
        run_headless()
