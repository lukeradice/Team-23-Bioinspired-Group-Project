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

from capyle.ca import Grid2D, Neighbourhood, randomise2d
import capyle.utils as utils
import numpy as np

BURNING_STATE = 0
BURNT_STATE = 1
CHAPARRAL_STATE = 2
FOREST_STATE = 3
LAKE_STATE = 4
FLAMMABLE_SCRUBLAND_STATE = 5
TOWN_STATE = 6

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
    config.grid_dims = (200,200)

    config.initial_grid = get_initial_grid()        
    # ----------------------------------------------------------------------

    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()
    return config

def get_initial_grid():
    initial_grid = np.zeros((200, 200), dtype=int)

    # Chaparral
    initial_grid[:, :] = CHAPARRAL_STATE

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

def transition_function(grid, neighbourstates, neighbourcounts):
    """Function to apply the transition rules
    and return the new grid"""
    # YOUR CODE HERE
    return grid


def main():
    """ Main function that sets up, runs and saves CA"""
    # Get the config object from set up
    config = setup(sys.argv[1:])

    # Create grid object using parameters from config + transition function
    grid = Grid2D(config, transition_function)

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # Save updated config to file
    config.save()
    # Save timeline to file
    utils.save(timeline, config.timeline_path)

if __name__ == "__main__":
    main()
