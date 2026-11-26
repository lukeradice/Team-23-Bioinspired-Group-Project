"""
Time for the fire to reach the town for each wind direction (+ no wind)

python experiments/wind_directions.py              -> power plant
python experiments/wind_directions.py incinerator  -> incinerator

Results go in experiments/results. For other tests (water, hills, temp etc)
change the Run class in hours_to_reach_town
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# make the model file importable from here
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(EXPERIMENTS_DIR), 'ca_descriptions'))
import forest_fire_model as ffm
from capyle.ca import Grid2D

BURNING_STATE = 0

WIND_SPEED = 10.0  # m/s

RUNS_PER_DIRECTION = 20  # same seeds for every direction

MAX_HOURS = 500  # give up after this, e.g. if water stops the fire

FIRE_STARTS = {
    'power_plant': (0, 0),  # top left corner
    'incinerator': (0, 199),  # top right corner
}

# where the wind comes from, compass degrees
WIND_DIRECTIONS = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}

RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'results')


def hours_to_reach_town(fire_start, wind_deg, wind_ms, run_seed):
    """One run, returns hours until the town catches fire or None if it never does"""

    # override Context settings for this run here
    class Run(ffm.Context):
        seed = run_seed
        wind_from_deg = wind_deg
        wind_speed = wind_ms

    config = ffm.setup([])
    config.fill_in_defaults()

    # setup lights the incinerator so redo the grid with just our fire
    config.initial_grid = ffm.build_terrain_grid(config.grid_dims)
    config.initial_grid[fire_start] = ffm.BURNING_STATE

    ctx = Run.from_config(config)
    grid = Grid2D(config, (ffm.transition_function, ctx))
    town = ctx.terrain_grid == ffm.TOWN_STATE

    # Step manually, grid.run() would open a progress window
    max_steps = int(MAX_HOURS / ctx.timestep_hours)
    while ctx.town_arrival_step is None and ctx.step < max_steps and ctx.cells_burning:
        grid.step()

    if ctx.town_arrival_step is None:
        return None
    # exact ignition time, not the step time
    return float(ctx.ignited_at_hours[town].min())


def summarise(hours):
    # mean, sd, min, max, how many reached the town
    reached = [h for h in hours if h is not None]
    if not reached:
        return None, None, None, None, 0
    return np.mean(reached), np.std(reached), min(reached), max(reached), len(reached)


def save_csv(path, start_name, results):
    # one row per run
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fire_start', 'wind_from', 'wind_speed_ms', 'seed', 'hours_to_town'])
        for label, hours in results.items():
            speed = 0.0 if label == 'No wind' else WIND_SPEED
            for seed, h in enumerate(hours):
                writer.writerow([start_name, label, speed, seed, '' if h is None else round(h, 2)])


def save_chart(path, start_name, results):
    # mean +- 1 sd for each direction
    labels = list(results.keys())
    stats = [summarise(results[label]) for label in labels]
    means = [s[0] if s[0] is not None else 0 for s in stats]
    stds = [s[1] if s[1] is not None else 0 for s in stats]

    # no wind in grey
    colours = ['#898781' if label == 'No wind' else '#2a78d6' for label in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, width=0.5, color=colours, zorder=2)
    ax.errorbar(labels, means, yerr=stds, fmt='none', ecolor='#52514e', elinewidth=1.5, capsize=4, zorder=3)

    # leave room above the bars for the labels
    top = max(m + s for m, s in zip(means, stds))
    y_max = max(top*1.15, 1.0)
    ax.set_ylim(0, y_max)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + top * 0.02, '{:.1f}'.format(mean),
                ha='center', va='bottom', fontsize=9)

    start_title = start_name.replace('_', ' ')
    ax.set_title('Hours for fire to reach the town from the {} ({} m/s wind, {} runs each)'.format(
        start_title, WIND_SPEED, RUNS_PER_DIRECTION), fontsize=11, loc='left', pad=12)
    ax.set_xlabel('Direction the wind blows from')
    ax.set_ylabel('Hours to reach the town (mean ± 1 SD)')
    ax.yaxis.grid(True, color='#e1e0d9', zorder=0)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    start_name = sys.argv[1] if len(sys.argv) > 1 else 'power_plant'
    print(f"starting fire experiments from {start_name}")
    fire_start = FIRE_STARTS[start_name]

    # direction -> hours for each seed
    results = {}
    for label, degrees in WIND_DIRECTIONS.items():
        print('Wind from {:2s} ...'.format(label))
        results[label] = [hours_to_reach_town(fire_start, degrees, WIND_SPEED, seed)
                          for seed in range(RUNS_PER_DIRECTION)]

    print('No wind ...')
    results['No wind'] = [hours_to_reach_town(fire_start, 0.0, 0.0, seed)
                          for seed in range(RUNS_PER_DIRECTION)]

    # Summary table
    print()
    print('Fire start: {}, wind speed {} m/s, {} runs each'.format(start_name, WIND_SPEED, RUNS_PER_DIRECTION))
    print('{:8s} {:>9s} {:>7s} {:>7s} {:>7s} {:>8s}'.format('Wind', 'Mean (h)', 'SD', 'Min', 'Max', 'Reached'))
    for label, hours in results.items():
        mean, std, low, high, reached = summarise(hours)
        if reached == 0:
            print('{:8s} {:>9s}'.format(label, 'never'))
        else:
            print('{:8s} {:9.1f} {:7.1f} {:7.1f} {:7.1f} {:5d}/{}'.format(
                label, mean, std, low, high, reached, len(hours)))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'wind_directions_{}.csv'.format(start_name))
    chart_path = os.path.join(RESULTS_DIR, 'wind_directions_{}.png'.format(start_name))
    save_csv(csv_path, start_name, results)
    save_chart(chart_path, start_name, results)
    print()
    print('Saved', csv_path)
    print('Saved', chart_path)


if __name__ == '__main__':
    main()
