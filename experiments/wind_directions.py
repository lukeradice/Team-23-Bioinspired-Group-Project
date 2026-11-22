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


WIND_SPEED = 5.0

RUNS_PER_DIRECTION = 20

MAX_HOURS = 500

FIRE_STARTS = {
    'power_plant': (0, 0),  # top left corner
    'incinerator': (0, 199),  # top right corner
}

WIND_DIRECTIONS = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}

RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'results')


def hours_to_reach_town(fire_start, wind_from_deg, wind_speed, seed):
    config = ffm.setup([])
    config.fill_in_defaults()

    config.initial_grid = ffm.build_terrain_grid(config.grid_dims)
    config.initial_grid[fire_start] = ffm.BURNING_STATE

    ffm.Context.wind_from_deg = wind_from_deg
    ffm.Context.wind_speed = wind_speed
    ffm.Context.rng = np.random.default_rng(seed)

    ctx = ffm.Context.from_config(config)
    ctx.rng = ffm.Context.rng
    grid = Grid2D(config, (ffm.transition_function, ctx))

    # Step manually, grid.run() would open a progress window
    max_steps = int(MAX_HOURS / ctx.timestep_hours)
    while ctx.town_arrival_step is None and ctx.step < max_steps:
        grid.step()

    if ctx.town_arrival_step is None:
        return None
    return ctx.town_arrival_step * ctx.timestep_hours


def summarise(hours):
    """Returns (mean, standard deviation, min, max, runs that reached the town)."""
    reached = [h for h in hours if h is not None]
    if not reached:
        return None, None, None, None, 0
    return np.mean(reached), np.std(reached), min(reached), max(reached), len(reached)


def save_csv(path, start_name, results):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fire_start', 'wind_from', 'wind_speed_ms', 'seed', 'hours_to_town'])
        for label, hours in results.items():
            speed = 0.0 if label == 'No wind' else WIND_SPEED
            for seed, h in enumerate(hours):
                writer.writerow([start_name, label, speed, seed, '' if h is None else round(h, 2)])


def save_chart(path, start_name, results):
    labels = list(results.keys())
    stats = [summarise(results[label]) for label in labels]
    means = [s[0] if s[0] is not None else 0 for s in stats]
    stds = [s[1] if s[1] is not None else 0 for s in stats]

    colours = ['#898781' if label == 'No wind' else '#2a78d6' for label in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, width=0.5, color=colours, zorder=2)
    ax.errorbar(labels, means, yerr=stds, fmt='none', ecolor='#52514e', elinewidth=1.5, capsize=4, zorder=3)

    top = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, top * 1.15)
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
    fire_start = FIRE_STARTS[start_name]

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
