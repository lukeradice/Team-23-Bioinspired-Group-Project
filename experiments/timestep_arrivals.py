"""Compare town arrival across timesteps on flat chaparral with no wind or water drop.

Each timestep replays the same seeds. The first ignition hour should agree with
the finest timestep for each seed; the first visible frame can end later.
Write one CSV row per run and a chart under experiments/results by default.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import math
import os
from pathlib import Path
import sys
from typing import NamedTuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR.parent))
from ca_descriptions import forest_fire_model as fire
from capyle.ca import Grid2D


class Arrival(NamedTuple):
    frame: Optional[int]
    ignition_hours: Optional[float]
    visible_hours: Optional[float]


def run_trial(config, town, seed_value: int, dt: float, horizon: float) -> Arrival:
    class Trial(fire.Context):
        seed = seed_value
        timestep_hours = dt
        wind_speed = 0.0
        ambient_temperature = 20
        water_deploy_hours = -1

    ctx = Trial.from_config(config)
    grid = Grid2D(config, (fire.transition_function, ctx))
    # Run through the frame containing the common time horizon; censor by exact ignition time.
    for _ in range(math.ceil(horizon / dt)):
        grid.step()
        if ctx.town_arrival_step is not None:
            contact = float(ctx.ignited_at_hours[town].min())
            if contact <= horizon:
                return Arrival(ctx.step, contact, ctx.step * dt)
            break
    return Arrival(None, None, None)


def save_chart(path, start, steps, mean_visible_hours) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, mean_visible_hours, "s--", label="Speed-corrected model")
    ax.set(xscale="log", yscale="log", xlabel="Main timestep (hours)",
           ylabel="Mean time to first visible town fire (hours)",
           title=f"Town arrival from the {start.replace('_', ' ')} on flat chaparral")
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{dt:g}" for dt in steps])
    ax.set_xticks([], minor=True)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:g}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", nargs="?", default="incinerator", choices=("power_plant", "incinerator"))
    parser.add_argument("--timesteps", type=float, nargs="+", default=[0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625],
                        help="Main timestep sizes in hours")
    parser.add_argument("--runs", type=int, default=50, help="Number of paired seeds")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Maximum simultaneous trials (default: up to 4)")
    parser.add_argument("--max-hours", type=float, default=10_000, help="Common physical-time horizon")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_DIR / "results")
    args = parser.parse_args()
    steps = sorted(set(args.timesteps))
    if len(steps) < 2 or any(not math.isfinite(dt) or dt <= 0 for dt in steps):
        parser.error("provide at least two distinct positive finite timesteps")
    if args.runs < 1 or not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error("runs and max-hours must be positive and finite")
    workers = min(args.workers, args.runs)
    if workers < 1 or (sys.platform == "win32" and workers > 61):
        parser.error("use at least one worker and at most 61 simultaneous workers on Windows")

    config = fire.setup([])
    config.fill_in_defaults()
    # Keep the model town and its distance from the source. Remove hills, lakes and fuel variation by terrain.
    town = config.initial_grid == fire.TOWN_STATE
    config.initial_grid.fill(fire.CHAPARRAL_STATE)
    config.initial_grid[town] = fire.TOWN_STATE
    start_col = 0 if args.start == "power_plant" else config.grid_dims[1] - 1
    config.initial_grid[0, start_col] = fire.BURNING_STATE
    config.heights = np.zeros(config.grid_dims)

    results = {}
    seeds = range(args.runs)

    print(f"Using {workers} workers.")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for dt in steps:
            print(f"Running {args.runs} paired seeds at {dt} h per step ...", flush=True)
            trials = {pool.submit(run_trial, config, town, seed, dt, args.max_hours): seed for seed in seeds}
            by_seed = {}
            for trial in as_completed(trials):
                by_seed[trials[trial]] = trial.result()
                print("-", end="", flush=True)
            print()
            # Workers finish in any order; comparisons below pair the same seed at every timestep.
            results[dt] = [by_seed[seed] for seed in seeds]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"timestep_arrivals_{args.start}"
    failures = 0
    mean_visible_hours = []
    with stem.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["start", "seed", "timestep_hours", "horizon_hours", "first_frame", "ignition_hours",
                         "visible_hours", "ignition_difference_from_finest_hours", "check"])
        print("Step (h)   Reached   Mean ignition (h)   Mean visible (h)   Max paired difference (min)")
        for dt in steps:
            reached = []
            differences = []
            for seed, trial, reference in zip(seeds, results[dt], results[steps[0]]):
                difference = None
                if trial.frame is None and reference.frame is None:
                    check = "censored"
                elif trial.frame is None or reference.frame is None:
                    check = "reach mismatch"
                else:
                    difference = trial.ignition_hours - reference.ignition_hours
                    differences.append(abs(difference) * 60)
                    # The first visible frame must contain the exact town ignition.
                    in_frame = (trial.frame - 1) * dt < trial.ignition_hours <= trial.visible_hours
                    check = "pass" if abs(difference) <= 1e-9 and in_frame else "timing mismatch"
                if trial.frame is not None:
                    reached.append(trial)
                failures += check.endswith("mismatch")
                writer.writerow([args.start, seed, dt, args.max_hours, *trial, difference, check])
            exact = np.mean([r.ignition_hours for r in reached]) if reached else np.nan
            visible = np.mean([r.visible_hours for r in reached]) if reached else np.nan
            mean_visible_hours.append(visible)
            max_difference = max(differences, default=np.nan)
            print(f"{dt:8g}   {len(reached):3d}/{args.runs:<3d}   {exact:17.5f}   "
                  f"{visible:16.5f}   {max_difference:g}")

    save_chart(stem.with_suffix(".png"), args.start, steps, mean_visible_hours)
    print(f"Saved {stem}.csv and {stem}.png")
    if failures:
        sys.exit(f"{failures} paired checks failed; see CSV.")
    if all(r.frame is None for r in results[steps[0]]):
        sys.exit("No reference run reached town within the horizon; comparison is inconclusive.")


if __name__ == "__main__":
    main()
