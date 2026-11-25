"""Watch the fire model live. Run with --help for timestep, seed and frame options."""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Button, Slider
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ca_descriptions import forest_fire_model as fire


def get_spread_steps(previous: np.ndarray, newly_lit: np.ndarray, ctx: fire.Context) -> np.ndarray:
    """Count neighbour steps from a previous flame along the recorded ignition route.

    Zero marks cells outside this frame's spread. If equally fast routes reach a
    cell, visiting one neighbour at a time gives the shorter route in cell steps.
    """
    steps = np.zeros(previous.shape, dtype=np.int32)
    sources = fire.get_fire_front(previous, newly_lit, ctx)
    depth = 0
    while sources[0].size:
        depth += 1
        reached = np.zeros_like(newly_lit)
        source_hours = ctx.ignited_at_hours[sources]
        burnout_hours = source_hours + ctx.burn_durations_hours[sources]
        for direction, offset in enumerate(ctx.neighbour_offsets):
            dest, valid = fire.get_neighbour_indices_in_direction(sources, offset, previous.shape)
            arrival = source_hours[valid] + ctx.neighbour_spread_time_hours[direction][dest]
            # Only trace a spread that explains the actual ignition and beat the source's burnout.
            on_route = (arrival == ctx.ignited_at_hours[dest]) & (arrival <= burnout_hours[valid])
            reached[dest] |= newly_lit[dest] & (steps[dest] == 0) & on_route
        steps[reached] = depth
        sources = np.nonzero(reached)
    return steps


class FireSpreadViewer:
    def __init__(self, config, ctx: fire.Context, frames: int):
        self.ctx = ctx
        self.grid = fire.Grid2D(config, (fire.transition_function, ctx))
        self.frames = frames
        self.index = 0
        self.playing = False
        initial = self.grid.grid.astype(np.uint8)
        self.history = [(initial, np.zeros_like(initial), 0)]

        self.fig, (terrain, spread) = plt.subplots(1, 2, figsize=(11, 6))
        self.fig.subplots_adjust(bottom=0.28, top=0.87)
        self.title = self.fig.suptitle("")
        terrain.set_title("Fire and terrain")
        self.terrain = terrain.imshow(initial, cmap=ListedColormap(config.state_colors),
                                      vmin=0, vmax=6, interpolation="nearest")
        terrain_names = ["Burning", "Burnt", "Chaparral", "Forest", "Water", "Scrubland", "Town"]
        terrain.legend(handles=[Patch(color=c, label=n) for c, n in zip(config.state_colors, terrain_names)],
                       loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4, fontsize="small")
        spread.set_title("Neighbour steps within this frame")
        colours = ["#eeeeee", "#3665a3", "#ffd166", "#f99345", "#d84f34", "#8e1847"]
        self.spread = spread.imshow(self.history[0][1], cmap=ListedColormap(colours),
                                    vmin=0, vmax=5, interpolation="nearest")
        labels = ["Already burning", "1 step", "2 steps", "3 steps", "4+ steps"]
        spread.legend(handles=[Patch(color=c, label=n) for c, n in zip(colours[1:], labels)],
                      loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3, fontsize="small")
        for axis in (terrain, spread):
            axis.set_axis_off()

        self.slider = Slider(self.fig.add_axes([0.16, 0.14, 0.68, 0.035]), "Frame", 0, 1,
                             valinit=0, valstep=1, valfmt="%d")
        self.slider.on_changed(self.seek)
        self.back = Button(self.fig.add_axes([0.28, 0.045, 0.13, 0.055]), "Step back")
        self.play = Button(self.fig.add_axes([0.435, 0.045, 0.13, 0.055]), "Play")
        self.next = Button(self.fig.add_axes([0.59, 0.045, 0.13, 0.055]), "Step forward")
        self.back.on_clicked(lambda event: self.seek(max(0, self.index - 1)))
        self.play.on_clicked(lambda event: self.set_playing(not self.playing))
        self.next.on_clicked(lambda event: self.step_forward())
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.advance)
        self.fig.canvas.mpl_connect("close_event", lambda event: self.timer.stop())
        self.show_frame(0)

    def advance(self):
        # Replay cached frames first. Only the newest frame advances the live model.
        if self.index == self.ctx.step and self.index < self.frames:
            previous = self.history[-1][0]
            unreached = np.isinf(self.ctx.ignited_at_hours)
            self.grid.step()
            newly_lit = unreached & np.isfinite(self.ctx.ignited_at_hours)
            steps = get_spread_steps(previous, newly_lit, self.ctx)
            # Store compact display values: 0 background, 1 old flame, 2–5 new spread.
            overlay = (previous == fire.BURNING_STATE).astype(np.uint8)
            overlay[newly_lit] = 1 + np.minimum(steps[newly_lit], 4)
            self.history.append((self.grid.grid.astype(np.uint8), overlay, int(steps.max())))
            self.slider.valmax = self.ctx.step
            self.slider.ax.set_xlim(0, self.ctx.step)
        self.show_frame(min(self.index + 1, self.ctx.step))
        if self.index == self.frames:
            self.set_playing(False)

    def step_forward(self):
        self.set_playing(False)
        self.advance()

    def set_playing(self, playing: bool):
        self.playing = playing and self.index < self.frames
        self.play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def seek(self, index: int):
        self.set_playing(False)
        self.show_frame(min(int(index), self.ctx.step))

    def show_frame(self, index: int):
        self.index = index
        grid, overlay, depth = self.history[index]
        self.terrain.set_data(grid)
        self.spread.set_data(overlay)
        self.title.set_text(f"Frame {index}   |   {index * self.ctx.timestep_hours:g} hours"
                            f"   |   Longest spread this frame: {depth} neighbour steps")
        # Moving the slider during playback must not trigger a manual seek.
        self.slider.eventson = False
        self.slider.set_val(index)
        self.slider.eventson = True
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestep", type=float, default=fire.Context.timestep_hours, help="Hours per frame")
    parser.add_argument("--seed", type=int, default=fire.Context.seed)
    parser.add_argument("--frames", type=int, help="Number of frames to simulate")
    args = parser.parse_args()
    if not np.isfinite(args.timestep) or args.timestep <= 0 or args.seed < 0 or (args.frames is not None and args.frames < 1):
        parser.error("use a positive finite timestep, a non-negative seed and at least one frame")

    class Run(fire.Context):
        timestep_hours = args.timestep
        seed = args.seed

    config = fire.setup([])
    config.fill_in_defaults()
    viewer = FireSpreadViewer(config, Run.from_config(config), args.frames or config.num_generations)
    viewer.set_playing(True)
    plt.show()


if __name__ == "__main__":
    main()
