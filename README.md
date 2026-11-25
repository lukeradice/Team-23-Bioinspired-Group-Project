# CAPyLE
CAPyLE is a cross-platform teaching tool designed and built as part of a final year computer science project. It aims to aid the teaching of cellular automata and how they can be used to model natural systems.

It is written completely in python with minimal dependencies.

![CAPyLE Screenshot on macOS](http://pjworsley.github.io/capyle/sample.png)

## Installation
The installation guide can be found on the [CAPyLE webpages](http://pjworsley.github.io/capyle/installationguide.html)

## Usage
Detailed usage can be found on the [CAPyLE webpages](http://pjworsley.github.io/capyle/).

See below for a quickstart guide:

1. `git clone https://github.com/pjworsley/capyle.git [target-directory]`
2. `cd [target-directory]`
3. Execute main.py either by:
    * `run.bat` / `run.sh`
    * `python main.py`
2. Use the menu bar to select File -> Open. This will open in the folder `./ca_descriptions`.
3. Open one of the example files;
  - `wolframs_1d.py` is Wolfram's elementary 1D automata
  - `gol_2d.py` is Conway's 2D game of life
4. The main GUI elements will now load, feel free to customise the CA parameters on the left hand panel
5. Run the CA with your parameters by clicking the bottom left button 'Apply configuration & run CA'
6. The progress bar will appear as the CA is run
7. After the CA has been run, use the playback controls at the top and the slider at the bottom to run through the simulation.
8. You may save an image of the currently displayed output using the 'Take screenshot' button

## Fire spread experiments

Run these commands from the repository root with NumPy and Matplotlib installed. The live viewer also needs a graphical display.

**Watch the fire spread.** This replays the settings used in the [example video](experiments/results/fire-spread-0p4h.mp4). Use the playback buttons or frame slider to inspect how many neighbours the fire spread through in one frame.

```sh
python experiments/fire_spread_visualiser.py --seed 67 --timestep 0.4 --frames 155
```

**Compare town arrival across timesteps.** This runs the current model on flat chaparral with the incinerator as the source. It uses the same 50 seeds at each of eight timesteps and writes a CSV and a first-visible-frame plot to `experiments/results/`.

```sh
python experiments/timestep_arrivals.py incinerator
```

For a quick run, append `--runs 3 --timesteps 0.2 0.4`. Both scripts accept `--help`. The saved [comparison graph](experiments/results/town-arrival.png) and [per-seed data](experiments/results/frame_time_comparison.csv) also include the original model.

## Acknowledgements
Special thanks to [Dr Dawn Walker](http://staffwww.dcs.shef.ac.uk/people/D.Walker/) for proposing and supervising this project.

Also thanks to the COM2005 2016/2017 cohort for being the guinea-pigs!

## Licence
CAPyLE is licensed under a BSD licence, the terms of which can be found in the LICENCE file.

Copyright 2017 Peter Worsley
