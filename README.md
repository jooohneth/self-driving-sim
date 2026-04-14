# Self-driving car simulation using cnn

1. install python 3.11 and [uv](https://github.com/astral-sh/uv)
2. clone the repo and run `uv sync` to install all dependencies (handles tensorflow-macos and tensorflow-metal automatically for apple silicon)
3. put your simulator data in `data/` — it should contain `driving_log.csv` and an `IMG/` folder
4. train the model: `uv run python src/train.py` — saves the best checkpoint to `models/model.h5`
5. download the udacity term 1 simulator for mac from the [self-driving-car-sim repo](https://github.com/udacity/self-driving-car-sim) and unzip it
6. if the simulator is blocked: `xattr -cr path/to/beta_simulator_mac.app` then `chmod +x path/to/beta_simulator_mac.app/Contents/MacOS/beta_simulator_mac`
7. start the server: `uv run python src/test_simulation.py`
8. open the simulator, pick the lake track, select autonomous mode — the car should start driving on its own

### Note

The reason why we have commits that are like super close to the deadline is because we migrated from the original repo where we did the implementation, because something went wrong there and we decided it would be better to start from a clean slate and recommit all the code.
