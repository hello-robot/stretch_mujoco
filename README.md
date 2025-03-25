# Stretch Mujoco

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="./docs/stretch_mujoco.png" title="Stretch In Kitchen" width="100%">

This library provides the simulation stack for Stretch with [Mujoco](https://github.com/google-deepmind/mujoco).
Currently only Stretch 3 is fully supported with a position control interface for all arm/head/gripper joints and velocity control for base. Camera data with depth perception and camera parameters are provided. The library supports simuation with GUI or headless mode. Also, Stretch can be spawned in any Robocasa-provided kitchen environment.

## Getting Started

First, install [`uv`](https://docs.astral.sh/uv/#getting-started). Uv is a package manager that we'll use to run this project.

Then, clone this repo:

```
git clone https://github.com/hello-robot/stretch_mujoco --recurse-submodules
cd stretch_mujoco
```

> If you've already cloned the repo without `--recurse-submodules`, run `git submodule update --init` to pull the submodule.

Then, install this repo:

```
uv venv
uv pip install -e .
```

Lastly, run the simulation:

```
uv run launch_sim.py
```

To exit, press `Ctrl+C` in the terminal.

<p>
    <img src="./docs/camera_streams.png" title="Camera Streams" height="250px">
    <img src="./docs/stretch3_in_mujoco.png" title="Camera Streams" height="250px">
</p>

## Try Example Scripts

[Keyboard teleop](./examples/keyboard_teleop.py)

```
uv run examples/keyboard_teleop.py
```

[Gamepad teleop](./examples/gamepad_teleop.py)

Control Stretch in simulation using any xbox type gamepad (uses xinput)

```
uv run examples/gamepad_teleop.py
```

[Robocasa environments](./examples/robocasa_environment.py)

```
# Setup
uv pip install -e .[robocasa]
uv pip install -e "robocasa@third_party/robocasa"
uv pip install -e "robosuite@third_party/robosuite"
uv run third_party/robosuite/robosuite/scripts/setup_macros.py
uv run third_party/robocasa/robocasa/scripts/setup_macros.py
uv run third_party/robocasa/robocasa/scripts/download_kitchen_assets.py

# Run sim
uv run examples/robocasa_environment.py
```

Ignore any warnings.

<img src="./docs/robocasa_scene_1.png" title="Camera Streams" width="300px">
<img src="./docs/robocasa_scene_camera_data.png" title="Camera Streams" width="300px">

## Try Writing Code

Use the [StretchMujocoSimulator](./stretch_mujoco/stretch_mujoco.py) class to:

 * start the simulation
 * position control the robot's ranged joints
 * velocity control the robot's mobile base
 * read joint states
 * read camera imagery

Try the code below using `uv run ipython`. For advanced Mujoco users, the class also exposes the `mjModel` and `mjData`. See the [official Mujoco documentation](https://mujoco.readthedocs.io/en/stable/python.html).

```python
from stretch_mujoco import StretchMujocoSimulator

sim = StretchMujocoSimulator()
sim.start(headless=False) # This will open a Mujoco-Viewer window

# Poses
sim.stow()
sim.home()

# Position Control 
sim.move_to('lift', 1.0)
sim.move_by('head_pan', -1.1)
sim.move_by('base_translate', 0.1)

# Base Velocity control
sim.set_base_velocity(0.3, -0.1)

# Get Joint Status
from pprint import pprint
pprint(sim.pull_status())
"""
Output:
{'time': 6.421999999999515,
 'base': {'x_vel': -3.293721562016785e-07,'theta_vel': -3.061556698064456e-05},
 'lift': {'pos': 0.5889703729548038, 'vel': 1.3548342274419937e-08},
 'arm': {'pos': 0.09806380391427844, 'vel': -0.0001650879063921366},
 'head_pan': {'pos': -4.968686850480367e-06, 'vel': 3.987855066304579e-08},
 'head_tilt': {'pos': -0.00451929555883404, 'vel': -2.2404905787897265e-09},
 'wrist_yaw': {'pos': 0.004738908190630005, 'vel': -5.8446467640096307e-05},
 'wrist_pitch': {'pos': -0.0033446975569971366,'vel': -4.3182498418896415e-06},
 'wrist_roll': {'pos': 0.0049449466225058416, 'vel': 1.27366845279872e-08},
 'gripper': {'pos': -0.00044654737698173895, 'vel': -8.808287459130369e-07}}
"""

# Get Camera Frames
camera_data = sim.pull_camera_data()
pprint(camera_data)
"""
Output:
{'time': 80.89999999999286,
 'cam_d405_rgb': array([[...]]),
 'cam_d405_depth': array([[...]]),
 'cam_d435i_rgb': array([[...]]),
 'cam_d435i_depth': array([[...]]),
 'cam_nav_rgb': array([[...]]),
 'cam_d405_K': array([[...]]),
 'cam_d435i_K': array([[...]])}
"""

# Kills simulation process
sim.stop()
```

### Loading Robocasa Kitchen Scenes

The `stretch_mujoco.robocasa_gen.model_generation_wizard()` method gives you:

- Wizard/API to generate a kitchen model for a given task, layout, and style.
- If layout and style are not provided, it will take you through a wizard to choose them in the terminal.
- If robot_spawn_pose is not provided, it will spawn the robot to the default pose from robocasa fixtures.
- You can also write the generated xml model with absolutepaths to a file.

```python
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.robocasa_gen import model_generation_wizard

# Use the wizard:
model, xml, objects_info = model_generation_wizard()

# Or, launch a specific task/layout/style
model, xml = model_generation_wizard(
    task=<task_name>,
    layout=<layout_id>,
    style=<style_id>,
    wrtie_to_file=<filename>,
)

sim = StretchMujocoSimulator(model=model)
sim.start()
```

### Docs

Check out the following documentation resources:

- [Using the Mujoco Simulator with Stretch](./docs/using_mujoco_simulator_with_stretch.md)

### Feature Requests and Bug reporting

All the enhancements/missing features/Bugfixes are tracked by [Issues](https://github.com/hello-robot/stretch_mujoco/issues) filed. Please feel free to file an issue if you would like to report bugs or request a feature addition.

## Acknowledgment

The assets in this repository contain significant contributions and efforts from [Kevin Zakka](https://github.com/kevinzakka) and [Google Deepmind](https://github.com/google-deepmind), along with others in Hello Robot Inc. who helped us in modeling Stretch in Mujoco. Thank you for your contributions.
