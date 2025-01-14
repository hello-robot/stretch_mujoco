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
git clone https://github.com/hello-robot/stretch_mujoco
cd stretch_mujoco
```

Lastly, run the simulation. It'll spawn Stretch in default scene and pop up 5 additional windows that shows what the 2 depth cameras and 1 wide-angle camera sees.

```
uv run launch_sim.py
```

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
# Download assets first TODO
uv run ./.venv/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
uv run https://raw.githubusercontent.com/robocasa/robocasa/refs/heads/main/robocasa/scripts/download_kitchen_assets.py

# Run sim
uv run examples/robocasa_environment.py
```

<img src="./docs/robocasa_scene_1.png" title="Camera Streams" width="300px">
<img src="./docs/robocasa_scene_camera_data.png" title="Camera Streams" width="300px">

## Try Writing Code

Use the [StretchMujocoSimulator](https://github.com/hello-robot/stretch_mujoco/blob/main/stretch_mujoco.py) class implementation which provides the control interface for starting the Simulation, position control the robot, read joint status and read all the camera streams. You can try the below lines simply from Ipython terminal. The class also has `mjModel` and `mjData` elements which Advanced users take advantage with [official Mujoco documentation](https://mujoco.readthedocs.io/en/stable/python.html).

```python
from stretch_mujoco import StretchMujocoSimulator

robot_sim = StretchMujocoSimulator('./scene.xml')
robot_sim.start() # This will start the simulation and open Mujoco-Viewer window

# Poses
robot_sim.home()
robot_sim.stow()

# Position Control 
robot_sim.move_to('lift',0.6)
robot_sim.move_by('head_pan',0.1)

# Base Velocity control
robot_sim.set_base_velocity(0.3,-0.1)

# Get Joint Status (updated continuously in simulation callback mjcb_control)
print(robot_sim.status)
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
camera_data = robot_sim.pull_camera_data()
print(camera_data)
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
robot_sim.stop()
```

### Loading Robocasa Kitchen Scenes

`robocasa_gen.model_generation_wizard()` API

- Wizard/API to generate a kitchen model for a given task, layout, and style.
- If layout and style are not provided, it will take you through a wizard to choose them in the terminal.
- If robot_spawn_pose is not provided, it will spawn the robot to the default pose from robocasa fixtures.
- You can also write the generated xml model with absolutepaths to a file.

```python
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.robocasa_gen import model_generation_wizard
model, xml = model_generation_wizard(
    task=task_name,
    layout=layout_id,
    style=style_id,
    wrtie_to_file=filename,
)
robot_sim = StretchMujocoSimulator(model=model)
robot_sim.start()
```

### Feature Requests and Bug reporting

All the enhancements/missing features/Bugfixes are tracked by [Issues](https://github.com/hello-robot/stretch_mujoco/issues) filed. Please feel free to file an issue if you would like to report bugs or request a feature addition.

## Acknowledgment

The assets in this repository contain significant contributions and efforts from [Kevin Zakka](https://github.com/kevinzakka) and [Google Deepmind](https://github.com/google-deepmind), along with others in Hello Robot Inc. who helped us in modeling Stretch in Mujoco. Thank you for your contributions.
