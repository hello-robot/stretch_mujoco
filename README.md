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

<img src="./docs/stretch3_in_mujoco.png" title="Mujoco Viewer" height="250px">

```
uv run launch_sim.py
```

To exit, press `Ctrl+C` in the terminal.

> [!NOTE]
> On MacOS, if `mjpython` fails to locate `libpython3.10.dylib` or `libz.1.dylib`, run these commands:
> ```shell
> # First, reload your terminal and/or IDE window. Then, run:
> source .venv/bin/activate
> 
> # When `libpython3.10.dylib` is missing, run:
> PYTHON_LIB_DIR=$(python3 -c 'from distutils.sysconfig import get_config_var; print(get_config_var("LIBDIR"))')
> ln -s "$PYTHON_LIB_DIR/libpython3.10.dylib" ./.venv/lib/libpython3.10.dylib
> 
> # When `libz.1.dylib` is missing, run:
> export DYLD_LIBRARY_PATH=/usr/lib:$DYLD_LIBRARY_PATH
> ```

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
uv pip install -e ".[robocasa]"
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
StatusStretchJoints(time=7.879999999999354,
                    fps=341.805136891146,
                    base=BaseStatus(x=-0.011022089958242815,
                                    y=0.005272088098178076,
                                    theta=-0.02275072930118742,
                                    x_vel=-1.5179662584149273e-08,
                                    theta_vel=-9.925945325105483e-08),
                    lift=PositionVelocity(pos=0.5904024480107699,
                                          vel=0.00023585559532521424),
                    arm=PositionVelocity(pos=0.09999622630047073,
                                         vel=1.510665637401413e-10),
                    head_pan=PositionVelocity(pos=-5.005046267753644e-06,
                                              vel=-2.683534987547478e-13),
                    head_tilt=PositionVelocity(pos=-0.004519272499151502,
                                               vel=-5.433457555935975e-13),
                    wrist_yaw=PositionVelocity(pos=0.00010899348366191074,
                                               vel=-4.516959470390041e-05),
                    wrist_pitch=PositionVelocity(pos=-0.005324520063887358,
                                                 vel=-6.641705404829458e-09),
                    wrist_roll=PositionVelocity(pos=-9.58899676474824e-05,
                                                vel=3.4738029638534585e-08),
                    gripper=PositionVelocity(pos=-0.06399749253897319,
                                             vel=2.7702326383783765e-09))
"""

# Get Camera Frames
camera_data = sim.pull_camera_data()
pprint(camera_data)
"""
Output:
StatusStretchCameras(time=77.77400000000014,
                     fps=29.46741244088998,
                     cam_d405_rgb=array([[...]]),
                     cam_d405_depth=array([[...]]),
                     cam_d405_K=array([[514.68166092,   0.        , 320.        ],
                                       [  0.        , 514.68166092, 240.        ],
                                       [  0.        ,   0.        ,   1.        ]]),
                     cam_d435i_rgb=array([[...]]),
                     cam_d435i_depth=array([[...]]),
                     cam_d435i_K=array([[399.42707576,   0.        , 320.        ],
                                        [  0.        , 399.42707576, 240.        ],
                                        [  0.        ,   0.        ,   1.        ]]),
                     cam_nav_rgb=array([[...]]))
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
