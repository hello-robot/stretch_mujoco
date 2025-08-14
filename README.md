# Stretch Mujoco

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/stretch_mujoco.png" title="Stretch In Kitchen" width="100%">

This library provides a simulation stack for Stretch, built on [MuJoCo](https://github.com/google-deepmind/mujoco). There is position control for the arm, head, and gripper joints, velocity control for mobile base, calibrated camera RGB + depth imagery, 2D spinning lidar scans, and more. There is a visualizer that supports [user interaction](https://youtu.be/2P-Dt-Jfd6U), or a more efficient headless mode. There is a [ROS2 package](https://github.com/hello-robot/stretch_ros2/tree/humble/stretch_simulation), built on this library, that works with Nav2, Web Teleop, and more. There is 100s of permutations of Robocasa-provided kitchen environments that Stretch can spawn into. The MuJoCo API can be used for features like deformables, procedural model generation, SDF collisions, cloth simulation, and more.

Check out the [highlight reel](https://www.youtube.com/watch?v=SWPJt67IB0Q) for features that have been recently added.


## Getting Started
Start with Google Colab:

 - Getting Started Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hello-robot/stretch_mujoco/blob/main/docs/getting_started.ipynb)

**or** follow these instructions on your computer:

First, install [`uv`](https://docs.astral.sh/uv/#getting-started). Uv is a package manager that we'll use to run this project.

Then, clone this repo:

```
git clone https://github.com/hello-robot/stretch_mujoco --recurse-submodules
cd stretch_mujoco
```

> If you've already cloned the repo without `--recurse-submodules`, run `git submodule update --init` to pull the submodule.

Lastly, run the simulation:

```
uv run launch_sim
```

To exit, press `Ctrl+C` in the terminal.

<p>
    <img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/camera_streams.png" title="Camera Streams" height="250px">
    <img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/stretch3_in_mujoco.png" title="Camera Streams" height="250px">
</p>

> On MacOS, if `mjpython` fails to locate `libpython3.10.dylib` and `libz.1.dylib`, run these commands:
```shell
# Before proceeding, please reload your terminal and/or IDE window, to make sure the correct UV environment variables are loaded.

source .venv/bin/activate

# When `libpython3.10.dylib` is missing, run:
PYTHON_LIB_DIR=$(python3 -c 'from distutils.sysconfig import get_config_var; print(get_config_var("LIBDIR"))')
ln -s "$PYTHON_LIB_DIR/libpython3.10.dylib" ./.venv/lib/libpython3.10.dylib

# When `libz.1.dylib` is missing, run:
export DYLD_LIBRARY_PATH=/usr/lib:$DYLD_LIBRARY_PATH
```

## Example Scripts

[Keyboard teleop](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/keyboard_teleop.py)

```
uv run examples/keyboard_teleop.py
```

[Gamepad teleop](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/gamepad_teleop.py)

Control Stretch in simulation using any xbox type gamepad (uses xinput)

```
uv run examples/gamepad_teleop.py
```

[Robocasa environments](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/robocasa_environment.py)

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

<img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/robocasa_scene_1.png" title="Camera Streams" width="300px">
<img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/robocasa_scene_camera_data.png" title="Camera Streams" width="300px">

## Writing Code

Use the [StretchMujocoSimulator](https://github.com/hello-robot/stretch_mujoco/tree/main/stretch_mujoco/stretch_mujoco.py) class to:

 * start the simulation
 * position control the robot's ranged joints
 * velocity control the robot's mobile base
 * read joint states
 * read camera imagery

Try the code below using `uv run ipython`. For advanced Mujoco users, the class also exposes the `mjModel` and `mjData`. See the [official Mujoco documentation](https://mujoco.readthedocs.io/en/stable/python.html).

```python
from stretch_mujoco import StretchMujocoSimulator

if __name__ == "__main__":
    sim = StretchMujocoSimulator()
    sim.start(headless=False) # This will open a Mujoco-Viewer window
    
    # Poses
    sim.stow()
    sim.home()
    
    # Position Control 
    sim.move_to('lift', 1.0)
    sim.move_by('head_pan', -1.1)
    sim.move_by('base_translate', 0.1)

    sim.wait_until_at_setpoint('lift')
    sim.wait_while_is_moving('base_translate')
    
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

Note that the `if __name__ == "__main__":` guard is necessary, as explained in the [Python Docs](https://docs.python.org/3/library/multiprocessing.html#:~:text=For%20an%20explanation%20of%20why%20the%20if%20__name__%20%3D%3D%20%27__main__%27%20part%20is%20necessary%2C%20see%20Programming%20guidelines.).

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

### ROS2

You can use this simulation in ROS2 using the [`stretch_simulation` package](https://github.com/hello-robot/stretch_ros2/tree/humble/stretch_simulation) in `stretch_ros2`.

### Docs

Check out the following documentation resources:

- [Using the Mujoco Simulator with Stretch](./docs/using_mujoco_simulator_with_stretch.md)
- [Getting Started jupyter notebook](./docs/getting_started.ipynb)
- [Releasing to PyPi](./docs/releasing_to_pypi.md)
- [Contributing to this project](./docs/contributing.md)
- [Changelog](./CHANGELOG.md)

### Feature Requests and Bug reporting

All enhancements/missing features/bugfixes are tracked by [Issues](https://github.com/hello-robot/stretch_mujoco/issues) filed. Please feel free to file an issue if you would like to report bugs or request a feature addition. Pull requests are welcome! Please see the [contributing guide](./docs/contributing.md).

## Acknowledgment

The assets in this repository contain significant contributions and efforts from [Kevin Zakka](https://github.com/kevinzakka) and [Google Deepmind](https://github.com/google-deepmind), along with others in Hello Robot Inc. who helped us in modeling Stretch in Mujoco. Thank you for your contributions.
