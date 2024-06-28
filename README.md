# Stretch Mujoco
This repo provides assets and sample scripts for simulating Stretch with [Mujoco](https://github.com/google-deepmind/mujoco).
Currently only Stretch 3 version is fully supported with position control interface for robot joints and velocity control for base.

## Getting Started
1. Install Mujoco (v3.0>), older versions might work but not tested.
```
pip3 install mujoco
```

2. Run Stretch in simulation via Mujoco Viewer and see the camera frames
```
git clone https://github.com/hello-robot/stretch_mujoco
cd stretch_mujoco
python3 stretch_mujoco.py
```

3. Control Stretch in simulation using any xbox type gamepad (needs Stretch Body installed)
```
python3 stretch_mujoco_gamepad.py
```
<img src="./docs/stretch3_in_mujoco.png" title="Camera Streams" width="400px"> <img src="./docs/camera_streams.png" title="Camera Streams" width="500px">

## Try Writing Code
You can refer to the [StretchMujocoSimulator](https://github.com/hello-robot/stretch_mujoco/blob/main/stretch_mujoco.py#L11) class implementation which provides a basic interface for starting the Simulation, position control the robot, read joint status and read all the camera streams. You can try the below lines from Ipython terminal. Advanced users are recommended to reffer to [official Mujoco documentation](https://mujoco.readthedocs.io/en/stable/python.html).
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
 'cam_nav_rgb': array([[...]]),}
"""

```

### Feature Requests and Bug reporting
All the enhacements/missing features/Bugfixes are tracked by [Issues](https://github.com/hello-robot/stretch_mujoco/issues) filed. Please free to file an issue if you would like to report bugs or request a feature addition.

## Acknowledgment
The assets in this repository contains significant contributions and efforts from [Kevin Zakka](https://github.com/kevinzakka) and [Google Deepmind](https://github.com/google-deepmind) along with others in Hello Robot Inc. who helped us in modelling Stretch in Mujoco. Thankyou for your contributions.

