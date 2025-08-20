# Contributing Guide

Thank you for considering contributing to this repository. Stretch Mujoco is a high-fidelity simulation of Stretch built on the MuJoCo simulation engine. This repo contains the assets that describe Stretch, and code that mimics the real robot's API to motion commands and reading sensors. This guide explains the layout of the repo and how to make changes to the code or models.

## Repo Layout

 - `README.md` - quickstart for new users / overview of the software's capabilities
 - `pyproject.toml` - defines the project's dependencies
 - `mkdocs.yml` - defines the tutorials that should appear on the [main docs page](https://docs.hello-robot.com/0.3/#simulating-stretch)
 - `docs/` - tutorials and documentation
 - `examples/` - small scripts that show some aspect of the sim
 - `stretch_mujoco/` - source code for the sim
     - `mujoco_server_<>.py` - server side of the simulation; `<>` can be `passive`, `managed` for simulation with visualization (passive vs. managed viz is [described here](https://mujoco.readthedocs.io/en/stable/python.html#interactive-viewer)), or empty for headless simulation.
     - `stretch_mujoco_simulation.py` - client side of the simulation; the API is defined here. Communication between client and server is managed by shared memory.
     - `robocasa_gen.py` - generates the xml for the kitchen-style environment
     - `datamodels/` - defines custom [dataclasses](https://www.dataquest.io/blog/how-to-use-python-data-classes/), which is returned to the user on the client side. E.g. `StatusStretchCameras` holds image data from the simulated cameras.
     - `enums/` - enumerates the actuators, cameras, sensors, etc. E.g. instead of using strings like 'joint_arm' in the API, `move_by('joint_arm', 0.1)`, you can use the enum value, `move_by(Actuators.arm, 0.1)`. This reduces the probability of string related errors.
     - `models/` - contains all the XMLs, mesh files, image textures, material files, etc., that describe the robot, environments, and other assets
 - `third_party/` - robosuite and robocasa repos are cloned within here. This folder is a few gigabytes in size.

## Client Server

Let's say you wanted to add a new feature that allows users to visualize coordinate frames in the visualizer. Ideally, the API is something like `sim.add_world_frame((x, y, z), (r, p, y))`. To implement this, you'll need to modify both the **client** and the **server**.

`StretchMujocoSimulator` is the client. When you instance it and call `sim.start()`, it spawns an instance of `MujocoServer` in a separate process. No other clients can connect to this server. All physics, rendering, and visualizing happens in this server process. The two processes communicate over shared memory, so when you call `add_world_frame((x, y, z), (r, p, y))` in the client, it'll add a command called `CommandCoordinateFrameArrowsViz` to a data structure that is shared with the server.

```python
class StretchMujocoSimulator:
    ...

    @require_connection
    def add_world_frame(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Add a world frame to the simulator for visualization.
        Args:
            position: tuple of (x, y, z) coordinates in the world frame
            rotation: tuple of (x, y, z) angle in radians for the rotation around each axis
        """
        with self._command_lock:
            command = self.data_proxies.get_command()
            command.coordinate_frame_arrows_viz.append(
                CommandCoordinateFrameArrowsViz(position=position, rotation=rotation, trigger=True)
            )
            self.data_proxies.set_command(command)
```

The shared data structure, `data_proxies`, is a dataclass composed of Python dictionaries managed by `multiprocessing.SyncManager`. On the server side, it uses its own data proxy to read the command and process it.

```python
class MujocoServerPassive(MujocoServer):
    ...

    def push_command(self, command_status: StatusCommand):

        command_arrows = command_status.coordinate_frame_arrows_viz.copy()

        for arrows in command_arrows:
            if arrows.trigger:
                self._add_axes_to_user_scn(self.viewer.user_scn, np.array(arrows.position) , arrows.rotation)

                command_status.coordinate_frame_arrows_viz.remove(arrows)

        super().push_command(command_status)
```

Other types of commands processed in `push_command()` include move_bys, move_tos, set_base_velocitys, homing & stowing commands. Most of these are processed within `MujocoServer`, but since visualizing coordinate frames requires a visualizer, these commands are processed in `MujocoServerPassive`.


## Creating new assets

All the models used in this simulation live within the `stretch_mujoco/models/` folder. At the time of writing, this includes:
 - `stretch.xml` - describes the robot
 - `docking_station.xml` - describes the docking station
 - `scene.xml` - includes the robot, the dock, a small table, some objects on top of the table, and a floor (so the robot has something to roll around on)
 - `stretch_mj_3.3.0.xml` - an experimental descripion for Stretch which will enable v3.3.0 of MuJoCo to simulate the robot in the future. It currently shows unstable behavior.
 - `assets/` - contains the mesh files, image textures, material files, etc. that are used by the above XML models

The MuJoCo XML format is described here: https://mujoco.readthedocs.io/en/stable/XMLreference.html. Creating a model of something as complex as Stretch 3 takes a lot of effort. A collaboration between Binit from Hello Robot and Kevin from Google Deepmind took on & off effort of roughly a year to finish it. Before we started, we already had a URDF and obj mesh files for Stretch 3, which was generated from Solidworks (the solidworks-urdf-exporter). The URDF was manually corrected to follow ROS guidelines on link orientation and position. MuJoCo can import the URDF and spit out an initial XML description. Then, each of the mesh files were colored and textured in [Blender](https://blender.org/) and exported to .dae files. Important collision meshes were decomposed into convex pieces using Kevin's [`obj2mjcf`](https://github.com/kevinzakka/obj2mjcf) CLI. Mass/inertia of each body and motion characteristics of each joint were estimated here: https://forum.hello-robot.com/t/physical-aspects-of-the-robot-for-simulation/792. These visual meshes, collision meshes, mass/inertia parameters were incorported into the XML file. Then, significant time was spent tuning motion gains (to match the motion characteristics), friction models (e.g. the caster wheel is a frictionless ball), collision geometry (e.g. the drive wheels are cylinders, not meshes), and more. A coupled gripper kinematic model was added to mimic the gripping motion of the compliant gripper that comes standard with Stretch 3. Lastly, we ported autonomy & teleop software written for the real robot to this simulation and verified the two exhibited similar behaviors as a sort of test of the simulation's accuracy.

MuJoCo's XML `<include>` tag makes it easy to put together assets in a modular way. For example, Stretch is designed to support swappable tools through the quick-connect interface, and simulating Stretch with different tools could be done by creating a new XML for each tool and including it in `stretch.xml`. If this of interest, let us know and we can move the Standard Gripper out into its own model.

Creating new assets for the simulation would follow a similar process. If you have a CAD, you can export out to URDF and have MuJoCo generate an initial XML file. If your model has any concave geometry, you can use `obj2mjcf` CLI to decompose it. If visual fidelity is important, you can color the visual meshes in Blender. Also, Blender's UV unwrap feature makes it easy to texture meshes with an image, which can be useful for adding [ArUco markers](https://youtu.be/whbgfYb1x7Y) to your model. Then you can iterate on the model by simulating it and tuning parameters to more closely match behavior from the real object. If you have access to a mocap system, MuJoCo provides functionality for automatically tuning parameters based on mocap data.

It's much easier to use existing model repositories. Here's a few:
 - [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/) - High quality models of other robots
 - [Robocasa](https://robocasa.ai/) - High quality assets of kitchens, interactable appliances (e.g. opening the microwave door), and more.
