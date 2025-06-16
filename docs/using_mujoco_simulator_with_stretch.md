# Using the Mujoco Simulator with Stretch

When using Mujoco to simulate Stretch, you can command [joints](https://github.com/hello-robot/stretch_mujoco/tree/main/stretch_mujoco/enums/actuators.py) and access joint poses and [camera](https://github.com/hello-robot/stretch_mujoco/tree/main/stretch_mujoco/enums/stretch_cameras.py) data.

## Getting Started

1. Read the [README](https://github.com/hello-robot/stretch_mujoco/tree/main/README.md) to install the required dependencies.
2. Check out the controller examples, such as:
- [keyboard_teleop.py](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/keyboard_teleop.py)
- [gamepad_teleopy.py](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/gamepad_teleop.py)
3. Check out the headless examples, such as:
- [draw_circles.py](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/draw_circles.py)
- [camera_feeds.py](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/camera_feeds.py)
4. Check out the sensor example: [laser_sca.py](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/laser_scan.py)

### Terminology

The following words apply to this document only, to make it easier to read:

- [Mujoco](https://mujoco.readthedocs.io/en/stable/overview.html): An open-source physics engine.
- [Mujoco Viewer](https://mujoco.readthedocs.io/en/stable/programming/samples.html#sasimulate): An interactive Mujoco GUI that ships with Mujoco. This is spawned when you don't use `headless` mode.
- [Headless Mode](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#main-simulation): Using the Mujoco simulation without the Mujoco Viewer. This calls `mj_step` directly to step the simulation. For the purposes of Stretch Mujoco Simulations, this is a performant mode to run simulations in.
- [Stretch Mujoco Simulator](https://github.com/hello-robot/stretch_mujoco/tree/main/stretch_mujoco/stretch_mujoco_simulator.py): A scaffolding that enables you to send commands and receive sensor data to and from Stretch in a Mujoco environment.

## Control Flow

All simulations using [Stretch Mujoco Simulator](https://github.com/hello-robot/stretch_mujoco/tree/main/stretch_mujoco/stretch_mujoco_simulator.py) should have an entry point that looks similar to this:

```python
if __name__ == "__main__":

    # You can use all the camera's, but it takes longer to render, and may affect the overall simulation FPS.
    # cameras_to_use = StretchCameras.all()
    cameras_to_use = [StretchCameras.cam_d405_rgb]

    sim = StretchMujocoSimulator(cameras_to_use=cameras_to_use)

    sim.start(headless=True)
```

> You can use the `start()` method to specify headless or UI view modes. The Passive Viewer is the default and recommended non-headless mode.

> If you are using the simulation for machine-learning applications, it is recommended to use the headless mode for better performance.

> If you are displaying camera data or doing heavy computations on your Control Loop, it is recommended to move your control commands to a thread, and display the camera data on the main thread. See [Displaying camera data using OpenCV](#displaying-camera-data-using-opencv) below and the [camera_feeds.py](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/camera_feeds.py) example for more information.

### Commanding Stretch

When the simulator is running, you can use your script to send commands to Stretch, or read data from the simulation.

Use the following command to move the lift to `0.5m`: `sim.move_to(Actuators.lift, 0.5)`

### Reading data from Stretch

There are two methods for pulling data from the simulation: `sim.pull_status()` and `sim.pull_camera_data()`.

#### Stretch Status

Use `sim.pull_status()` to fetch the joint states of the robot.

This method returns a `StatusStretchJoints` [dataclass](https://github.com/hello-robot/stretch_mujoco/tree/main/stretch_mujoco/datamodels/status_stretch_joints.py) with the names of all the joints populated.

The statuses of all the joints are fetched at the same time.

#### Stretch Sensors

Use `sim.pull_sensor_data()` to fetch data from sensors on Stretch.

This methods returns a `StatusStretchSensors` [dataclass](https://github.com/hello-robot/stretch_mujoco/tree/main/stretch_mujoco/datamodels/status_stretch_sensors.py)

The statuses of all the sensors are fetched at the same time.

All the sensors defined in [`stretch.xml`](https://github.com/hello-robot/stretch_mujoco/tree/main/stretch_mujoco/models/scene.xml) are fetched:

```
  <sensor>
    <gyro name="base_gyro" site="base_imu"/>
    <accelerometer name="base_accel" site="base_imu"/>
    <rangefinder name="base_lidar" site="lidar" cutoff="10.0"/>
  </sensor>
```

> Note: The Lidar sensor (implemented as `<rangefinder/>`) is compute intensive. Comment it out in the XML, if you are not using it. 


#### Stretch Cameras

Use `sim.pull_camera_data()` to fetch the pixel values from Stretch's cameras.

This method returns a `StatusStretchCameras` [dataclass](https://github.com/hello-robot/stretch_mujoco/tree/main/stretch_mujoco/datamodels/status_stretch_camera.py) with the names of all the cameras populated.

The pixel values of all the renderings of the cameras are fetched at the same time.

Note: this operation is computationally heavy. The more cameras that are requested, the slower the simulation may run.

##### Displaying camera data using OpenCV

The [camera_feeds.py](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/camera_feeds.py) example shows a sample to display camera data using `cv2.imshow()`.

```python

camera_data = sim.pull_camera_data()

for camera in cameras_to_use:
    cv2.imshow(camera.name, camera_data.get_camera_data(camera))

cv2.waitKey(1)
```

> Important Note: you should call `cv2.imshow()` on the MAIN THREAD to avoid getting graphics library (GL) related errors from your OS.

### Misc Stretch Mujoco Simulator API calls

#### World Coordinate Frame Arrows

You can use `sim.add_world_frame((0.1, 0.0, 0.0))` to add arrows dynamically to the Mujoco viewer:

<img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/coordinate_frame_arrows.png" width=400/>

This also supports rotating the frame:

```
sim.add_world_frame((0.1,0,0), (0,0,0))
sim.add_world_frame((0.2,0,0), (1.57,0,0)) # (x, y, z), (r, p, y)
```

<img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/rotate_coordinate_frames.png" width=400/>

## More to know

### Behind the scenes

When you call `sim.start()`, the following process diagram explains how Mujoco is launched and your Control Loop are managed.

> tl;dr The Mujoco simulator is started on a spawned process, and data is communicated between your main process and the Mujoco process using a [Multiprocesing Manager](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Manager).

<img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/mujoco_server_process_diagram.jpg" width=600>


### Mujoco rendering locks

Mujoco's [passive](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer) and headless modes need some access governance over the `mjdata` and `mjmodel` objects to behave correctly.

When using either of these modes, we are responsible for calling `mj_step` to step the simulation. This also means we're responsible for managing when Mujoco should render scenes. Calling `mj_step` is not a thread-safe operation, and if it's done while a render is rendering, bad things happen. So we use some mutexes to ensure these steps are in sync.

> Note: ignoring the use of mutexes to lock `mjdata` and `mjmodel` before rendering could cause Mujoco to crash, or instability errors such as "WARNING: Inertia matrix is too close to singular at DOF 11. Check model. WARNING: Nan, Inf or huge value in QPOS at DOF 0. The simulation is unstable."
