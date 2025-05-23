import atexit
from multiprocessing import Lock, Manager, Process

import multiprocessing
import platform
import signal
import sys
import threading
import time

import click
import numpy as np
from mujoco._structs import MjModel

from stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from stretch_mujoco.datamodels.status_stretch_sensors import StatusStretchSensors
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.mujoco_server import MujocoServer, MujocoServerProxies
from stretch_mujoco.mujoco_server_managed import MujocoServerManaged
from stretch_mujoco.mujoco_server_passive import MujocoServerPassive
from stretch_mujoco.datamodels.status_command import (
    CommandBaseVelocity,
    CommandKeyframe,
    CommandMove,
    StatusCommand,
)
import stretch_mujoco.utils as utils
from stretch_mujoco.utils import require_connection, wait_and_check


class StretchMujocoSimulator:
    """
    Stretch Mujoco Simulator class for interfacing with the Mujoco Server.

    Calling `start()` will spawn a new process that runs `MujocoServer` and the simulator.

    You can specify `start(headless=True)` to run the simulation without a GUI.

    Data from the MujocoServer is sent to StretchMujocoSimulator using proxies.

    Use `pull_status()` and `pull_camera_data()` to access simulation data.
    """

    def __init__(
        self,
        scene_xml_path: str | None = None,
        model: MjModel | None = None,
        camera_hz: float = 30,
        cameras_to_use: list[StretchCameras] = [],
    ) -> None:
        self.scene_xml_path = scene_xml_path
        self.model = model
        self.camera_hz = camera_hz
        self.urdf_model = utils.URDFmodel()
        self._server_process = None
        self._cameras_to_use = cameras_to_use

        self.is_stop_called = False

        self._manager = Manager()
        self._stop_mujoco_process_event = self._manager.Event()

        self.data_proxies = MujocoServerProxies.default(self._manager)
        
        self._command_lock = Lock()

    def start(
        self, show_viewer_ui: bool = False, headless: bool = False, use_passive_viewer: bool = True
    ) -> None:
        """
        Start the simulator

        Args:
            show_viewer_ui: bool, whether to show the Mujoco viewer UI
            headless: bool, whether to run the simulation in headless mode
            use_passive_viewer: bool, to use the passive or managed mujoco UI viewer.
        """
        self.is_stop_called = False

        mujoco_server = MujocoServer  # Headless

        if not headless:
            mujoco_server = MujocoServerPassive if use_passive_viewer else MujocoServerManaged

        if platform.system() == "Darwin" and mujoco_server is MujocoServerPassive:
            # On a mac, the process for MujocoServerPassive needs to be started with mjpython
            mjpython_path = sys.executable.replace("bin/python3", "bin/mjpython").replace(
                "bin/python", "bin/mjpython"
            )
            print(f"{mjpython_path=}")
            multiprocessing.set_executable(mjpython_path)

        multiprocessing.set_start_method("spawn", force=True)

        self._server_process = Process(
            target=mujoco_server.launch_server,
            name="MujocoProcess",
            args=(
                self.scene_xml_path,
                self.model,
                self.camera_hz,
                show_viewer_ui,
                self._stop_mujoco_process_event,
                self.data_proxies,
                self._cameras_to_use,
            ),
            daemon=False,  # We're gonna handle terminating this in stop_mujoco_process()
        )
        self._server_process.start()

        # Handle stopping, in all its various ways:
        signal.signal(signal.SIGTERM, lambda num, sig: self.stop())
        signal.signal(signal.SIGINT, lambda num, sig: self.stop())
        atexit.register(self.stop)

        click.secho("Starting Stretch Mujoco Simulator...", fg="green")
        while self.pull_status().time == 0 or self.pull_camera_data().time == 0:
            time.sleep(1)
            click.secho("Still waiting to connect to the Mujoco Simulatior.", fg="yellow")

            if not self.is_running():
                click.secho("The simulator is not running anymore, quitting..", fg="yellow")
                return

        click.secho("The Mujoco Simulatior is connected.", fg="green")

        self.home()

    def stop(self) -> None:
        """
        This is called at exit to gracefully terminate the simulation and the Mujoco Process, and their many threads.

        Fingers-crossed we get a SIGTERM, and not a SIGKILL..
        """
        if self.is_stop_called:
            return

        self.is_stop_called = True

        try:
            simulation_time_message = self.data_proxies.get_status().time
            simulation_time_message = f" simulated runtime= {simulation_time_message:.1f}s"
        except:
            simulation_time_message = ""

        click.secho(
            f"Stopping Stretch Mujoco Simulator...{simulation_time_message}",
            fg="red",
        )

        self.stop_mujoco_process()

        # We're going to try to wait for threads to end. They might not gracefully stop before hitting an exception. Race conditions are rampant.
        # For example, the main thread or a thread may not be checking `sim.is_running()` and is oblivious that it should stop. Nothing we can do to stop it except sigkill.
        active_threads = threading.enumerate()
        for index, thread in enumerate(active_threads):
            if (
                thread != threading.current_thread()
                and thread != threading.main_thread()
                and not isinstance(thread, threading._DummyThread)
            ):
                click.secho(
                    f"Stopping thread {index}/{len(active_threads)-1}.",
                    fg="yellow",
                )
                thread.join(timeout=10.0)
                if thread.is_alive():
                    click.secho(
                        f"{thread.name} is not terminating. Make sure to check 'sim.is_running()' in threading loops.",
                        fg="red",
                    )

        click.secho(
            f"The Stretch Mujoco Simulator has ended. Good-bye!",
            fg="red",
        )

    def stop_mujoco_process(self):

        if self._server_process and not self._server_process.is_alive():
            click.secho(
                f"The Mujoco process has already terminated.",
                fg="red",
            )
            return

        click.secho(
            f"Sending signal to stop the Mujoco process...",
            fg="red",
        )

        # Wait until the main control loop ends before sending this stop event.
        self._stop_mujoco_process_event.set()
        if self._server_process:
            # self._server_process.terminate() # ask it nicely.
            self._server_process.join()

        click.secho(
            f"The Mujoco process has ended.",
            fg="red",
        )

    @require_connection
    def home(self) -> None:
        """
        Move the robot to home position
        """
        with self._command_lock:
            self.data_proxies.set_command(
                StatusCommand(keyframe=CommandKeyframe(name="home", trigger=True))
            )
        time.sleep(2)

    @require_connection
    def stow(self) -> None:
        """
        Move the robot to stow position
        """
        with self._command_lock:
            self.data_proxies.set_command(
                StatusCommand(keyframe=CommandKeyframe(name="stow", trigger=True))
            )
        time.sleep(2)

    @require_connection
    def move_to(self, actuator: str | Actuators, pos: float, timeout: float | None = 15.0) -> bool:
        """
        Move the actuator to an absolute position.
        Args:
            actuator: string name of the actuator or Actuator enum instance
            pos: float, absolute position goal
            timeout: if not None, then it will wait for the joint to reach that position, or return False
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
            Actuators.base_rotate,
            Actuators.base_translate,
        ]:
            click.secho(
                f"Cannot set an absolute position for a continuous joint {actuator.name}",
                fg="red",
            )
            return False

        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_move_to(CommandMove(actuator_name=actuator.name, pos=pos, trigger=True))

            self.data_proxies.set_command(command)

        if timeout:
            if not wait_and_check(
                timeout,
                lambda: np.isclose(actuator.get_position(self.pull_status()), pos, atol=0.05)
                == True,
                self.is_running,
            ):
                actual = actuator.get_position(self.pull_status())
                error = pos - actual
                click.secho(
                    f"Joint {actuator.name} did not reach {pos}. Actual: {actual:.4f} Diff: {error*100:.4f}cm",
                    fg="red",
                )
                return False
        return True

    @require_connection
    def move_by(self, actuator: str | Actuators, pos: float, timeout: float | None = None) -> bool:
        """
        Move the actuator by a relative amount.
        Args:
            actuator: string name of the actuator or Actuator enum instance
            pos: float, position to increment by
            timeout: if not None, then it will wait for the joint to reach that position, or return False.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [Actuators.left_wheel_vel, Actuators.right_wheel_vel]:
            click.secho(
                f"Cannot set a position for a velocity joint {actuator.name}",
                fg="red",
            )
            return False

        with self._command_lock:
            command = self.data_proxies.get_command()
            
            command.set_move_by(
                # We set the pos here, and not new_position, because this relative motion math is handled by mujoco_server:
                CommandMove(actuator_name=actuator.name, pos=pos, trigger=True)
            )

            self.data_proxies.set_command(command)

        if timeout is not None:

            if actuator in [Actuators.base_rotate, Actuators.base_translate]:
                raise NotImplementedError(f"move_by Timeout is not supported for {actuator}.")
            
            initial_position = actuator.get_position(self.pull_status())

            new_position = pos + initial_position

            check = lambda: np.isclose(actuator.get_position(self.pull_status()), new_position, atol=0.01) == True

            if not wait_and_check(
                wait_timeout=timeout,
                check=check,
                is_alive=self.is_running,
            ):
                actual = actuator.get_position(self.pull_status())
                error = new_position - actual
                click.secho(
                    f"Joint {actuator.name} did not move by {pos}. Expected: {new_position:.4f} Actual: {actual:.4f} Diff: {error*100:.4f}cm",
                    fg="red",
                )
                return False
        return True

    @require_connection
    def set_base_velocity(self, v_linear: float, omega: float) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """

        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_base_velocity(CommandBaseVelocity(v_linear=v_linear, omega=omega, trigger=True))

            self.data_proxies.set_command(command)

    @require_connection
    def get_base_pose(self):
        """Get the se(2) base pose: x, y, and theta"""
        status = self.pull_status()
        return (status.base.x, status.base.y, status.base.theta)

    @require_connection
    def get_ee_pose(self) -> np.ndarray:
        return self.get_link_pose("link_grasp_center")

    @require_connection
    def get_link_pose(self, link_name: str) -> np.ndarray:
        """Pose of link in world frame"""
        status = self.pull_status()
        cfg = {
            "wrist_yaw": status.wrist_yaw.pos,
            "wrist_pitch": status.wrist_pitch.pos,
            "wrist_roll": status.wrist_roll.pos,
            "lift": status.lift.pos,
            "arm": status.arm.pos,
            "head_pan": status.head_pan.pos,
            "head_tilt": status.head_tilt.pos,
        }
        transform = self.urdf_model.get_transform(cfg, link_name)
        base_xyt = self.get_base_pose()
        base_4x4 = np.eye(4)
        base_4x4[:3, :3] = utils.Rz(base_xyt[2])
        base_4x4[:2, 3] = base_xyt[:2]
        world_coord = np.matmul(base_4x4, transform)
        return world_coord

    @require_connection
    def pull_camera_data(self) -> StatusStretchCameras:
        """
        Pull camera data from the simulator and return as a StatusStretchCameras
        """
        return self.data_proxies.get_cameras()

    @require_connection
    def pull_sensor_data(self) -> StatusStretchSensors:
        """
        Pull sensor data from the simulator and return as a StatusStretchSensors
        """
        return self.data_proxies.get_sensors()

    @require_connection
    def pull_status(self) -> StatusStretchJoints:
        """
        Pull robot joint states from the simulator and return as a StatusStretchJoints
        """
        return self.data_proxies.get_status()
    
    @require_connection
    def pull_joint_limits(self) -> dict[Actuators, tuple[float, float]]:
        """
        Pull robot joint limuts from the simulator and return as a dict
        """
        return self.data_proxies.get_joint_limits()

    def is_mujoco_process_dead_or_stopevent_triggered(self):
        return (
            self._server_process is None
            or not self._server_process.is_alive()
            or self._stop_mujoco_process_event.is_set()
        )

    def is_running(self) -> bool:
        """
        Check if the simulator and mujoco are running, or if the stopevent signal has been triggered.

        Side-effect here is that if the mujoco process is terminated or the stopevent is triggered, `self.stop()` is called.
        """
        if self.is_mujoco_process_dead_or_stopevent_triggered():
            # Send the signal to stop the program:
            self.stop()
            return False

        return not self.is_stop_called
