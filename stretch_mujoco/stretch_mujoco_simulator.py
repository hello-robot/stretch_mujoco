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
    CommandCoordinateFrameArrowsViz,
    CommandKeyframe,
    CommandMove,
    StatusCommand,
)
import stretch_mujoco.utils as utils
from stretch_mujoco.utils import require_connection, block_until_check_succeeds


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
        self.wait_while_is_moving(Actuators.lift)

    @require_connection
    def stow(self) -> None:
        """
        Move the robot to stow position
        """
        with self._command_lock:
            self.data_proxies.set_command(
                StatusCommand(keyframe=CommandKeyframe(name="stow", trigger=True))
            )

        self.wait_while_is_moving(Actuators.wrist_pitch)

    def is_reached_set_position(self, actuator: str | Actuators, position_tolerance: float = 0.05):
        """
        Checks if the joint has reached a previously commanded location.

        Only listens to the `move_to` command.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [
            Actuators.base_rotate,
            Actuators.base_translate,
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
        ]:
            raise NotImplementedError(f"Check joint reached is not supported for {actuator}.")

        move_command = self.data_proxies.get_command().move_to.get(actuator.name)

        if not move_command:
            click.secho(
                "Warning: Position check requested, but the joint was not commanded to move.",
                fg="yellow",
            )
            return True

        set_position = move_command.pos

        current_position = actuator.get_position(self.pull_status())

        return bool(np.isclose(current_position, set_position, atol=position_tolerance))

    def wait_until_at_setpoint(
        self, actuator: str | Actuators, timeout: float = 5.0, position_tolerance: float = 0.05
    ):
        """Blocks until the actuator reaches its previously set point."""
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        move_command = self.data_proxies.get_command().move_to.get(actuator.name)

        if not move_command:
            return True

        if not block_until_check_succeeds(
            wait_timeout=timeout,
            check=lambda: self.is_reached_set_position(
                actuator=actuator, position_tolerance=position_tolerance
            )
            == True,
            is_alive=self.is_running,
        ):
            pos = move_command.pos
            actual = actuator.get_position(self.pull_status())
            error = pos - actual
            click.secho(
                f"Timeout: Joint {actuator.name} did not reach {pos}. Actual: {actual:.4f} Diff: {error*100:.4f}cm",
                fg="red",
            )
            return False
        return True

    _last_movement_positions: dict[Actuators, float | tuple[float, float, float]] = {}

    def wait_while_is_moving(
        self,
        actuator: str | Actuators,
        timeout: float | None = 5.0,
        check_interval: float = 0.1,
        position_tolerance: float = 0.0005,
    ):
        """
        Checks position after a delay, and blocks if position has changed.
        If `timeout` is None, will block indefinitely.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        def check_if_moved():
            """Checks movement, returns True if movement is detected."""
            time.sleep(check_interval)

            if actuator in [
                Actuators.left_wheel_vel,
                Actuators.right_wheel_vel,
                Actuators.base_rotate,
                Actuators.base_translate,
            ]:
                current_position = actuator.get_position_relative(
                    self.pull_status()
                )
                if actuator == Actuators.left_wheel_vel or actuator == Actuators.base_translate:
                    current_position = current_position[0]
                elif actuator == Actuators.right_wheel_vel:
                    current_position = current_position[1]
                elif actuator == Actuators.base_rotate:
                    current_position = current_position[2]
            else:
                current_position = actuator.get_position(self.pull_status())

            if not actuator in self._last_movement_positions:
                self._last_movement_positions[actuator] = current_position
                return True

            last_position = self._last_movement_positions[actuator]

            is_moved = not np.isclose(current_position, last_position, atol=position_tolerance)

            self._last_movement_positions[actuator] = current_position

            return is_moved

        if not block_until_check_succeeds(
            wait_timeout=timeout,
            check=lambda: check_if_moved() == False,
            is_alive=self.is_running,
        ):
            if timeout is not None:
                click.secho(
                    f"Timeout: Joint {actuator.name} is still moving after {timeout}.",
                    fg="red",
                )
            return False
        return True

    @require_connection
    def move_to(self, actuator: str | Actuators, pos: float) -> None:
        """
        Move the actuator to an absolute position.
        Args:
            actuator: string name of the actuator or Actuator enum instance
            pos: float, absolute position goal

        Use `wait_until_at_setpoint()` or `wait_while_is_moving()` to block until the joint reaches its location.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
            Actuators.base_rotate,
            Actuators.base_translate,
        ]:
            raise Exception(
                f"Cannot set an absolute position for a continuous joint {actuator.name}"
            )

        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_move_to(CommandMove(actuator_name=actuator.name, pos=pos, trigger=True))

            self.data_proxies.set_command(command)

    @require_connection
    def move_by(self, actuator: str | Actuators, pos: float):
        """
        Move the actuator by a relative amount.
        Args:
            actuator: string name of the actuator or Actuator enum instance
            pos: float, position to increment by

        Use `wait_until_at_setpoint()` or `wait_while_is_moving()` to block until the joint reaches its location.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [Actuators.left_wheel_vel, Actuators.right_wheel_vel]:
            click.secho(
                f"Cannot set a position for a velocity joint {actuator.name}",
                fg="red",
            )
            raise Exception(
                f"Cannot set an absolute position for a continuous joint {actuator.name}"
            )

        with self._command_lock:
            command = self.data_proxies.get_command()

            command.set_move_by(
                # We set the pos here, and not new_position, because this relative motion math is handled by mujoco_server:
                CommandMove(actuator_name=actuator.name, pos=pos, trigger=True)
            )

            self.data_proxies.set_command(command)

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
            command.set_base_velocity(
                CommandBaseVelocity(v_linear=v_linear, omega=omega, trigger=True)
            )

            self.data_proxies.set_command(command)

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
