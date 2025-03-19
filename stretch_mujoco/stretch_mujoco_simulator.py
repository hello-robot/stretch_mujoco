import atexit
from multiprocessing import Manager, Process
import copy
import multiprocessing
import platform
import signal
import sys
import threading
import time


import click
import numpy as np
from mujoco._structs import MjModel

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCamera
from stretch_mujoco.mujoco_server import MujocoServer
from stretch_mujoco.mujoco_server_passive import MujocoServerPassive
from stretch_mujoco.status import StretchCameraStatus, StretchStatus
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
        cameras_to_use: list[StretchCamera] = [],
    ) -> None:
        self.scene_xml_path = scene_xml_path
        self.model = model
        self.camera_hz = camera_hz
        self.urdf_model = utils.URDFmodel()
        self._server_process = None
        self._running = False
        self._cameras_to_use = cameras_to_use

        self._manager = Manager()
        self._stop_event = self._manager.Event()
        self._command = self._manager.dict({"val": {}})
        self._status = self._manager.dict({"val": StretchStatus.default().to_dict()})
        self._cameras = self._manager.dict({"val": StretchCameraStatus.default().to_dict()})

    def start(self, show_viewer_ui: bool = False, headless: bool = False, use_passive_viewer: bool = False) -> None:
        """
        Start the simulator

        Args:
            show_viewer_ui: bool, whether to show the Mujoco viewer UI
            headless: bool, whether to run the simulation in headless mode
        """
        mujoco_server = MujocoServer if use_passive_viewer else MujocoServerPassive
        
        if platform.system() == "Darwin" and isinstance(mujoco_server, MujocoServerPassive):
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
                headless,
                self._stop_event,
                self._command,
                self._status,
                self._cameras,
                self._cameras_to_use,
            ),
            daemon=False,  # We're gonna handle terminating this in stop_mujoco_process()
        )
        self._server_process.start()

        # Handle stopping, in all its various ways:
        signal.signal(signal.SIGTERM, lambda num, sig: self.stop())
        signal.signal(signal.SIGINT, lambda num, sig: self.stop())
        atexit.register(self.stop)

        self._running = True
        click.secho("Starting Stretch Mujoco Simulator...", fg="green")
        while (self.pull_status().time == 0) or (self.pull_camera_data().time == 0):
            time.sleep(1)
            click.secho("Still waiting to connect to the Mujoco Simulatior.", fg="yellow")

        self.home()

    def stop(self) -> None:
        """
        This is called at exit to gracefully terminate the simulation and the Mujoco Process, and their many threads.

        Fingers-crossed we get a SIGTERM, and not a SIGKILL..
        """
        if not self._running:
            return

        simulation_time = self._status["val"]["time"]

        click.secho(
            f"Stopping Stretch Mujoco Simulator... simulated runtime={simulation_time:.1f}s",
            fg="red",
        )

        self._running = False

        # We're going to try to wait for threads to end. They might not gracefully stop before hitting an exception. Race conditions are rampant.
        # For example, the main thread or a thread may not be checking `sim.is_running()` and is oblivious that it should stop. Nothing we can do to stop it except sigkill.
        active_threads = threading.enumerate()
        for index, thread in enumerate(active_threads):
            if thread != threading.current_thread() and thread != threading.main_thread() and not isinstance(thread, threading._DummyThread):
                click.secho(
                    f"Stopping thread {index}/{len(active_threads)-1}.",
                    fg="yellow",
                )
                thread.join(timeout=2.0)
                if thread.is_alive():
                    click.secho(
                        f"{thread.name} is not terminating. Make sure to check 'sim.is_running()' in threading loops.",
                        fg="red",
                    )

        time.sleep(1)  # Not great, but wait for main thread to really settle.

        atexit.register(
            self.stop_mujoco_process
        )  # Calling it directly doesn't always work if the main thread isn't

    def stop_mujoco_process(self):
        click.secho(
            f"Sending signal to stop the Mujoco process...",
            fg="red",
        )

        # Wait until the main control loop ends before sending this stop event.
        self._stop_event.set()
        if self._server_process:
            # self._server_process.terminate() # ask it nicely.
            self._server_process.join()

        click.secho(
            f"The Mujoco process has ended. Good-bye!",
            fg="red",
        )

    @require_connection
    def home(self) -> None:
        """
        Move the robot to home position
        """
        self._command["val"] = {"keyframe": {"name": "home", "trigger": True}}

    @require_connection
    def stow(self) -> None:
        """
        Move the robot to stow position
        """
        self._command["val"] = {"keyframe": {"name": "stow", "trigger": True}}

    @require_connection
    def move_to(self, actuator: Actuators, pos: float, timeout: float | None = 15.0) :
        """
        Move the actuator to a specific position
        Args:
            actuator_name: str, name of the actuator
            pos: float, absolute position goal
            timeout: if not None, then it will wait for the joint to reach that position, or return False
        """
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
            return

        self._command["val"] = {
            "move_to": {"actuator_name": actuator.name, "pos": pos, "trigger": True}
        }

        if timeout:
            if not wait_and_check(
                timeout,
                lambda: np.isclose(actuator.get_position(self.pull_status()), pos, atol=0.05) == True,
                self.is_running
            ):
                click.secho(f"Joint {actuator.name} did not reach {pos}. Actual: {actuator.get_position(self.pull_status())}", fg="red")
                return False
        return True
            

    @require_connection
    def move_by(self, actuator: Actuators, pos: float) -> None:
        """
        Move the actuator by a specific amount
        Args:
            actuator_name: Actuators, name of the actuator
            pos: float, position to increment by
            timeout: if not None, then it will wait for the joint to reach that position, or throw.
        """
        if actuator in [Actuators.left_wheel_vel, Actuators.right_wheel_vel]:
            click.secho(
                f"Cannot set a position for a velocity joint {actuator.name}",
                fg="red",
            )
            return

        self._command["val"] = {
            "move_by": {"actuator_name": actuator.name, "pos": pos, "trigger": True}
        }

        # if timeout:
        #     if actuator in [Actuators.base_rotate, Actuators.base_translate]:

        #         initial_position = actuator.get_position_relative(self.pull_status())

        #         # TODO: implement the check for moving the base
        #         check = lambda: True
        #     else:   
        #         initial_position = actuator.get_position(self.pull_status())

        #         check = lambda: np.isclose(initial_position-actuator.get_position(self.pull_status()), pos,  atol=0.05) == True

        #     if not wait_and_check(
        #         timeout,
        #         check
        #     ):
        #         raise Exception(f"Joint {actuator.name} did not move by {pos}.")

    @require_connection
    def set_base_velocity(self, v_linear: float, omega: float) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """
        self._command["val"] = {
            "set_base_velocity": {"v_linear": v_linear, "omega": omega, "trigger": True}
        }

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
    def pull_camera_data(self) -> StretchCameraStatus:
        """
        Pull camera data from the simulator and return as a dictionary
        """
        return StretchCameraStatus(**copy.copy(self._cameras["val"]))

    @require_connection
    def pull_status(self) -> StretchStatus:
        """
        Pull robot joint states from the simulator and return as a dictionary
        """
        return StretchStatus.from_dict(copy.copy(self._status["val"]))

    def is_server_alive_or_stopevent_untriggered(self):
        return (
            self._server_process is not None
            and self._server_process.is_alive()
            and not self._stop_event.is_set()
        )

    def is_running(self) -> bool:
        """
        Check if the simulator and mujoco are running, or if the stopevent signal has been triggered.

        Side-effect here is that if the mujoco process is terminated or the stopevent is triggered, `self.stop()` is called.
        """
        if not self.is_server_alive_or_stopevent_untriggered():
            # Send the signal to stop the program:
            self.stop()
            return False

        return self._running
