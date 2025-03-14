from multiprocessing import Manager, Process
import copy
import multiprocessing
import platform
import signal
import sys
import time

import click
import numpy as np
from mujoco._structs import MjModel

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.cameras import StretchCameras
import stretch_mujoco.config as config
from stretch_mujoco.mujoco_server_passive import MujocoServerPassive
from stretch_mujoco.status import StretchCameraStatus, StretchStatus
import stretch_mujoco.utils as utils
from stretch_mujoco.utils import require_connection


class StretchMujocoSimulator:
    """
    Stretch Mujoco Simulator class for interfacing with the Mujoco Server.

    Calling `run()` will spawn a new process that runs `MujocoServer` that runs the actual simulator.

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
        self._running = False
        self._cameras_to_use = cameras_to_use

        signal.signal(signal.SIGTERM, self._stop_handler)
        signal.signal(signal.SIGINT, self._stop_handler)

        self._manager = Manager()
        self._stop_event = self._manager.Event()
        self._command = self._manager.dict({"val": {}})
        self._status = self._manager.dict({"val": StretchStatus.default().to_dict()})
        self._imagery = self._manager.dict({"val": StretchCameraStatus.default().to_dict()})

    def start(self, show_viewer_ui: bool = False, headless: bool = False) -> None:
        """
        Start the simulator

        Args:
            show_viewer_ui: bool, whether to show the Mujoco viewer UI
            headless: bool, whether to run the simulation in headless mode
        """
        if platform.system() == "Darwin":
            # On a mac, the process needs to be started with mjpython
            mjpython_path = sys.executable.replace("bin/python3", "bin/mjpython").replace(
                "bin/python", "bin/mjpython"
            )
            print(f"{mjpython_path=}")
            multiprocessing.set_executable(mjpython_path)

        self._server_process = Process(
            target=MujocoServerPassive.launch_server,
            args=(
                self.scene_xml_path,
                self.model,
                self.camera_hz,
                show_viewer_ui,
                headless,
                self._stop_event,
                self._command,
                self._status,
                self._imagery,
                self._cameras_to_use,
            ),
        )
        self._server_process.start()
        self._running = True
        click.secho("Starting Stretch Mujoco Simulator...", fg="green")
        while (self.pull_status().time == 0) or (self.pull_camera_data().time == 0):
            time.sleep(0.2)
        self.home()

    def _stop_handler(self, signum, frame):
        self.stop()

    def stop(self) -> None:
        """
        Stop the simulator
        """
        if not self._running:
            return
        click.secho(
            f"Stopping Stretch Mujoco Simulator... simulated runtime={self.pull_status().time:.1f}s",
            fg="red",
        )
        self._running = False
        self._stop_event.set()
        if self._server_process:
            self._server_process.join()

        exit(0)  # exit this script, otherwise it tries to keep running even though we're stopped.

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
    def move_to(self, actuator: Actuators, pos: float) -> None:
        """
        Move the actuator to a specific position
        Args:
            actuator_name: str, name of the actuator
            pos: float, absolute position goal
        """
        if not actuator.is_position_actuator:
            click.secho(
                f"Actuator {actuator} not recognized."
                f"\n Available position actuators: {Actuators.position_actuators()}",
                fg="red",
            )
            return
        if actuator in ["base_translate", "base_rotate"]:
            click.secho(f"{actuator} not allowed for move_to", fg="red")
            return

        self._command["val"] = {
            "move_to": {"actuator_name": actuator.name, "pos": pos, "trigger": True}
        }

    @require_connection
    def move_by(self, actuator: Actuators, pos: float) -> None:
        """
        Move the actuator by a specific amount
        Args:
            actuator_name: Actuators, name of the actuator
            pos: float, position to increment by
        """
        if not actuator.is_position_actuator:
            click.secho(
                f"Actuator {actuator} not recognized."
                f"\n Available position actuators: {Actuators.position_actuators()}",
                fg="red",
            )
            return

        self._command["val"] = {
            "move_by": {"actuator_name": actuator.name, "pos": pos, "trigger": True}
        }

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
        return StretchCameraStatus(**copy.copy(self._imagery["val"]))

    @require_connection
    def pull_status(self) -> StretchStatus:
        """
        Pull robot joint states from the simulator and return as a dictionary
        """
        return StretchStatus.from_dict(copy.copy(self._status["val"]))

    def is_running(self) -> bool:
        """
        Check if the simulator is running
        """
        return self._running
