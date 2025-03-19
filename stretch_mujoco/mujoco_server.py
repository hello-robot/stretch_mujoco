from multiprocessing.managers import DictProxy
import threading
import time

import click
import mujoco
import mujoco._functions
import mujoco._callbacks
import mujoco._render
import mujoco._enums
import mujoco.viewer
import numpy as np
from mujoco._structs import MjData, MjModel

from stretch_mujoco.enums.stretch_cameras import StretchCamera
import stretch_mujoco.config as config
from stretch_mujoco.mujoco_server_camera_manager import (
    MujocoServerCameraManagerThreaded,
    MujocoServerCameraManagerSync,
)
from stretch_mujoco.status import StretchStatus
import stretch_mujoco.utils as utils
from stretch_mujoco.utils import FpsCounter


class MujocoServer:
    """
    Use `MucocoServer.launch_server()` to start the headless simulator.

    This uses the mujoco simulator in headless mode.
    """

    def __init__(
        self,
        scene_xml_path: str | None,
        model: MjModel | None,
        stop_event: threading.Event,
        command: DictProxy,
        status: DictProxy,
        imagery: DictProxy,
    ):
        """
        Initialize the Simulator handle with a scene
        Args:
            scene_xml_path: str, path to the scene xml file
            model: MjModel, Mujoco model object
        """
        if scene_xml_path is None:
            scene_xml_path = utils.default_scene_xml_path
            self.mjmodel = MjModel.from_xml_path(scene_xml_path)
        elif model is None:
            self.mjmodel = MjModel.from_xml_path(scene_xml_path)
        if model is not None:
            self.mjmodel = model
        self.mjdata = MjData(self.mjmodel)

        self._base_in_pos_motion = False

        self.stop_event = stop_event
        self.command = command
        self.status = status
        self.cameras = imagery

        self.physics_fps_counter = FpsCounter()

    def set_camera_manager(
        self, use_camera_thread: bool, camera_hz: float, cameras_to_use: list[StretchCamera], *, use_threadpool_executor: bool
    ):
        """
        This should be called before trying to render offscreen cameras.

        If `use_camera_thread` is false, `self.camera_manager.pull_camera_data_at_camera_rate()` should be called on a UI thread. 
        This is the recommended usage.

        If `use_camera_thread` is true, a thread will be spawned to call Renderer.render().
        This may not work on all platforms since rendering should happen on the main thread.
        This mode is mainly used with the Mujoco Managed Viewer, to avoid rendering on the physics thread.
        """
        if use_camera_thread or use_threadpool_executor:
            self.camera_manager = MujocoServerCameraManagerThreaded(use_camera_thread=use_camera_thread,use_threadpool_executor=use_threadpool_executor,
                camera_hz=camera_hz, cameras_to_use=cameras_to_use, mujoco_server=self
            )
        else:
            self.camera_manager = MujocoServerCameraManagerSync(
                camera_hz=camera_hz, cameras_to_use=cameras_to_use, mujoco_server=self
            )

    @classmethod
    def launch_server(
        cls,
        scene_xml_path: str | None,
        model: MjModel | None,
        camera_hz: float,
        show_viewer_ui: bool,
        stop_event: threading.Event,
        command: DictProxy,
        status: DictProxy,
        imagery: DictProxy,
        cameras_to_use: list[StretchCamera],
    ):
        server = cls(scene_xml_path, model, stop_event, command, status, imagery)
        server.run(
            show_viewer_ui=show_viewer_ui,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

    def run(
        self,
        show_viewer_ui: bool,
        camera_hz: float,
        cameras_to_use: list[StretchCamera],
    ):
        self.__run_headless_simulation(camera_hz=camera_hz, cameras_to_use=cameras_to_use)

    def close(self):
        """
        Clean up C++ resources
        """
        self.camera_manager.close()

    def _run_ui_simulation(self, show_viewer_ui: bool) -> None:
        """
        Run the simulation with the viewer
        """
        raise NotImplementedError("This is headless mode. Use MujocoServerPassive or MujocoServerManaged to run the UI simulator.")


    def __run_headless_simulation(self,
        camera_hz: float,
        cameras_to_use: list[StretchCamera]) -> None:
        """
        Run the simulation without the viewer headless.

        Headless mode manages its own `set_camera_manager()` call.
        """
        print("Running headless simulation...")

        self.set_camera_manager(
            use_camera_thread=False, 
            use_threadpool_executor=False,
            camera_hz=camera_hz, 
            cameras_to_use=cameras_to_use
        )

        while not self.stop_event.is_set():
            start_ts = time.perf_counter()
            mujoco._functions.mj_step(self.mjmodel, self.mjdata)
            self._ctrl_callback(self.mjmodel, self.mjdata)
            self.camera_manager.pull_camera_data_at_camera_rate()
            elapsed = time.perf_counter() - start_ts
            if elapsed < self.mjmodel.opt.timestep:
                time.sleep(self.mjmodel.opt.timestep - elapsed)


    def _ctrl_callback(self, model: MjModel, data: MjData) -> None:
        """
        Callback function that gets executed with mj_step
        """
        self.physics_fps_counter.tick()

        self.mjdata = data
        self.mjmodel = model
        self.pull_status()
        self.push_command()

    def get_base_pose(self) -> np.ndarray:
        """Get the se(2) base pose: x, y, and theta"""
        xyz = self.mjdata.body("base_link").xpos
        rotation = self.mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])

    def pull_status(self):
        """
        Pull joints status of the robot from the simulator
        """

        new_status = StretchStatus.default()
        new_status.fps = self.physics_fps_counter.fps

        if not self.mjdata or not self.mjdata.time:
            print("WARNING: no mujoco data to report")
            return

        new_status.time = self.mjdata.time
        new_status.lift.pos = self.mjdata.actuator("lift").length[0]
        new_status.lift.vel = self.mjdata.actuator("lift").velocity[0]

        new_status.arm.pos = self.mjdata.actuator("arm").length[0]
        new_status.arm.vel = self.mjdata.actuator("arm").velocity[0]

        new_status.head_pan.pos = self.mjdata.actuator("head_pan").length[0]
        new_status.head_pan.vel = self.mjdata.actuator("head_pan").velocity[0]

        new_status.head_tilt.pos = self.mjdata.actuator("head_tilt").length[0]
        new_status.head_tilt.vel = self.mjdata.actuator("head_tilt").velocity[0]

        new_status.wrist_yaw.pos = self.mjdata.actuator("wrist_yaw").length[0]
        new_status.wrist_yaw.vel = self.mjdata.actuator("wrist_yaw").velocity[0]

        new_status.wrist_pitch.pos = self.mjdata.actuator("wrist_pitch").length[0]
        new_status.wrist_pitch.vel = self.mjdata.actuator("wrist_pitch").velocity[0]

        new_status.wrist_roll.pos = self.mjdata.actuator("wrist_roll").length[0]
        new_status.wrist_roll.vel = self.mjdata.actuator("wrist_roll").velocity[0]

        real_gripper_pos = self._to_real_gripper_range(self.mjdata.actuator("gripper").length[0])
        new_status.gripper.pos = real_gripper_pos
        new_status.gripper.vel = self.mjdata.actuator("gripper").velocity[
            0
        ]  # This is still in sim gripper range

        left_wheel_vel = self.mjdata.actuator("left_wheel_vel").velocity[0]
        right_wheel_vel = self.mjdata.actuator("right_wheel_vel").velocity[0]
        (
            new_status.base.x_vel,
            new_status.base.theta_vel,
        ) = utils.diff_drive_fwd_kinematics(left_wheel_vel, right_wheel_vel)
        (
            new_status.base.x,
            new_status.base.y,
            new_status.base.theta,
        ) = self.get_base_pose()

        self.status["val"] = new_status.to_dict()

    def _to_real_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to real gripper range
        """
        return utils.map_between_ranges(
            pos,
            config.robot_settings["sim_gripper_min_max"],
            config.robot_settings["gripper_min_max"],
        )

    def push_command(self):
        # move_by
        if "move_by" in self.command["val"] and self.command["val"]["move_by"]["trigger"]:
            actuator_name = self.command["val"]["move_by"]["actuator_name"]
            pos = self.command["val"]["move_by"]["pos"]
            if actuator_name in ["base_translate", "base_rotate"]:
                if self._base_in_pos_motion:
                    self._stop_base_pos_tracking()
                    time.sleep(1 / 20)
                if actuator_name == "base_translate":
                    threading.Thread(target=self._base_translate_by, args=(pos,), daemon=True).start()
                else:
                    threading.Thread(target=self._base_rotate_by, args=(pos,), daemon=True).start()
            else:
                if actuator_name == "gripper":
                    self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(
                        self.status["val"][actuator_name]["pos"] + pos
                    )
                else:
                    self.mjdata.actuator(actuator_name).ctrl = (
                        self.status["val"][actuator_name]["pos"] + pos
                    )
            self.command["val"] = {}

        # move_to
        if "move_to" in self.command["val"] and self.command["val"]["move_to"]["trigger"]:
            actuator_name = self.command["val"]["move_to"]["actuator_name"]
            pos = self.command["val"]["move_to"]["pos"]
            if actuator_name == "gripper":
                self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(pos)
            else:
                self.mjdata.actuator(actuator_name).ctrl = pos
            self.command["val"] = {}

        # set_base_velocity
        if (
            "set_base_velocity" in self.command["val"]
            and self.command["val"]["set_base_velocity"]["trigger"]
        ):
            self.set_base_velocity(
                self.command["val"]["set_base_velocity"]["v_linear"],
                self.command["val"]["set_base_velocity"]["omega"],
            )
            self.command["val"] = {}

        # keyframe
        if "keyframe" in self.command["val"] and self.command["val"]["keyframe"]["trigger"]:
            name = self.command["val"]["keyframe"]["name"]
            self.mjdata.ctrl = self.mjmodel.keyframe(name).ctrl
            self.command["val"] = {}

    def _base_translate_by(self, x_inc: float) -> None:
        """
        Translate the base by a certain w.r.t base global pose
        """
        start_pose = self.get_base_pose()[:2]
        self._base_in_pos_motion = True
        sign = 1 if x_inc > 0 else -1
        start_ts = time.perf_counter()
        while np.linalg.norm(self.get_base_pose()[:2] - start_pose) <= abs(x_inc):
            if self._base_in_pos_motion:
                self.set_base_velocity(
                    config.base_motion["default_x_vel"] * sign, 0, _override=True
                )
                if time.perf_counter() - start_ts > config.base_motion["timeout"]:
                    click.secho("Base translation timeout", fg="red")
                    break
            else:
                break
            time.sleep(1 / 30)
        self.set_base_velocity(0, 0)
        self._base_in_pos_motion = False

    def _base_rotate_by(self, theta_inc: float) -> None:
        """
        Rotate the base by a certain w.r.t base global pose
        """
        start_pose = self.get_base_pose()[-1]
        self._base_in_pos_motion = True
        sign = 1 if theta_inc > 0 else -1
        start_ts = time.perf_counter()
        while abs(start_pose - self.get_base_pose()[-1]) <= abs(theta_inc):
            if self._base_in_pos_motion:
                self.set_base_velocity(
                    0, config.base_motion["default_r_vel"] * sign, _override=True
                )
                time.sleep(1 / 30)
                if time.perf_counter() - start_ts > config.base_motion["timeout"]:
                    click.secho("Base rotation timeout", fg="red")
                    break
            else:
                break
        self.set_base_velocity(0, 0)
        self._base_in_pos_motion = False

    def set_base_velocity(self, v_linear: float, omega: float, _override=False) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """
        if not _override and self._base_in_pos_motion:
            self._stop_base_pos_tracking()
            time.sleep(1 / 20)
        w_left, w_right = utils.diff_drive_inv_kinematics(v_linear, omega)
        self.mjdata.actuator("left_wheel_vel").ctrl = w_left
        self.mjdata.actuator("right_wheel_vel").ctrl = w_right

    def _to_sim_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to sim gripper range
        """
        return utils.map_between_ranges(
            pos,
            config.robot_settings["gripper_min_max"],
            config.robot_settings["sim_gripper_min_max"],
        )

    def _stop_base_pos_tracking(self) -> None:
        """
        Stop the base position tracking
        """
        self._base_in_pos_motion = False
