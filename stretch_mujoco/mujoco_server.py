import contextlib
from dataclasses import dataclass
from multiprocessing.managers import DictProxy
import signal
import threading
import time
from typing import Callable

import click
import mujoco
import mujoco._functions
import mujoco._callbacks
import mujoco._render
import mujoco._enums
import mujoco.viewer
import numpy as np
from mujoco._structs import MjData, MjModel

from stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from stretch_mujoco.enums.stretch_cameras import StretchCameras
import stretch_mujoco.config as config
from stretch_mujoco.mujoco_server_camera_manager import (
    MujocoServerCameraManagerThreaded,
    MujocoServerCameraManagerSync,
)
from stretch_mujoco.datamodels.status_command import StatusCommand
import stretch_mujoco.utils as utils
from stretch_mujoco.utils import FpsCounter


@dataclass
class MujocoServerProxies:
    _command: "DictProxy[str, StatusCommand]"
    _status: "DictProxy[str, StatusStretchJoints]"
    _cameras: "DictProxy[str, StatusStretchCameras]"

    def __setattr__(self, name: str, value) -> None:
        try:
            super().__setattr__(name, value)
        except BrokenPipeError:
            ...

    def get_status(self):
        return self._status["val"]

    def set_status(self, value: StatusStretchJoints):
        self._status["val"] = value

    def get_command(self):
        return self._command["val"]

    def set_command(self, value: StatusCommand):
        self._command["val"] = value

    def get_cameras(self):
        return self._cameras["val"]

    def set_cameras(self, value: StatusStretchCameras):
        self._cameras["val"] = value


class MujocoServer:
    """
    Use `MucocoServer.launch_server()` to start the headless simulator.

    This uses the mujoco simulator in headless mode.
    """

    def __init__(
        self,
        scene_xml_path: str | None,
        model: MjModel | None,
        stop_mujoco_process_event: threading.Event,
        data_proxies: MujocoServerProxies,
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

        self._stop_mujoco_process_event = stop_mujoco_process_event

        self.data_proxies = data_proxies

        self.physics_fps_counter = FpsCounter()

        signal.signal(signal.SIGTERM, lambda num, h: self.request_to_stop())
        signal.signal(signal.SIGINT, lambda num, h: self.request_to_stop())

    def set_camera_manager(
        self,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
        *,
        use_camera_thread: bool,
        use_threadpool_executor: bool,
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
            self.camera_manager = MujocoServerCameraManagerThreaded(
                use_camera_thread=use_camera_thread,
                use_threadpool_executor=use_threadpool_executor,
                camera_hz=camera_hz,
                cameras_to_use=cameras_to_use,
                mujoco_server=self,
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
        stop_mujoco_process_event: threading.Event,
        data_proxies: MujocoServerProxies,
        cameras_to_use: list[StretchCameras],
    ):
        server = cls(scene_xml_path, model, stop_mujoco_process_event, data_proxies)
        server.run(
            show_viewer_ui=show_viewer_ui,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

    def run(
        self,
        show_viewer_ui: bool,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
    ):
        # self.__run_headless_simulation(camera_hz=camera_hz, cameras_to_use=cameras_to_use)
        self.__run_headless_simulation_with_physics_thread(
            camera_hz=camera_hz, cameras_to_use=cameras_to_use
        )

    def _is_requested_to_stop(self):
        try:
            return self._stop_mujoco_process_event.is_set()
        except (EOFError, BrokenPipeError):
            # We likely lost connection to the main process if we've hit this.
            return True

    def request_to_stop(self):
        try:
            self._stop_mujoco_process_event.set()
        except (EOFError, BrokenPipeError):
            # We likely lost connection to the main process if we've hit this.
            ...

    def close(self):
        """
        Clean up C++ resources
        """
        if isinstance(self.camera_manager, MujocoServerCameraManagerThreaded):
            self.camera_manager.cameras_thread.join()

        self.camera_manager.close()

    def _run_ui_simulation(self, show_viewer_ui: bool) -> None:
        """
        Run the simulation with the viewer
        """
        raise NotImplementedError(
            "This is headless mode. Use MujocoServerPassive or MujocoServerManaged to run the UI simulator."
        )

    def _physics_step(self, lock: contextlib.AbstractContextManager):
        """
        Calls mj_step and _ctrl_callback, and sleeps until the next timestep.
        """
        start_time = time.perf_counter()

        with lock:
            mujoco._functions.mj_step(self.mjmodel, self.mjdata)
            self._ctrl_callback(self.mjmodel, self.mjdata)

        time_until_next_step = self.mjmodel.opt.timestep - (time.perf_counter() - start_time)
        if time_until_next_step > 0:
            # Sleep to match the timestep.
            time.sleep(time_until_next_step)

    def _physics_loop(
        self, lock: contextlib.AbstractContextManager, termination_check: Callable[[], bool]
    ):
        """
        A loop to use when starting physics in a thread.
        """
        while termination_check():
            self._physics_step(lock=lock)

        click.secho("Physics Loop has terminated.", fg="red")

    def __run_headless_simulation(
        self, camera_hz: float, cameras_to_use: list[StretchCameras]
    ) -> None:
        """
        Run the simulation without the viewer headless.

        Headless mode manages its own `set_camera_manager()` call.
        """
        print("Running headless simulation...")

        self.set_camera_manager(
            use_camera_thread=False,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        while not self._is_requested_to_stop():
            self._physics_step(contextlib.nullcontext())
            self.camera_manager.pull_camera_data_at_camera_rate(is_sleep_until_ready=False)

        self.close()

    def __run_headless_simulation_with_physics_thread(
        self, camera_hz: float, cameras_to_use: list[StretchCameras]
    ) -> None:
        """
        Run the simulation without the viewer headless.

        Headless mode manages its own `set_camera_manager()` call.
        """
        print("Running headless simulation...")

        self.set_camera_manager(
            use_camera_thread=False,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        physics_thread = threading.Thread(
            target=self._physics_loop,
            args=(self.camera_manager.camera_lock, lambda: not self._is_requested_to_stop()),
            daemon=True,
        )
        physics_thread.start()

        while not self._is_requested_to_stop():
            self.camera_manager.pull_camera_data_at_camera_rate(is_sleep_until_ready=True)

        physics_thread.join()
        self.close()

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

        new_status = StatusStretchJoints.default()
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

        self.data_proxies.set_status(new_status)

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
        command_status = self.data_proxies.get_command()
        # move_by
        if command_status.move_by is not None:
            for command in command_status.move_by:
                if command.trigger:
                    actuator_name = command.actuator_name
                    pos = command.pos
                    if actuator_name in ["base_translate", "base_rotate"]:
                        if self._base_in_pos_motion:
                            self._stop_base_pos_tracking()
                            time.sleep(1 / 20)
                        if actuator_name == "base_translate":
                            threading.Thread(
                                target=self._base_translate_by, args=(pos,), daemon=True
                            ).start()
                        else:
                            threading.Thread(
                                target=self._base_rotate_by, args=(pos,), daemon=True
                            ).start()
                    else:
                        current_value = self.data_proxies.get_status()[actuator_name].pos
                        if actuator_name == "gripper":
                            self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(
                                current_value + pos
                            )
                        else:
                            self.mjdata.actuator(actuator_name).ctrl = current_value + pos

        # move_to
        if command_status.move_to is not None:
            for command in command_status.move_to:
                if command.trigger:
                    actuator_name = command.actuator_name
                    pos = command.pos
                    if actuator_name == "gripper":
                        self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(pos)
                    else:
                        self.mjdata.actuator(actuator_name).ctrl = pos

        # set_base_velocity
        if (
            command_status.set_base_velocity is not None
            and command_status.set_base_velocity.trigger
        ):
            self.set_base_velocity(
                command_status.set_base_velocity.v_linear,
                command_status.set_base_velocity.omega,
            )

        # keyframe
        if command_status.keyframe is not None and command_status.keyframe.trigger:
            self.mjdata.ctrl = self.mjmodel.keyframe(command_status.keyframe.name).ctrl

        self.data_proxies.set_command(StatusCommand.default())

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
