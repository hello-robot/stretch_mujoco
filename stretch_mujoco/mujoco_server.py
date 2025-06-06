import contextlib
from dataclasses import dataclass
from multiprocessing.managers import DictProxy, SyncManager
import signal
import threading
import time
from typing import Callable

import click
import mujoco
import mujoco._functions
import mujoco._enums
import numpy as np
from mujoco._structs import MjData, MjModel
import mujoco._enums

from stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from stretch_mujoco.datamodels.status_stretch_sensors import StatusStretchSensors
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
import stretch_mujoco.config as config
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.mujoco_server_camera_manager import (
    MujocoServerCameraManagerThreaded,
    MujocoServerCameraManagerSync,
)
from stretch_mujoco.datamodels.status_command import CommandBaseVelocity, CommandMove, StatusCommand
from stretch_mujoco.mujoco_server_sensor_manager import MujocoServerSensorManagerThreaded
import stretch_mujoco.utils as utils
from stretch_mujoco.utils import FpsCounter


@dataclass
class MujocoServerProxies:
    _command: "DictProxy[str, StatusCommand]"
    _status: "DictProxy[str, StatusStretchJoints]"
    _cameras: "DictProxy[str, StatusStretchCameras]"
    _sensors: "DictProxy[str, StatusStretchSensors]"
    _joint_limits: "DictProxy[str, dict[Actuators, tuple[float, float]]]"

    def __setattr__(self, name: str, value) -> None:
        try:
            super().__setattr__(name, value)
        except BrokenPipeError:
            ...

    def get_status(self) -> StatusStretchJoints:
        return self._status["val"]

    def set_status(self, value: StatusStretchJoints):
        self._status["val"] = value

    def get_command(self) -> StatusCommand:
        return self._command["val"]

    def set_command(self, value: StatusCommand):
        self._command["val"] = value

    def get_cameras(self) -> StatusStretchCameras:
        return self._cameras["val"]

    def set_cameras(self, value: StatusStretchCameras):
        self._cameras["val"] = value

    def get_sensors(self) -> StatusStretchSensors:
        return self._sensors["val"]

    def set_sensors(self, value: StatusStretchSensors):
        self._sensors["val"] = value

    def get_joint_limits(self) -> dict[Actuators, tuple[float, float]]:
        return self._joint_limits["val"]

    def set_joint_limit(self, actuator: Actuators, min_max: tuple[float, float]):
        limits = self._joint_limits["val"]
        limits[actuator] = min_max

        self._joint_limits["val"] = limits

    @staticmethod
    def default(manager: SyncManager) -> "MujocoServerProxies":
        return MujocoServerProxies(
            _command=manager.dict({"val": StatusCommand.default()}),
            _status=manager.dict({"val": StatusStretchJoints.default()}),
            _cameras=manager.dict({"val": StatusStretchCameras.default()}),
            _sensors=manager.dict({"val": StatusStretchSensors.default()}),
            _joint_limits=manager.dict({"val": {}}),
        )


class BaseController:

    def __init__(self, mujoco_server: "MujocoServer") -> None:
        self.mujoco_server = mujoco_server
        self.last_command: CommandMove | CommandBaseVelocity | None = None
        self.start_pose = np.array([0, 0, 0])

    def push_command(self, command: CommandMove | CommandBaseVelocity):
        """Push a command to the base. Call `update()` to set the next trajectory."""
        self.last_command = command
        self.start_pose = self.get_base_pose()

    def _clear_command(self, is_stop_motion: bool):
        self.last_command = None

        if is_stop_motion:
            self._set_base_velocity(0.0, 0.0)

    def update(self):
        """
        The update method to set mujoco ctrl's for the base while in motion.
        """
        if self.last_command is None:
            return

        if isinstance(self.last_command, CommandMove):
            return self.handle_move_by(self.last_command)

        if isinstance(self.last_command, CommandBaseVelocity):
            return self._set_base_velocity(self.last_command.v_linear, self.last_command.omega)

    def get_base_pose(self) -> np.ndarray:
        """Get the se(2) base pose: x, y, and theta"""
        xyz = self.mujoco_server.mjdata.body("base_link").xpos
        rotation = self.mujoco_server.mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])

    def handle_move_by(self, command: CommandMove):
        if command.actuator_name == Actuators.base_translate.name:
            return self._base_translate_by(
                command.pos,
            )

        if command.actuator_name == Actuators.base_rotate.name:
            return self._base_rotate_by(
                command.pos,
            )

        raise NotImplementedError(f"Actuator {command.actuator_name} is not supported.")

    def _base_translate_by(self, x_inc: float) -> None:
        """
        Translate the base by a certain w.r.t base global pose
        """
        start_pose = self.start_pose[:2]

        sign = 1 if x_inc > 0 else -1
        if not np.linalg.norm(self.get_base_pose()[:2] - start_pose) <= abs(x_inc):
            return self._clear_command(is_stop_motion=True)

        self._set_base_velocity(config.base_motion["default_x_vel"] * sign, 0)

    def _base_rotate_by(self, theta_inc: float) -> None:
        """
        Rotate the base by a certain w.r.t base global pose
        """
        start_pose = self.start_pose[-1]
        sign = 1 if theta_inc > 0 else -1
        if not abs(start_pose - self.get_base_pose()[-1]) <= abs(theta_inc):
            return self._clear_command(is_stop_motion=True)

        self._set_base_velocity(0, config.base_motion["default_r_vel"] * sign)

    def _set_base_velocity(self, v_linear: float, omega: float) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """
        w_left, w_right = utils.diff_drive_inv_kinematics(v_linear, omega)
        self.mujoco_server.mjdata.actuator(Actuators.left_wheel_vel.name).ctrl = w_left
        self.mujoco_server.mjdata.actuator(Actuators.right_wheel_vel.name).ctrl = w_right


class MujocoServer:
    """
    Use `MucocoServer.launch_server()` to start the headless simulator.

    This uses the mujoco simulator in headless mode.
    """

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

        self.base_controller = BaseController(self)

        self.physics_fps_counter = FpsCounter()

        self.sensor_manager = MujocoServerSensorManagerThreaded(
            sensor_hz=15,
            sensors_to_use=StretchSensors.from_mjmodel(self.mjmodel),
            mujoco_server=self,
        )

        self.update_joint_limits()

        signal.signal(signal.SIGTERM, lambda num, h: self.request_to_stop())
        signal.signal(signal.SIGINT, lambda num, h: self.request_to_stop())

    def update_joint_limits(self):
        for i in range(self.mjmodel.njnt):
            name = mujoco._functions.mj_id2name(self.mjmodel, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            joint_range = self.mjmodel.jnt_range[i]  # This gives [lower_limit, upper_limit]
            try:
                actuator = Actuators.get_actuator_by_joint_names_in_mjcf(name)
                self.data_proxies.set_joint_limit(
                    actuator=actuator, min_max=(joint_range[0], joint_range[1])
                )
            except:
                ...

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
        self.request_to_stop()

        if isinstance(self.camera_manager, MujocoServerCameraManagerThreaded):
            self.camera_manager.cameras_thread.join()

        if isinstance(self.sensor_manager, MujocoServerSensorManagerThreaded):
            self.sensor_manager.sensors_thread.join()

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
        self.mjdata = data
        self.mjmodel = model

        if not self.mjdata or not self.mjdata.time:
            print("WARNING: no mujoco data to report")
            return

        self.physics_fps_counter.tick(sim_time=data.time)
        self.pull_status()
        self.push_command(self.data_proxies.get_command())

    def pull_status(self):
        """
        Pull joints status of the robot from the simulator
        """

        new_status = StatusStretchJoints.default()
        new_status.fps = self.physics_fps_counter.fps

        new_status.time = self.mjdata.time
        new_status.sim_to_real_time_ratio_msg = self.physics_fps_counter.sim_to_real_time_ratio_msg
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

        new_status.gripper.pos = self._to_real_gripper_range(
            self.mjdata.actuator("gripper").length[0]
        )
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
        ) = self.base_controller.get_base_pose()

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

    def push_command(self, command_status:StatusCommand):
        """
        Handles setting mujoco ctrl properties to move joints.
        """
        # move_by
        for _, command in command_status.move_by.items():
            if command.trigger:
                command.trigger = False
                actuator_name = command.actuator_name
                pos = command.pos
                if actuator_name in (Actuators.base_translate.name, Actuators.base_rotate.name):
                    self.base_controller.push_command(command)
                else:
                    if actuator_name == Actuators.gripper.name:
                        current_value = self._to_real_gripper_range(
                            self.mjdata.actuator("gripper").length[0]
                        )
                        self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(
                            current_value + pos
                        )
                    else:
                        current_value = self.mjdata.actuator(actuator_name).length[0]
                        self.mjdata.actuator(actuator_name).ctrl = current_value + pos

        # move_to
        for _, command in command_status.move_to.items():
            if command.trigger:
                command.trigger = False
                actuator_name = command.actuator_name
                pos = command.pos
                if actuator_name == Actuators.gripper.name:
                    self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(pos)
                elif actuator_name in (Actuators.base_translate.name, Actuators.base_rotate.name):
                    raise NotImplementedError(
                        f"Cannot set move_to for {actuator_name}, which is a relative joint."
                    )
                else:
                    self.mjdata.actuator(actuator_name).ctrl = pos

        # set_base_velocity
        if command_status.base_velocity is not None and command_status.base_velocity.trigger:
            command_status.base_velocity.trigger = False
            self.base_controller.push_command(command_status.base_velocity)

        # keyframe
        if command_status.keyframe is not None and command_status.keyframe.trigger:
            command_status.keyframe.trigger = False
            self.mjdata.ctrl = self.mjmodel.keyframe(command_status.keyframe.name).ctrl

        self.base_controller.update()

        self.data_proxies.set_command(command_status)

    def _to_sim_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to sim gripper range
        """
        return utils.map_between_ranges(
            pos,
            config.robot_settings["gripper_min_max"],
            config.robot_settings["sim_gripper_min_max"],
        )
