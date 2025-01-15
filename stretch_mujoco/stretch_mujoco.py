from multiprocessing import Manager, Process
import os
import copy
import signal
import threading
import time
from typing import Any, Dict, Optional

import click
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjData, MjModel

import stretch_mujoco.config as config
import stretch_mujoco.utils as utils
from stretch_mujoco.utils import require_connection


def launch_server(scene_xml_path, model, show_viewer_ui, headless, stop_event, command, status, imagery):
    server = MujocoServer(scene_xml_path, model, stop_event, command, status, imagery)
    server.run(show_viewer_ui, headless)


class MujocoServer:

    def __init__(self, scene_xml_path, model, stop_event, command, status, imagery):
        """
        Initialize the Simulator handle with a scene
        Args:
            scene_xml_path: str, path to the scene xml file
            model: MjModel, Mujoco model object
        """
        if scene_xml_path is None:
            scene_xml_path = utils.default_scene_xml_path
            self.mjmodel = mujoco.MjModel.from_xml_path(scene_xml_path)
        elif model is None:
            self.mjmodel = mujoco.MjModel.from_xml_path(scene_xml_path)
        if model is not None:
            self.mjmodel = model
        self.mjdata = mujoco.MjData(self.mjmodel)
        self._set_camera_properties()

        self.rgb_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer.enable_depth_rendering()

        self.viewer = mujoco.viewer
        self._base_in_pos_motion = False

        self.stop_event = stop_event
        self.command = command
        self.status = status
        self.imagery = imagery

    def run(self, show_viewer_ui, headless):
        if headless:
            self.__run_headless_simulation()
        else:
            self.__run(show_viewer_ui)

    def __run(self, show_viewer_ui: bool) -> None:
        """
        Run the simulation with the viewer
        """
        mujoco.set_mjcb_control(self.__ctrl_callback)
        self.viewer.launch(
            self.mjmodel,
            show_left_ui=show_viewer_ui,
            show_right_ui=show_viewer_ui,
        )

    def __run_headless_simulation(self) -> None:
        """
        Run the simulation without the viewer headless
        """
        print("Running headless simulation...")
        while not self.stop_event.is_set():
            start_ts = time.perf_counter()
            mujoco.mj_step(self.mjmodel, self.mjdata)
            self.__ctrl_callback(self.mjmodel, self.mjdata)
            elapsed = time.perf_counter() - start_ts
            if elapsed < self.mjmodel.opt.timestep:
                time.sleep(self.mjmodel.opt.timestep - elapsed)

    def __ctrl_callback(self, model: MjModel, data: MjData) -> None:
        """
        Callback function that gets executed with mj_step
        """
        if self.stop_event.is_set():
            os.kill(os.getpid(), 9)
        self.mjdata = data
        self.mjmodel = model
        self.pull_status()
        # self.pull_camera_data()
        self.push_command()

    def get_base_pose(self) -> np.ndarray:
        """Get the se(2) base pose: x, y, and theta"""
        xyz = self.mjdata.body("base_link").xpos
        rotation = self.mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])

    def pull_status(self) -> Dict[str, Any]:
        """
        Pull joints status of the robot from the simulator
        """
        new_status = {
            "time": None,
            "base": {"x": None, "y": None, "theta": None, "x_vel": None, "theta_vel": None},
            "lift": {"pos": None, "vel": None},
            "arm": {"pos": None, "vel": None},
            "head_pan": {"pos": None, "vel": None},
            "head_tilt": {"pos": None, "vel": None},
            "wrist_yaw": {"pos": None, "vel": None},
            "wrist_pitch": {"pos": None, "vel": None},
            "wrist_roll": {"pos": None, "vel": None},
            "gripper": {"pos": None, "vel": None},
        }
        new_status["time"] = self.mjdata.time
        new_status["lift"]["pos"] = self.mjdata.actuator("lift").length[0]
        new_status["lift"]["vel"] = self.mjdata.actuator("lift").velocity[0]

        new_status["arm"]["pos"] = self.mjdata.actuator("arm").length[0]
        new_status["arm"]["vel"] = self.mjdata.actuator("arm").velocity[0]

        new_status["head_pan"]["pos"] = self.mjdata.actuator("head_pan").length[0]
        new_status["head_pan"]["vel"] = self.mjdata.actuator("head_pan").velocity[0]

        new_status["head_tilt"]["pos"] = self.mjdata.actuator("head_tilt").length[0]
        new_status["head_tilt"]["vel"] = self.mjdata.actuator("head_tilt").velocity[0]

        new_status["wrist_yaw"]["pos"] = self.mjdata.actuator("wrist_yaw").length[0]
        new_status["wrist_yaw"]["vel"] = self.mjdata.actuator("wrist_yaw").velocity[0]

        new_status["wrist_pitch"]["pos"] = self.mjdata.actuator("wrist_pitch").length[0]
        new_status["wrist_pitch"]["vel"] = self.mjdata.actuator("wrist_pitch").velocity[0]

        new_status["wrist_roll"]["pos"] = self.mjdata.actuator("wrist_roll").length[0]
        new_status["wrist_roll"]["vel"] = self.mjdata.actuator("wrist_roll").velocity[0]

        real_gripper_pos = self._to_real_gripper_range(self.mjdata.actuator("gripper").length[0])
        new_status["gripper"]["pos"] = real_gripper_pos
        new_status["gripper"]["vel"] = self.mjdata.actuator("gripper").velocity[
            0
        ]  # This is still in sim gripper range

        left_wheel_vel = self.mjdata.actuator("left_wheel_vel").velocity[0]
        right_wheel_vel = self.mjdata.actuator("right_wheel_vel").velocity[0]
        (
            new_status["base"]["x_vel"],
            new_status["base"]["theta_vel"],
        ) = utils.diff_drive_fwd_kinematics(left_wheel_vel, right_wheel_vel)
        (
            new_status["base"]["x"],
            new_status["base"]["y"],
            new_status["base"]["theta"],
        ) = self.get_base_pose()

        self.status['val'] = new_status

    def _to_real_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to real gripper range
        """
        return utils.map_between_ranges(
            pos,
            config.robot_settings["sim_gripper_min_max"],
            config.robot_settings["gripper_min_max"],
        )

    def get_base_pose(self):
        xyz = self.mjdata.body("base_link").xpos
        rotation = self.mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])


    def pull_camera_data(self):
        """
        Pull camera data from the simulator and return as a dictionary
        """
        self.imagery["time"] = self.mjdata.time

        self.rgb_renderer.update_scene(self.mjdata, "d405_rgb")
        self.depth_renderer.update_scene(self.mjdata, "d405_rgb")

        self.imagery["cam_d405_rgb"] = self.rgb_renderer.render()
        self.imagery["cam_d405_depth"] = utils.limit_depth_distance(
            self.depth_renderer.render(), config.depth_limits["d405"]
        )
        self.imagery["cam_d405_K"] = self.get_camera_params("d405_rgb")

        self.rgb_renderer.update_scene(self.mjdata, "d435i_camera_rgb")
        self.depth_renderer.update_scene(self.mjdata, "d435i_camera_rgb")

        self.imagery["cam_d435i_rgb"] = self.rgb_renderer.render()
        self.imagery["cam_d435i_depth"] = utils.limit_depth_distance(
            self.depth_renderer.render(), config.depth_limits["d435i"]
        )
        self.imagery["cam_d435i_K"] = self.get_camera_params("d435i_camera_rgb")

        self.rgb_renderer.update_scene(self.mjdata, "nav_camera_rgb")
        self.imagery["cam_nav_rgb"] = self.rgb_renderer.render()

    def get_camera_params(self, camera_name: str) -> np.ndarray:
        """
        Get camera parameters
        """
        cam = self.mjmodel.camera(camera_name)
        d = {
            "fovy": cam.fovy,
            "f": self.mjmodel.cam_intrinsic[cam.id][:2],
            "p": self.mjmodel.cam_intrinsic[cam.id][2:],
            "res": self.mjmodel.cam_resolution[cam.id],
        }
        K = utils.compute_K(d["fovy"][0], d["res"][0], d["res"][1])
        return K

    def set_camera_params(self, camera_name: str, fovy: float, res: tuple) -> None:
        """
        Set camera parameters
        Args:
            camera_name: str, name of the camera
            fovy: float, vertical field of view in degrees
            res: tuple, size of the camera Image
        """
        cam = self.mjmodel.camera(camera_name)
        self.mjmodel.cam_fovy[cam.id] = fovy
        self.mjmodel.cam_resolution[cam.id] = res

    def _set_camera_properties(self):
        """
        Set the camera properties
        """
        for camera_name, settings in config.camera_settings.items():
            self.set_camera_params(
                camera_name, settings["fovy"], (settings["width"], settings["height"])
            )

    def push_command(self):
        # move_by
        if 'move_by' in self.command['val'] and self.command['val']['move_by']['trigger']:
            actuator_name = self.command['val']['move_by']['actuator_name']
            pos = self.command['val']['move_by']['pos']
            if actuator_name in ["base_translate", "base_rotate"]:
                if self._base_in_pos_motion:
                    self._stop_base_pos_tracking()
                    time.sleep(1 / 20)
                if actuator_name == "base_translate":
                    threading.Thread(target=self._base_translate_by, args=(pos,)).start()
                else:
                    threading.Thread(target=self._base_rotate_by, args=(pos,)).start()
            else:
                if actuator_name == "gripper":
                    self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(
                        self.status['val'][actuator_name]["pos"] + pos
                    )
                else:
                    self.mjdata.actuator(actuator_name).ctrl = (
                        self.status['val'][actuator_name]["pos"] + pos
                    )
            self.command['val'] = {}

        # move_to
        if 'move_to' in self.command['val'] and self.command['val']['move_to']['trigger']:
            actuator_name = self.command['val']['move_to']['actuator_name']
            pos = self.command['val']['move_to']['pos']
            if actuator_name == "gripper":
                self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(pos)
            else:
                self.mjdata.actuator(actuator_name).ctrl = pos
            self.command['val'] = {}

        # set_base_velocity
        if 'set_base_velocity' in self.command['val'] and self.command['val']['set_base_velocity']['trigger']:
            self.set_base_velocity(self.command['val']['set_base_velocity']['v_linear'], self.command['val']['set_base_velocity']['omega'])
            self.command['val'] = {}

        # keyframe
        if 'keyframe' in self.command['val'] and self.command['val']['keyframe']['trigger']:
            name = self.command['val']['keyframe']['name']
            self.mjdata.ctrl = self.mjmodel.keyframe(name).ctrl
            self.command['val'] = {}

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


class StretchMujocoSimulator:
    """
    Stretch Mujoco Simulator class for interfacing with the Mujoco simulator
    """

    def __init__(
        self, scene_xml_path: Optional[str] = None, model: Optional[MjModel] = None
    ) -> None:
        self.scene_xml_path = scene_xml_path
        self.model = model
        self.urdf_model = utils.URDFmodel()
        self._server_process = None
        self._manager = None
        self._command = None
        self._status = None
        self._imagery = None
        self._running = False
        self._stop_event = None

    @require_connection
    def home(self) -> None:
        """
        Move the robot to home position
        """
        self._command['val'] = {
            'keyframe': {
                'name': 'home',
                'trigger': True
            }
        }

    @require_connection
    def stow(self) -> None:
        """
        Move the robot to stow position
        """
        self._command['val'] = {
            'keyframe': {
                'name': 'stow',
                'trigger': True
            }
        }

    @require_connection
    def move_to(self, actuator_name: str, pos: float) -> None:
        """
        Move the actuator to a specific position
        Args:
            actuator_name: str, name of the actuator
            pos: float, absolute position goal
        """
        if actuator_name not in config.allowed_position_actuators:
            click.secho(
                f"Actuator {actuator_name} not recognized."
                f"\n Available position actuators: {config.allowed_position_actuators}",
                fg="red",
            )
            return
        if actuator_name in ["base_translate", "base_rotate"]:
            click.secho(f"{actuator_name} not allowed for move_to", fg="red")
            return

        self._command['val'] = {
            'move_to': {
                'actuator_name': actuator_name,
                'pos': pos,
                'trigger': True
            }
        }

    @require_connection
    def move_by(self, actuator_name: str, pos: float) -> None:
        """
        Move the actuator by a specific amount
        Args:
            actuator_name: str, name of the actuator
            pos: float, position to increment by
        """
        if actuator_name not in config.allowed_position_actuators:
            click.secho(
                f"Actuator {actuator_name} not recognized."
                f"\n Available position actuators: {config.allowed_position_actuators}",
                fg="red",
            )
            return

        self._command['val'] = {
            'move_by': {
                'actuator_name': actuator_name,
                'pos': pos,
                'trigger': True
            }
        }

    @require_connection
    def set_base_velocity(self, v_linear: float, omega: float) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """
        self._command['val'] = {
            'set_base_velocity': {
                'v_linear': v_linear,
                'omega': omega,
                'trigger': True
            }
        }

    @require_connection
    def get_base_pose(self) -> np.ndarray:
        """Get the se(2) base pose: x, y, and theta"""
        status = self.pull_status()
        return (status['base']['x'], status['base']['y'], status['base']['theta'])

    @require_connection
    def get_ee_pose(self) -> np.ndarray:
        return self.get_link_pose("link_grasp_center")

    @require_connection
    def get_link_pose(self, link_name: str) -> np.ndarray:
        """Pose of link in world frame"""
        status = self.pull_status()
        cfg = {
            "wrist_yaw": status["wrist_yaw"]["pos"],
            "wrist_pitch": status["wrist_pitch"]["pos"],
            "wrist_roll": status["wrist_roll"]["pos"],
            "lift": status["lift"]["pos"],
            "arm": status["arm"]["pos"],
            "head_pan": status["head_pan"]["pos"],
            "head_tilt": status["head_tilt"]["pos"],
        }
        T = self.urdf_model.get_transform(cfg, link_name)
        base_xyt = self.get_base_pose()
        base_4x4 = np.eye(4)
        base_4x4[:3, :3] = utils.Rz(base_xyt[2])
        base_4x4[:2, 3] = base_xyt[:2]
        world_coord = np.matmul(base_4x4, T)
        return world_coord

    @require_connection
    def pull_camera_data(self) -> dict:
        """
        Pull camera data from the simulator and return as a dictionary
        """
        return copy.copy(self._imagery)

    @require_connection
    def pull_status(self) -> dict:
        """
        Pull robot joint states from the simulator and return as a dictionary
        """
        return copy.copy(self._status['val'])

    def is_running(self) -> bool:
        """
        Check if the simulator is running
        """
        return self._running

    def start(self, show_viewer_ui: bool = False, headless: bool = False) -> None:
        """
        Start the simulator

        Args:
            show_viewer_ui: bool, whether to show the Mujoco viewer UI
            headless: bool, whether to run the simulation in headless mode
        """
        signal.signal(signal.SIGTERM, self._stop_handler)
        signal.signal(signal.SIGINT, self._stop_handler)
        self._manager = Manager()
        self._stop_event = self._manager.Event()
        # {
        #     "move_by" : {
        #         "trigger": None,
        #         "actuator_name": None,
        #         "pos": None,
        #     },
        #     "move_to" : {
        #         "trigger": None,
        #         "actuator_name": None,
        #         "pos": None,
        #     },
        #     "set_base_velocity" : {
        #         "trigger": None,
        #         "v_linear": None,
        #         "omega": None,
        #     },
        #     "keyframe" : {
        #         "trigger": None,
        #         "name": None,
        #     },
        # }
        self._command = self._manager.dict({'val': {}})
        self._status = self._manager.dict({'val': {
            "time": None,
            "base": {"x": None, "y": None, "theta": None, "x_vel": None, "theta_vel": None},
            "lift": {"pos": None, "vel": None},
            "arm": {"pos": None, "vel": None},
            "head_pan": {"pos": None, "vel": None},
            "head_tilt": {"pos": None, "vel": None},
            "wrist_yaw": {"pos": None, "vel": None},
            "wrist_pitch": {"pos": None, "vel": None},
            "wrist_roll": {"pos": None, "vel": None},
            "gripper": {"pos": None, "vel": None},
        }})
        self._imagery = self._manager.dict({
            "time": None,
            "cam_d405_rgb": None,
            "cam_d405_depth": None,
            "cam_d405_K": None,
            "cam_d435i_rgb": None,
            "cam_d435i_depth": None,
            "cam_d435i_K": None,
            "cam_nav_rgb": None,
        })
        self._server_process = Process(target=launch_server, args=(self.scene_xml_path, self.model, show_viewer_ui, headless, self._stop_event, self._command, self._status, self._imagery))
        self._server_process.start()
        self._running = True
        click.secho("Starting Stretch Mujoco Simulator...", fg="green")
        while not self.pull_status()["time"]:
            time.sleep(0.2)
        self.home()

    # def reset_sim(self) -> None:
    #     """
    #     Reset the simulator to initial state (experimental)
    #     """
    #     _headless_reset = self._headless_running
    #     if self._headless_running:
    #         self._headless_running = False
    #         time.sleep(0.3)
    #     else:
    #         click.secho(
    #             "StretchMujocoSimulator.reset_state() method is experimental with Viewer running",
    #             fg="yellow",
    #         )
    #     mujoco.mj_resetData(self.mjmodel, self.mjdata)
    #     print("Resetting the simulator to initial state...")
    #     if _headless_reset:
    #         threading.Thread(
    #             target=self.__run_headless_simulation, name="mujoco_headless_thread"
    #         ).start()
    #     while not self.mjdata.time:
    #         time.sleep(0.2)
    #     self.home()

    def _stop_handler(self, signum, frame):
        self.stop()

    def stop(self) -> None:
        """
        Stop the simulator
        """
        if not self._running:
            return
        click.secho(
            f"Stopping Stretch Mujoco Simulator... simulated runtime={self.pull_status()['time']:.1f}s",
            fg="red",
        )
        self._running = False
        self._stop_event.set()
        self._server_process.join()
        self._stop_event = None
        self._server_process = None
        self._manager = None
        self._command = None
        self._status = None
        self._imagery = None
