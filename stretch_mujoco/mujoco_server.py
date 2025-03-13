from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.managers import DictProxy
import os
import platform
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

from stretch_mujoco.cameras import StretchCameras
import stretch_mujoco.config as config
from stretch_mujoco.status import StretchCameraStatus, StretchStatus
import stretch_mujoco.utils as utils
from stretch_mujoco.utils import FpsCounter, switch_to_glfw_renderer


class MujocoServer:
    """
    Use `MucocoServer.launch_server()` to start the simulator.

    This uses the mujoco managed viewer.

    https://mujoco.readthedocs.io/en/stable/python.html#managed-viewer
    """

    def __init__(
        self,
        scene_xml_path: str | None,
        model: MjModel | None,
        camera_hz: float,
        stop_event: threading.Event,
        command: DictProxy,
        status: DictProxy,
        imagery: DictProxy,
        cameras_to_use: list[StretchCameras],
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

        self.viewer = mujoco.viewer
        self._base_in_pos_motion = False

        self.stop_event = stop_event
        self.command = command
        self.status = status
        self.cameras = imagery

        self._set_camera_properties()

        self.camera_rate = camera_hz

        self.camera_renderers: dict[StretchCameras, mujoco.Renderer] = {}

        for camera in cameras_to_use:
            # Add this camera for the cameras_rendering_thread_pool to do rendering on
            self._add_camera_renderer(camera)

        if platform.system() == "Darwin":
            self.cameras_rendering_thread_pool = ThreadPoolExecutor(
                max_workers=len(cameras_to_use) if cameras_to_use else 1
            )
        else:
            # Linux is currently struggling with multi-threaded camera rendering:
            self.cameras_rendering_thread_pool = ThreadPoolExecutor(max_workers=1)

        self.cameras_thread = threading.Thread(target=self._camera_loop)
        self.cameras_thread.start()

        self.simulation_fps_counter = FpsCounter()
        self.camera_fps_counter = FpsCounter()

    @classmethod
    def launch_server(
        cls,
        scene_xml_path: str | None,
        model: MjModel | None,
        camera_hz: float,
        show_viewer_ui: bool,
        headless: bool,
        stop_event: threading.Event,
        command: DictProxy,
        status: DictProxy,
        imagery: DictProxy,
        cameras_to_use: list[StretchCameras],
    ):
        server = cls(
            scene_xml_path, model, camera_hz, stop_event, command, status, imagery, cameras_to_use
        )
        server.run(show_viewer_ui, headless)

    def run(self, show_viewer_ui, headless):
        if headless:
            self._run_headless_simulation()
        else:
            self._run(show_viewer_ui)

    def _run(self, show_viewer_ui: bool) -> None:
        """
        Run the simulation with the viewer
        """
        mujoco._callbacks.set_mjcb_control(self._ctrl_callback)
        self.viewer.launch(
            self.mjmodel,
            show_left_ui=show_viewer_ui,
            show_right_ui=show_viewer_ui,
        )

    def _run_headless_simulation(self) -> None:
        """
        Run the simulation without the viewer headless
        """
        print("Running headless simulation...")
        while not self.stop_event.is_set():
            start_ts = time.perf_counter()
            mujoco._functions.mj_step(self.mjmodel, self.mjdata)
            self._ctrl_callback(self.mjmodel, self.mjdata)
            elapsed = time.perf_counter() - start_ts
            if elapsed < self.mjmodel.opt.timestep:
                time.sleep(self.mjmodel.opt.timestep - elapsed)

    def _ctrl_callback(self, model: MjModel, data: MjData) -> None:
        """
        Callback function that gets executed with mj_step
        """
        self.simulation_fps_counter.tick()

        if self.stop_event.is_set():
            self.cameras_thread.join()
            os.kill(os.getpid(), 9)
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
        new_status.fps = self.simulation_fps_counter.fps

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

    def _create_camera_renderer(self, is_depth: bool):
        renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)

        if platform.system() == "Darwin":
            # On MacOS, switch to glfw because CGL is not compatible with offscreen rendering, and blocks the camera renderers
            switch_to_glfw_renderer(self.mjmodel, renderer)

        if is_depth:
            renderer.enable_depth_rendering()

        return renderer

    def _render_camera(self, renderer: mujoco.Renderer, camera: StretchCameras):
        """
        This calls update_scene and render() for an offscreen camera buffer.

        Use this with the _toggle_camera() functionality in this class.
        """

        renderer.update_scene(self.mjdata, camera.camera_name_in_scene)

        render = renderer.render()
        # render = np.array([])

        post_render = camera.post_processing_callback
        if post_render:
            render = post_render(render)

        return (camera, render)

    def _remove_camera_renderer(self, camera: StretchCameras):
        """
        When a camera is not needed, it's removed from self.camera_renderers to save computation costs.

        Note: `_add_camera_renderer()` creates a renderer and render params for the cameras the user wants to use.
        """
        if camera in self.camera_renderers:
            del self.camera_renderers[camera]
            return

        raise Exception(f"Camera {camera} was not in {self.camera_renderers=}")

    def _add_camera_renderer(self, camera: StretchCameras):
        """
        Creates a renderer and render params for the cameras the user wants to use.

        Note: `_remove_camera_renderer()` removes the renderer in self.camera_renderers to save computation costs.
        """
        if camera in self.camera_renderers:
            raise Exception(f"Camera {camera} is already in {self.camera_renderers=}")

        self.camera_renderers[camera] = self._create_camera_renderer(is_depth=camera.is_depth)

    def _camera_loop(self):
        """
        This is the thread loop that handles camera rendering.
        """
        while not self.status["val"] or not self.status["val"]["time"]:
            # wait for sim to start
            time.sleep(0.1)
        time_start = time.perf_counter()
        camera_sleep_time = 1 / self.camera_rate  # Hz to seconds
        while not self.stop_event.is_set():
            self.camera_fps_counter.tick()

            elapsed = time.perf_counter() - time_start
            if elapsed < camera_sleep_time:
                time.sleep(camera_sleep_time - elapsed)
                time_start = time.perf_counter()

            self._pull_camera_data()

    def _pull_camera_data(self):
        """
        Render a scene at each camera using the simulator and populate the imagery dictionary with the raw image pixels and camera params.
        """
        new_imagery = StretchCameraStatus.default()
        new_imagery.time = self.mjdata.time
        new_imagery.fps = self.camera_fps_counter.fps

        # This is a bit hard to read, so here's an explanation,
        # we're using self.imagery_thread_pool, which is a ThreadPoolExecutor to handle calling self._render_camera off the UI thread.
        # the parameters for self._render_camera are being fetched from self.camera_renderers and passed along the call:
        futures = as_completed(
            [
                self.cameras_rendering_thread_pool.submit(self._render_camera, renderer, camera)
                for (camera, renderer) in self.camera_renderers.items()
            ]
        )

        for future in futures:
            # Put the rendered image data into the new_imagery dictionary
            (camera, render) = future.result()
            new_imagery.set_camera_data(camera, render)

        new_imagery.cam_d405_K = self.get_camera_params("d405_rgb")
        new_imagery.cam_d435i_K = self.get_camera_params("d435i_camera_rgb")

        self.cameras["val"] = new_imagery.to_dict()

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
        if "move_by" in self.command["val"] and self.command["val"]["move_by"]["trigger"]:
            actuator_name = self.command["val"]["move_by"]["actuator_name"]
            pos = self.command["val"]["move_by"]["pos"]
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
