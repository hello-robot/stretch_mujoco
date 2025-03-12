from multiprocessing import Manager, Process
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.managers import DictProxy
import os
import copy
import platform
import signal
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
from stretch_mujoco.utils import FpsCounter, require_connection

from mujoco.glfw import GLContext as GlFwContext


def launch_server(
    scene_xml_path: str | None,
    model,
    camera_hz: config.CameraRates,
    show_viewer_ui: bool,
    headless: bool,
    stop_event: threading.Event,
    command: DictProxy,
    status: DictProxy,
    imagery: DictProxy,
    cameras_to_use: list[StretchCameras]
):
    server = MujocoServer(scene_xml_path, model, camera_hz, stop_event, command, status, imagery, cameras_to_use)
    server.run(show_viewer_ui, headless)


def switch_to_glfw_renderer(mjmodel: MjModel, renderer: mujoco.Renderer):
    """
    On Darwin, the default renderer in `mujoco/gl_context.py` is CGL, which is not compatible with offscreen rendering.

    This function frees the initial display context and creates a new one with GLFW.
    """
    if renderer._gl_context:
        renderer._gl_context.free()
    if renderer._mjr_context:
        renderer._mjr_context.free()

    renderer._gl_context = GlFwContext(480, 640)

    renderer._gl_context.make_current()

    renderer._mjr_context = mujoco._render.MjrContext(
        mjmodel, mujoco._enums.mjtFontScale.mjFONTSCALE_150.value
    )
    mujoco._render.mjr_setBuffer(
        mujoco._enums.mjtFramebuffer.mjFB_OFFSCREEN.value, renderer._mjr_context
    )
    renderer._mjr_context.readDepthMap = mujoco._enums.mjtDepthMap.mjDEPTH_ZEROFAR


class MujocoServer:

    def __init__(
        self,
        scene_xml_path: str | None,
        model,
        camera_hz: config.CameraRates,
        stop_event: threading.Event,
        command: DictProxy,
        status: DictProxy,
        imagery:DictProxy,
        cameras_to_use: list[StretchCameras]
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
        self._set_camera_properties()

        self.camera_rate = camera_hz

        self.camera_renderers: dict[StretchCameras, mujoco.Renderer] = {}

        for camera in cameras_to_use:
            self._toggle_camera(camera)

        self.viewer = mujoco.viewer
        self._base_in_pos_motion = False

        self.stop_event = stop_event
        self.command = command
        self.status = status
        self.imagery = imagery

        self.imagery_thread_pool = ThreadPoolExecutor(max_workers=5)

        self.imagery_thread = threading.Thread(target=self._imagery_loop)
        self.imagery_thread.start()

        self.fps_counter = FpsCounter()

    def run(self, show_viewer_ui, headless):
        if headless:
            self.__run_headless_simulation()
        else:
            self.__run(show_viewer_ui)

    def __run(self, show_viewer_ui: bool) -> None:
        """
        Run the simulation with the viewer
        """
        mujoco._callbacks.set_mjcb_control(self.__ctrl_callback)
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
            mujoco._functions.mj_step(self.mjmodel, self.mjdata)
            self.__ctrl_callback(self.mjmodel, self.mjdata)
            elapsed = time.perf_counter() - start_ts
            if elapsed < self.mjmodel.opt.timestep:
                time.sleep(self.mjmodel.opt.timestep - elapsed)

    def __ctrl_callback(self, model: MjModel, data: MjData) -> None:
        """
        Callback function that gets executed with mj_step
        """
        if self.stop_event.is_set():
            self.imagery_thread.join()
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
        self.fps_counter.tick()

        new_status = StretchStatus.default()
        new_status.fps = self.fps_counter.fps
        
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

        post_render = camera.post_processing_callback
        if post_render:
            render = post_render(render)

        return (camera, render)

    def _toggle_camera(self, camera: StretchCameras):
        """
        Creates a renderer and render params for the cameras the user wants to use.

        When a camera is toggled off, it's removed from self.camera_renderers to save computation costs.
        """
        if camera in self.camera_renderers:
            del self.camera_renderers[camera]
            return

        self.camera_renderers[camera] = self._create_camera_renderer(is_depth=camera.is_depth)

    def _imagery_loop(self):
        while not self.status["val"] or not self.status["val"]["time"]:
            time.sleep(0.1)
        while not self.stop_event.is_set():
            if int(self.status["val"]["time"] * 100) % self.camera_rate.value == 0:
                self._pull_camera_data()
            else:
                time.sleep(0.001)

    def _pull_camera_data(self):
        """
        Render a scene at each camera using the simulator and populate the imagery dictionary with the raw image pixels and camera params.
        """
        new_imagery = StretchCameraStatus.default()
        new_imagery.time = self.mjdata.time

        # This is a bit hard to read, so here's an explanation,
        # we're using self.imagery_thread_pool, which is a ThreadPoolExecutor to handle calling self._render_camera off the UI thread.
        # the parameters for self._render_camera are being fetched from self.camera_renderers and passed along the call:
        futures = as_completed(
            [
                self.imagery_thread_pool.submit(self._render_camera, renderer, camera)
                for (camera, renderer) in self.camera_renderers.items()
            ]
        )

        for future in futures:
            # Put the rendered image data into the new_imagery dictionary
            (camera, render) = future.result()
            new_imagery.set_camera_data(camera, render)

        new_imagery.cam_d405_K = self.get_camera_params("d405_rgb")
        new_imagery.cam_d435i_K = self.get_camera_params("d435i_camera_rgb")

        self.imagery["val"] = new_imagery.to_dict()

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


class StretchMujocoSimulator:
    """
    Stretch Mujoco Simulator class for interfacing with the Mujoco simulator
    """

    def __init__(
        self,
        scene_xml_path: str | None = None,
        model: MjModel | None = None,
        camera_hz=config.CameraRates.tenHz,
        cameras_to_use: list[StretchCameras] = []
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
        self._status = self._manager.dict(
            {
                "val": StretchStatus.default().to_dict()
            }
        )
        self._imagery = self._manager.dict(
            {
                "val": StretchCameraStatus.default().to_dict()
            }
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
    def move_to(self, actuator:  config.Actuators, pos: float) -> None:
        """
        Move the actuator to a specific position
        Args:
            actuator_name: str, name of the actuator
            pos: float, absolute position goal
        """
        if not actuator.is_position_actuator:
            click.secho(
                f"Actuator {actuator} not recognized."
                f"\n Available position actuators: {config.Actuators.position_actuators()}",
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
    def move_by(self, actuator: config.Actuators, pos: float) -> None:
        """
        Move the actuator by a specific amount
        Args:
            actuator_name: Actuators, name of the actuator
            pos: float, position to increment by
        """
        if not actuator.is_position_actuator:
            click.secho(
                f"Actuator {actuator} not recognized."
                f"\n Available position actuators: {config.Actuators.position_actuators()}",
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

    def start(self, show_viewer_ui: bool = False, headless: bool = False) -> None:
        """
        Start the simulator

        Args:
            show_viewer_ui: bool, whether to show the Mujoco viewer UI
            headless: bool, whether to run the simulation in headless mode
        """
        self._server_process = Process(
            target=launch_server,
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
                self._cameras_to_use
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
        self._server_process = None
