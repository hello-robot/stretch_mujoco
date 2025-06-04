from concurrent.futures import ThreadPoolExecutor, as_completed
import platform
import threading
import time
from typing import TYPE_CHECKING
import mujoco
import mujoco._enums
import numpy as np

from stretch_mujoco import config, utils
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch_mujoco.utils import FpsCounter, switch_to_glfw_renderer

if TYPE_CHECKING:
    from stretch_mujoco.mujoco_server import MujocoServer


class MujocoServerCameraManagerSync:
    """
    Handles rendering scene cameras to a buffer.

    Call `pull_camera_data_at_camera_rate()` from the UI thread and the cameras will be rendered at the specified `camera_hz`.
    """

    def __init__(
        self, camera_hz: float, cameras_to_use: list[StretchCameras], mujoco_server: "MujocoServer"
    ) -> None:

        self.mujoco_server = mujoco_server

        self.camera_rate = 1 / camera_hz  # Hz to seconds

        self.camera_renderers: dict[StretchCameras, mujoco.Renderer] = {}

        self._set_camera_properties_and_create_renderers_in_mujoco(cameras_to_use)

        self.camera_fps_counter = FpsCounter()

        self.time_start = time.perf_counter()

        self.camera_lock = threading.Lock()

    def close(self):
        """
        Clean up renderer resources
        """
        for renderer in self.camera_renderers.values():
            renderer.close()

    def is_ready_to_pull_camera_data(self, is_sleep_until_ready: bool = False):
        """
        Checks to see if a duration of time has passed since the last call
        to this function to render camera at the specified `self.camera_rate`.
        """
        elapsed = time.perf_counter() - self.time_start
        if elapsed < self.camera_rate:
            # If we're not ready to render camera, don't render:
            if not is_sleep_until_ready:
                return False
            # sleep until ready:
            time.sleep(self.camera_rate - elapsed)

        self.time_start = time.perf_counter()
        return True

    def pull_camera_data_at_camera_rate(self, is_sleep_until_ready: bool):
        """
        Call this on the UI thread to render camera data.
        """

        if not self.is_ready_to_pull_camera_data(is_sleep_until_ready):
            return

        self._pull_camera_data()

        self.camera_fps_counter.tick()

    def _pull_camera_data(self):
        """
        Render a scene at each camera using the simulator and populate the imagery dictionary with the raw image pixels and camera params.
        """
        new_imagery = StatusStretchCameras.default()
        new_imagery.time = self.mujoco_server.mjdata.time
        new_imagery.fps = self.camera_fps_counter.fps

        for camera, renderer in self.camera_renderers.items():
            (_, data) = self._render_camera(renderer, camera)
            new_imagery.set_camera_data(camera, data)

        new_imagery.cam_d405_K = self.get_camera_params(StretchCameras.cam_d405_rgb)
        new_imagery.cam_d435i_K = self.get_camera_params(StretchCameras.cam_d435i_rgb)

        self.mujoco_server.data_proxies.set_cameras(new_imagery)

    def _create_camera_renderer(self, for_camera: StretchCameras):
        settings = for_camera.initial_camera_settings

        # Update mujoco's offscreen gl buffer size to accomodate bigger resolutions:
        offscreen_buffer_width = self.mujoco_server.mjmodel.vis.global_.offwidth
        offscreen_buffer_height = self.mujoco_server.mjmodel.vis.global_.offheight

        if settings.width > offscreen_buffer_width:
            self.mujoco_server.mjmodel.vis.global_.offwidth = settings.width
        if settings.height > offscreen_buffer_height:
            self.mujoco_server.mjmodel.vis.global_.offheight = settings.height

        renderer = mujoco.Renderer(
            self.mujoco_server.mjmodel, width=settings.width, height=settings.height
        )

        renderer._scene_option.flags[mujoco._enums.mjtVisFlag.mjVIS_RANGEFINDER] = False # Disables the lidar yellow lines.

        from stretch_mujoco.mujoco_server_passive import MujocoServerPassive

        if platform.system() == "Darwin" and not isinstance(
            self.mujoco_server, MujocoServerPassive
        ):
            # On MacOS, switch to glfw because CGL is not compatible with offscreen rendering on the managed viewer (because of mutex locking).
            switch_to_glfw_renderer(self.mujoco_server.mjmodel, renderer)

        if for_camera.is_depth:
            renderer.enable_depth_rendering()

        return renderer

    def _render_camera(self, renderer: mujoco.Renderer, camera: StretchCameras):
        """
        This calls update_scene and render() for an offscreen camera buffer.

        Use this with the _toggle_camera() functionality in this class.
        """

        with self.camera_lock:
            renderer.update_scene(data=self.mujoco_server.mjdata, camera=camera.camera_name_in_mjcf)

            render = renderer.render()

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

        self.camera_renderers[camera] = self._create_camera_renderer(for_camera=camera)

    def get_camera_params(self, camera: StretchCameras) -> np.ndarray:
        """
        Get camera parameters
        """
        cam = self.mujoco_server.mjmodel.camera(camera.camera_name_in_mjcf)
        d = {
            "f": self.mujoco_server.mjmodel.cam_intrinsic[cam.id][:2],
            "p": self.mujoco_server.mjmodel.cam_intrinsic[cam.id][2:],
            "res": self.mujoco_server.mjmodel.cam_resolution[cam.id],
        }
        camera_k = utils.compute_K(
            camera.initial_camera_settings.field_of_view_vertical_in_degrees,
            d["res"][0],
            d["res"][1],
        )
        return camera_k

    def set_camera_params(self, camera: StretchCameras) -> None:
        """
        Set camera parameters
        Args:
            camera_name: str, name of the camera
            fovy: float, vertical field of view in degrees
            res: tuple, size of the camera Image
        """
        cam = self.mujoco_server.mjmodel.camera(camera.camera_name_in_mjcf)

        settings = camera.initial_camera_settings

        self.mujoco_server.mjmodel.cam_fovy[cam.id] = settings.field_of_view_vertical_in_degrees
        self.mujoco_server.mjmodel.cam_intrinsic[cam.id] = list(settings.focal) + [
            0,
            0,
        ]  # a Mujoco takes: [fx, fy, px, py]
        self.mujoco_server.mjmodel.cam_resolution[cam.id] = (
            settings.sensor_resolution
            if settings.sensor_resolution is not None
            else (settings.width, settings.height)
        )
        if settings.sensor_size is not None:
            self.mujoco_server.mjmodel.cam_sensorsize[cam.id] = settings.sensor_size

        print(
            f"""
Initializing camera {camera.name}:
{settings=}
"""
        )

    def _set_camera_properties_and_create_renderers_in_mujoco(
        self, cameras_to_use: list[StretchCameras]
    ):
        """
        Set the camera properties and create a camera renderer for each camera in use.
        """
        for camera in cameras_to_use:
            self.set_camera_params(
                camera,
            )

            self._add_camera_renderer(camera)


class MujocoServerCameraManagerThreaded(MujocoServerCameraManagerSync):
    """
    Starts a camera loop on init to pull camera data using threading.
    """

    def __init__(
        self,
        use_camera_thread: bool,
        use_threadpool_executor: bool,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
        mujoco_server: "MujocoServer",
    ):
        """
        `use_threadpool_executor` will use a ThreadPoolExecutor to render all cameras. Setting to false will render each one synchronously.

        `use_camera_thread` can be set to false to use the ThreadPoolExecutor without the camera thread. `pull_camera_data_at_camera_rate()` must be called on the UI thread if this mode is used.
        """

        super().__init__(camera_hz, cameras_to_use, mujoco_server)

        self.use_camera_thread = use_camera_thread
        self.use_threadpool_executor = use_threadpool_executor

        if not use_camera_thread and not use_threadpool_executor:
            raise Exception("use_camera_thread and use_threadpool_executor cannot be both falsey")

        if use_threadpool_executor:
            if platform.system() == "Darwin":
                self.cameras_rendering_thread_pool = ThreadPoolExecutor(
                    max_workers=len(cameras_to_use) if cameras_to_use else 1
                )
            else:
                # Linux is currently struggling with multi-threaded camera rendering:
                self.cameras_rendering_thread_pool = ThreadPoolExecutor(max_workers=1)

        if use_camera_thread:
            self.cameras_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.cameras_thread.start()

    def pull_camera_data_at_camera_rate(self, is_sleep_until_ready: bool):
        """
        Uses the threadpool to pull camera data.

        Call this on the UI thread to render camera data.
        """
        if self.use_camera_thread:
            raise Exception(
                "This call is not allowed when This update is managed in the _camera_loop."
            )

        if not self.is_ready_to_pull_camera_data(is_sleep_until_ready):
            return

        self._pull_camera_data_threadpool()

        self.camera_fps_counter.tick()

    def _camera_loop(self):
        """
        This is the thread loop that handles camera rendering.
        """

        while (
            self.mujoco_server.data_proxies.get_status().time == 0
        ) and not self.mujoco_server._is_requested_to_stop():
            # wait for sim to start
            time.sleep(0.1)

        while not self.mujoco_server._is_requested_to_stop():

            if not self.is_ready_to_pull_camera_data(is_sleep_until_ready=True):
                return

            if self.use_threadpool_executor:
                self._pull_camera_data_threadpool()
            else:
                self._pull_camera_data()

            self.camera_fps_counter.tick()

    def _pull_camera_data_threadpool(self):
        """
        Uses a ThreadPoolExecutor to render a scene at each camera using the simulator and populate the imagery dictionary with the raw image pixels and camera params.
        """
        new_imagery = StatusStretchCameras.default()
        new_imagery.time = self.mujoco_server.mjdata.time
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

        new_imagery.cam_d405_K = self.get_camera_params(StretchCameras.cam_d405_rgb)
        new_imagery.cam_d435i_K = self.get_camera_params(StretchCameras.cam_d435i_rgb)

        self.mujoco_server.data_proxies.set_cameras(new_imagery)
