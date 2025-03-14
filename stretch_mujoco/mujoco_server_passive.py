import threading
import time
from typing import override

import mujoco
import mujoco._functions
import mujoco.viewer
from stretch_mujoco.enums.cameras import StretchCameras
from stretch_mujoco.mujoco_server import MujocoServer
from stretch_mujoco.utils import FpsCounter


class MujocoServerPassive(MujocoServer):
    """
    A MujocoServer flavor that uses the mujoco passive viewer.

    Use `MujocoServerPassive.launch_server()` to start the simulator.

    To render offscreen cameras, please call `set_camera_manager(False,,)` and then `camera_manager.pull_camera_data_at_camera_rate()` in the UI thread.

    On MacOS, this needs to be started with `mjpython`.

    https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
    """

    def _do_physics(self, viewer):
        fps = FpsCounter()
        while viewer.is_running() and not self.stop_event.is_set():
            fps.tick()

            with viewer.lock():
                mujoco._functions.mj_step(self.mjmodel, self.mjdata)

                self._ctrl_callback(self.mjmodel, self.mjdata)

    @override
    def run(
        self,
        show_viewer_ui: bool,
        headless: bool,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
    ):
        # We're using the passive viewer, and have access to the UI thread. We can manage camera rendering on the UI thread:
        self.set_camera_manager(
            use_camera_thread=False, camera_hz=camera_hz, cameras_to_use=cameras_to_use
        )

        if headless:
            self._run_headless_simulation()
        else:
            self._run_ui_simulation(show_viewer_ui)

    @override
    def _run_ui_simulation(self, show_viewer_ui: bool):
        with mujoco.viewer.launch_passive(
            self.mjmodel, self.mjdata, show_left_ui=show_viewer_ui, show_right_ui=show_viewer_ui
        ) as viewer:
            physics_thread = threading.Thread(target=self._do_physics, args=(viewer,))
            physics_thread.start()

            fps = FpsCounter()

            while viewer.is_running() and not self.stop_event.is_set():
                fps.tick()
                # print(f"UI thread: {fps.fps=}, {self.simulation_fps_counter.fps=}, {self.camera_manager.camera_fps_counter.fps=}")

                self.camera_manager.pull_camera_data_at_camera_rate()

                time_until_next_step = self.mjmodel.opt.timestep - (
                    time.time() - fps.fps_start_time
                )
                if time_until_next_step > 0:
                    # Put the UI thread to sleep so that the physics thread can do work.
                    time.sleep(time_until_next_step)

                viewer.sync()

            physics_thread.join()
