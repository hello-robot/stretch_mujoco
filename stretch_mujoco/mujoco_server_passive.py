import signal
import threading
import time
from typing import override

import click
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

    def _do_physics(self, viewer):
        """
        This is mujoco physics thread loop that handles stepping the mujoco simulation, and listens to commands from the main process via Proxies.
        """
        while viewer.is_running() and not self.stop_event.is_set():
            start_time = time.perf_counter()

            with viewer.lock():
                mujoco._functions.mj_step(self.mjmodel, self.mjdata)

                self._ctrl_callback(self.mjmodel, self.mjdata)

            time_until_next_step = self.mjmodel.opt.timestep - (
                time.perf_counter() - start_time
            )
            if time_until_next_step > 0:
                # Sleep to match the timestep.
                time.sleep(time_until_next_step)

    @override
    def _run_ui_simulation(self, show_viewer_ui: bool):
        """
        Starts the mujoco viewer in Passive Mode. Also starts a physics thread for stepping the simulation.

        On MacOS, this needs to be started with `mjpython`.

        https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
        """

        signal.signal(signal.SIGTERM, lambda num, h: self.stop_event.set())
        signal.signal(signal.SIGINT, lambda num, h: self.stop_event.set())
        with mujoco.viewer.launch_passive(
            self.mjmodel, self.mjdata, show_left_ui=show_viewer_ui, show_right_ui=show_viewer_ui
        ) as viewer:
            physics_thread = threading.Thread(target=self._do_physics, name="PhysicsThread", args=(viewer,), daemon=True)
            physics_thread.start()

            fps = FpsCounter()
            
            UI_FPS_CAP_RATE = 1/30 #1/Hz.Put the UI thread to sleep so that the physics thread can do work, to mitigate `viewer.lock()`.

            while viewer.is_running() and not self.stop_event.is_set():
                fps.tick()
                start_time = time.perf_counter()
                print(f"UI thread: {fps.fps=}, {self.physics_fps_counter.fps=}, {self.camera_manager.camera_fps_counter.fps=}")

                with viewer.lock():
                    self.camera_manager.pull_camera_data_at_camera_rate()

                viewer.sync()


                time_until_next_ui_update = UI_FPS_CAP_RATE - (
                    time.perf_counter() - start_time
                )
                print(f"{time_until_next_ui_update=}")
                if time_until_next_ui_update > 0:
                    # Put the UI thread to sleep so that the physics thread can do work, to mitigate `viewer.lock()`.
                    time.sleep(time_until_next_ui_update)

            # Wait for any active threads to close, otherwise the mujoco window gets stuck:
            active_threads = threading.enumerate()
            for index, thread in enumerate(active_threads): 
                if thread != threading.main_thread():
                    click.secho(
                        f"Stopping thread {index}/{len(active_threads)-1} on the Mujoco Process.",
                        fg="blue",
                    )
                    thread.join(timeout=5.0)

            click.secho("Mujoco viewer has terminated.", fg="blue")

