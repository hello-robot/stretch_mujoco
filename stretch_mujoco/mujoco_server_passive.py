import threading
import time
from stretch_mujoco.utils import override

import click
import mujoco
import mujoco._functions
import mujoco.viewer
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.mujoco_server import MujocoServer
from stretch_mujoco.utils import FpsCounter


class MujocoServerPassive(MujocoServer):
    """
    A MujocoServer flavor that uses the mujoco passive viewer.

    Use `MujocoServerPassive.launch_server()` to start the simulator.

    To render offscreen cameras, please call `set_camera_manager(False,,)` and then `camera_manager.pull_camera_data_at_camera_rate()` in the UI thread.

    On MacOS, this needs to be started with `mjpython`. If you're using StretchMujocoSimulator.start(), this is automatically handled.

    https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
    """

    @override
    def run(
        self,
        show_viewer_ui: bool,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
    ):
        # We're using the passive viewer, and have access to the UI thread. We can manage camera rendering on the UI thread:
        self.set_camera_manager(
            use_camera_thread=False,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        self._run_ui_simulation(show_viewer_ui)

    @override
    def _run_ui_simulation(self, show_viewer_ui: bool):
        """
        Starts the mujoco viewer in Passive Mode. Also starts a physics thread for stepping the simulation.

        On MacOS, this needs to be started with `mjpython`.

        https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
        """

        with mujoco.viewer.launch_passive(
            self.mjmodel, self.mjdata, show_left_ui=show_viewer_ui, show_right_ui=show_viewer_ui
        ) as viewer:

            physics_thread = threading.Thread(
                target=self._physics_loop,
                name="PhysicsThread",
                args=(viewer.lock(), lambda: viewer.is_running() and not self._is_requested_to_stop()),
                daemon=True,
            )
            physics_thread.start()

            fps = FpsCounter()

            UI_FPS_CAP_RATE = (
                self.camera_manager.camera_rate
            )  # 1/Hz.Put the UI thread to sleep so that the physics thread can do work, to mitigate `viewer.lock()` locking physics thread.

            click.secho(
                f"Using the Mujoco Passive Viewer. Note: UI thread and camera rendering is capped to {1/UI_FPS_CAP_RATE}Hz to increase performance. You can set this rate using the `camera_rate` arugment.",
                fg="green",
            )

            # Replace the camera_lock with the viewer lock so that we're not accessing mjdata at the same time as the physics thread.
            self.camera_manager.camera_lock = viewer.lock() #type: ignore

            while viewer.is_running() and not self._is_requested_to_stop():
                fps.tick()
                start_time = time.perf_counter()
                # print(f"UI thread: {fps.fps=}, {self.physics_fps_counter.fps=}, {self.camera_manager.camera_fps_counter.fps=}")

                self.camera_manager.pull_camera_data_at_camera_rate(is_sleep_until_ready=False)

                viewer.sync()

                time_until_next_ui_update = UI_FPS_CAP_RATE - (time.perf_counter() - start_time)
                if time_until_next_ui_update > 0:
                    # Put the UI thread to sleep so that the physics thread can do work, to mitigate `viewer.lock()` taking up ticks.
                    time.sleep(time_until_next_ui_update)
                else:
                    click.secho(
                        f"WARNING: Passive viewer and camera rendering is below the requested {1/self.camera_manager.camera_rate}FPS on the last render.",
                        fg="yellow",
                    )

            self.close()

            # Wait for any active threads to close, otherwise the mujoco window gets stuck:
            active_threads = threading.enumerate()
            for index, thread in enumerate(active_threads):
                if thread != threading.main_thread() and not isinstance(
                    thread, threading._DummyThread
                ):
                    click.secho(
                        f"Stopping thread {index}/{len(active_threads)-1} on the Mujoco Process.",
                        fg="blue",
                    )
                    thread.join(timeout=5.0)

            click.secho("Mujoco viewer has terminated.", fg="blue")
