import platform
import threading
import cv2
from stretch_mujoco.enums.cameras import StretchCameras
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


def show_camera_feeds_sync(
    sim: StretchMujocoSimulator, cameras_to_use: list[StretchCameras], print_fps: bool
):
    """
    Pull camera data from the simulator and display it using OpenCV.

    Call this after calling StretchMujocoSimulator::start().

    There will be a ~1ms delay due to cv2.waitKey(1)
    """
    camera_data = sim.pull_camera_data()

    if print_fps:
        print(f"Simulation fps: {sim.pull_status().fps}. Camera FPS: {camera_data.fps}.")

    for camera in cameras_to_use:
        image = cv2.cvtColor(camera_data.get_camera_data(camera), cv2.COLOR_RGB2BGR)
        cv2.imshow(camera.name, image)

    cv2.waitKey(1)


def _show_camera_feeds_loop(
    sim: StretchMujocoSimulator, cameras_to_use: list[StretchCameras], print_fps: bool
):
    if not cameras_to_use:
        print(
            "show_camera_feeds: The cameras_to_use array is empty. Did you mean to use StretchCameras.all()?"
        )
        return

    while sim._running:

        show_camera_feeds_sync(sim, cameras_to_use, print_fps)


def show_camera_feeds_async(
    sim: StretchMujocoSimulator, cameras_to_use: list[StretchCameras], print_fps: bool
):
    """
    Starts a thread to display camera feeds. Call this after calling StretchMujocoSimulator::start().
    """
    if platform.system() == "Darwin":
        print(
            "show_camera_feeds_async() does not work on MacOS because cv2.imshow() does not work from a thread. Use show_camera_feeds() instead."
        )
        return

    threading.Thread(target=_show_camera_feeds_loop, args=(sim, cameras_to_use, print_fps), daemon=True).start()
