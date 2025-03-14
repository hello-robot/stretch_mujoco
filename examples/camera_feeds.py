import platform
import threading
import cv2
from stretch_mujoco.enums.cameras import StretchCameras
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

def _show_camera_feeds(sim:StretchMujocoSimulator, cameras_to_use: list[StretchCameras], print_fps: bool):
    if not cameras_to_use:
        print("show_camera_feeds: The cameras_to_use array is empty. Did you mean to use StretchCameras.all()?")
        return
    
    while sim._running:

        camera_data = sim.pull_camera_data()

        if print_fps:
            print(f"Simulation fps: {sim.pull_status().fps}. Camera FPS: {camera_data.fps}.")

        for camera in cameras_to_use:
            cv2.imshow(camera.name, camera_data.get_camera_data(camera))

        cv2.waitKey(1)

def show_camera_feeds(sim:StretchMujocoSimulator, cameras_to_use: list[StretchCameras], print_fps: bool):
    """
    Starts a thread to display camera feeds. Call this after calling StretchMujocoSimulator::start().

    There will be a ~1ms delay due to cv2.waitKey(1)
    """
    if platform.system() == "Darwin":
        print("show_camera_feeds() does not work on MacOS because cv2.imshow() does not work from a thread.")
        return

    threading.Thread(target=_show_camera_feeds, args=(sim, cameras_to_use, print_fps)).start()