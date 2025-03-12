import threading
import cv2
from stretch_mujoco.cameras import StretchCameras
from stretch_mujoco.stretch_mujoco import StretchMujocoSimulator


def _show_camera_feeds(sim:StretchMujocoSimulator, cameras_to_use: list[StretchCameras]):
    while sim._running:
        camera_data = sim.pull_camera_data()
        for camera in cameras_to_use:
            cv2.imshow(camera.name, camera_data.get_camera_data(camera))
        print(f"got camera {sim.pull_status().time}")
        cv2.waitKey(5)

def show_camera_feeds(sim:StretchMujocoSimulator, cameras_to_use: list[StretchCameras]):
    threading.Thread(target=_show_camera_feeds, args=(sim, cameras_to_use)).start()