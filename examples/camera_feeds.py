import random
import threading
import time
import cv2
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCamera
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


def show_camera_feeds_sync(
    sim: StretchMujocoSimulator, 
    cameras_to_use: list[StretchCamera],
    print_fps: bool
):
    """
    Pull camera data from the simulator and display it using OpenCV.

    Call this after calling StretchMujocoSimulator::start().
    """

    camera_data = sim.pull_camera_data()

    if print_fps:
        print(f"Physics fps: {sim.pull_status().fps}. Camera FPS: {camera_data.fps}.")

    if not cameras_to_use and not print_fps:
        print(
            "show_camera_feeds: The cameras_to_use array is empty. Did you mean to use StretchCameras.all()?"
        )
        return
    
    for camera in cameras_to_use:
        image = cv2.cvtColor(camera_data.get_camera_data(camera), cv2.COLOR_RGB2BGR)
        cv2.imshow(camera.name, image)

    cv2.waitKey(1)


if __name__ == "__main__":
    def my_control_loop():
        while sim.is_running():
            sim.move_to(Actuators.lift, random.random())
            time.sleep(3)

    # You can use all the camera's, but it takes longer to render, and may affect overall simulation FPS.
    # cameras_to_use = StretchCamera.all()
    cameras_to_use = [StretchCamera.cam_d405_rgb]

    sim = StretchMujocoSimulator(cameras_to_use=cameras_to_use)

    sim.start()

    threading.Thread(target=my_control_loop, daemon=True).start()

    try:
        while sim.is_running():
            show_camera_feeds_sync(sim, cameras_to_use, True)

    except KeyboardInterrupt:
        sim.stop()