import random
import threading
import time
import cv2
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


def show_camera_feeds_sync(
    sim: StretchMujocoSimulator, 
    print_fps: bool
):
    """
    Pull camera data from the simulator and display it using OpenCV.

    Call this after calling StretchMujocoSimulator::start().
    """

    camera_data = sim.pull_camera_data()

    if print_fps:
        print(f"Physics fps: {sim.pull_status().fps}. Camera FPS: {camera_data.fps}. {sim.pull_status().sim_to_real_time_ratio_msg}")
    
    for camera_name, pixels in camera_data.get_all(use_depth_color_map=True).items():
        cv2.imshow(camera_name.name, pixels)

    cv2.waitKey(1)


def my_control_loop():
    while sim.is_running():
        sim.move_to(Actuators.lift, random.random())
        time.sleep(3)

if __name__ == "__main__":
    # You can use all the camera's, but it takes longer to render, and may affect the overall simulation FPS.
    # cameras_to_use = StretchCameras.all()
    cameras_to_use = [StretchCameras.cam_d405_rgb]

    sim = StretchMujocoSimulator(cameras_to_use=cameras_to_use)

    sim.start(headless=False)

    threading.Thread(target=my_control_loop, daemon=True).start()

    try:
        while sim.is_running():
            show_camera_feeds_sync(sim, True)

    except KeyboardInterrupt:
        sim.stop()