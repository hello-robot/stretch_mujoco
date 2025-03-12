
import time
import cv2
import numpy as np

from examples.camera_feeds import show_camera_feeds
from stretch_mujoco.cameras import StretchCameras
from stretch_mujoco.config import Actuators
from stretch_mujoco.stretch_mujoco import StretchMujocoSimulator


def draw_circle(n, diameter_m, arm_init, lift_init, sim:StretchMujocoSimulator):
    global cameras_to_use
    """
    From https://forum.hello-robot.com/t/creating-smooth-motion-using-trajectories/671
    """
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    x = (diameter_m / 2) * np.cos(t) + arm_init
    y = (diameter_m / 2) * np.sin(t) + lift_init
    circle_mat = np.c_[x, y]
    for pt in circle_mat:
        print(f"Moiving to {pt}")
        sim.move_to(Actuators.arm, pt[0])
        time.sleep(0.5)
        sim.move_to(Actuators.lift, pt[1])
        time.sleep(0.5)



if __name__ == "__main__":
    cameras_to_use = StretchCameras.all()
    cameras_to_use = [StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth]
    sim = StretchMujocoSimulator(camera_hz=20.0, cameras_to_use=cameras_to_use)
    try:
        sim.start()
        time.sleep(2)

        show_camera_feeds(sim, cameras_to_use, True)

        # sim.move_to(Actuators.wrist_yaw, 1.5707)
        # time.sleep(0.5)
        sim.move_to(Actuators.gripper, 100)
        time.sleep(0.5)
        # input('Press enter to close the gripper')
        # sim.move_to(Actuators.gripper, -100)
        # time.sleep(0.5)
        while True:
            status = sim.pull_status()
            draw_circle(25, 0.2, status.arm.pos, status.lift.pos, sim)
            time.sleep(1)
    except KeyboardInterrupt:
        sim.stop()