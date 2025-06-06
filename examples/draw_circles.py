
import threading
import time
import numpy as np

from examples.camera_feeds import show_camera_feeds_sync
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


def draw_circle(n, diameter_m, arm_init, lift_init, sim:StretchMujocoSimulator):
    """
    From https://forum.hello-robot.com/t/creating-smooth-motion-using-trajectories/671
    """
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    x = (diameter_m / 2) * np.cos(t) + arm_init
    y = (diameter_m / 2) * np.sin(t) + lift_init
    circle_mat = np.c_[x, y]
    for pt in circle_mat:
        print(f"Moving to {pt}")
        sim.move_to(Actuators.arm, pt[0])
        sim.move_to(Actuators.lift, pt[1])

        sim.wait_until_at_setpoint(Actuators.arm)
        sim.wait_until_at_setpoint(Actuators.lift)

def _run_draw_circle():
    time.sleep(2)
    try:
        while sim.is_running(): 
            sim.move_to(Actuators.head_tilt, -1.5707)
            sim.move_to(Actuators.head_pan, -0.7853)

            sim.move_to(Actuators.wrist_yaw, 1.5707)
            sim.move_to(Actuators.gripper, 0.5)
            sim.wait_until_at_setpoint(Actuators.wrist_yaw)
            sim.wait_until_at_setpoint(Actuators.gripper)

            sim.move_to(Actuators.gripper, pos=-0.15)
            sim.wait_until_at_setpoint(Actuators.gripper)
            
            status = sim.pull_status()
            draw_circle(25, 0.2, status.arm.pos, status.lift.pos, sim)
            time.sleep(1)
            sim.home()
            time.sleep(2)
    except ConnectionError: ...


if __name__ == "__main__":

    # You can use all the camera's, but it takes longer to render, and may affect the overall simulation FPS.
    # cameras_to_use = StretchCameras.all()
    cameras_to_use = [StretchCameras.cam_d405_rgb, StretchCameras.cam_d435i_rgb]

    sim = StretchMujocoSimulator(cameras_to_use=cameras_to_use)

    sim.start(headless=False)

    threading.Thread(target=_run_draw_circle, daemon=False).start()

    try:
        while sim.is_running():
            show_camera_feeds_sync(sim, True)

    except KeyboardInterrupt:
        sim.stop()