import cv2
import time

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras

from examples.keyboard_teleop import show_camera_feeds_sync
from examples.autodocking import *

if __name__ == "__main__":
    # 1. Welcome to the autodocking setup tool
    #    This tool calibrates two models required for autodocking's perception.
    #    Additionally, this tool helps you understand the failure cases of
    #    this routine, and how to best set up the dock to avoid them.
    # 2. Pick a place to put your docking station. There's a few requirements:
    #      - The dock should be placed against a wall
    #          [Bad photo of dock against a corner or in free space]
    #          [Good photo of dock against a wall]
    #      - The back stand of the dock must be easy for a 2D scanner to pick out
    #          [Bad photo of back stand flush with wall to left and right]
    #          [Good photo of back stand easy to pick out from the wall behind]
    #      - The dock should stay in one place. It shouldn't be moved often.
    # 3. Let's start with the calibration.
    #    Put the robot on the dock. Whether by back-driving it by hand or
    #    using the `stretch_keyboard_teleop.py` tool.
    #    Proceed [y/n] - check via charging conditions that robot is on the dock

    sim = StretchMujocoSimulator(cameras_to_use=[StretchCameras.cam_nav_rgb])
    sim.start()

    # put the robot on the dock
    sim.move_by('base_translate', 0.02)
    sim.wait_while_is_moving('base_translate')
    sim.move_by('base_translate', -0.065)
    sim.wait_while_is_moving('base_translate')

    # move the robot forward 1m
    sim.add_world_frame((1, 0, 0))
    sim.move_by('base_translate', 1.0)
    sim.wait_while_is_moving('base_translate', timeout=20.0)

    # turn the robot around
    sim.move_by('base_rotate', 3.1415)
    sim.wait_while_is_moving('base_rotate', timeout=20.0)

    # look down at the dock
    sim.move_to('head_tilt', -0.5)
    sim.wait_until_at_setpoint('head_tilt')
    for _ in range(10):
        show_camera_feeds_sync(sim, False)
    input('is looking at dock?')
    cv2.destroyAllWindows()

    # servo back to goal
    goalx, goaly, goalt = (0.05, 0.0, 0.0)
    sim.add_world_frame((goalx, goaly, 0.0), (0.0, 0.0, goalt))
    for _ in range(5000):
        # get current pose
        b = sim.pull_status().base
        currx, curry, currt = (b.x, b.y, b.theta)
        print(f"Current: ({currx:.3f}, {curry:.3f}, {currt:.3f})")

        # compute relative goal
        errx, erry, errt = inverse_3x3_matrix(rotation_3x3_matrix(currt)) @ np.array([goalx-currx, goaly-curry, wrap(goalt-currt)])
        print(f"Delta: ({errx:.3f}, {erry:.3f}, {errt:.3f})")

        # back out errpose in world frame
        Sb = rotation_3x3_matrix(currt) @ np.array([errx, erry, errt])
        errx_wrt_world = currx + Sb[0]
        erry_wrt_world = curry + Sb[1]
        errt_wrt_world = currt + Sb[2]
        print(f"Delta wrt World: ({errx_wrt_world:.3f}, {erry_wrt_world:.3f}, {errt_wrt_world:.3f})")

        # apply controller
        v, w = polar_controller(errx, erry, errt)
        print(f"Cmd: ({v:.4f}, {w:.4f})")
        sim.set_base_velocity(v, w)
        time.sleep(0.01)

    # wall_start = time.time()
    # sim_start = sim.pull_status().time
    # for _ in range(20):
    #     time.sleep(0.5)
    #     print(f"Sim is running {(sim.pull_status().time - sim_start)/(time.time() - wall_start):.3f}x as fast as realtime")
    #     wall_start = time.time()
    #     sim_start = sim.pull_status().time

    sim.stop()
