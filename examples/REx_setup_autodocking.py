import time

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras


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
    input("Is the robot on the dock? [y/n]")

    # move the robot forward 1m
    sim.move_by('base_translate', 1.0)
    sim.wait_while_is_moving('base_translate', timeout=20.0)

    wall_start = time.time()
    sim_start = sim.pull_status().time
    for _ in range(20):
        time.sleep(0.5)
        print(f"Sim is running {(sim.pull_status().time - sim_start)/(time.time() - wall_start):.3f}x as fast as realtime")
        wall_start = time.time()
        sim_start = sim.pull_status().time

    sim.stop()
