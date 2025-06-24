from stretch_mujoco import StretchMujocoSimulator

import time
import math
import numpy as np
from pprint import pprint


if __name__ == "__main__":
    sim = StretchMujocoSimulator()
    sim.start(headless=True)

    print("Start!")
    for _ in range(10):
        time.sleep(1)
        scan = sim.pull_sensor_data().lidar
        print(scan.shape)

    print('Done!')
    sim.stop()

