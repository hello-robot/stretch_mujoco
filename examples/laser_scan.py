import time
import matplotlib.pyplot as plt
import numpy as np

from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


def show_laser_scan(scan_data: np.ndarray):

    lower_bound = 0.2
    upper_bound = 5

    mask_lower = scan_data >= lower_bound
    mask_upper = scan_data <= upper_bound

    filtered_distance = scan_data[mask_lower & mask_upper]

    if len(filtered_distance) == 0:
        return time.sleep(1 / 15)
    
    degrees = np.linspace(0, 359, len(scan_data))
    
    degrees_filtered = degrees[mask_lower & mask_upper]

    x = filtered_distance * np.cos(degrees_filtered) * -1
    y = filtered_distance * np.sin(degrees_filtered) * -1

    plt.scatter(x, y, color="r", s=5)
    max_x = np.abs(x).max()
    max_y = np.abs(y).max()
    plt.xlim([-max_x-1, max_x+1])
    plt.ylim([-max_y-1, max_y+1])

    plt.pause(1 / 15)
    plt.cla()


if __name__ == "__main__":
    cameras_to_use = StretchCameras.none()

    sim = StretchMujocoSimulator(cameras_to_use=cameras_to_use)

    sim.start(headless=False)

    try:
        sim.set_base_velocity(v_linear=5.0, omega=30)

        target = 1.1  # m
        while sim.is_running():
            status = sim.pull_status()
            sensor_data = sim.pull_sensor_data()

            show_laser_scan(scan_data=sensor_data.get_data(StretchSensors.base_lidar))

            current_position = status.base.x

            if target > 0 and current_position > target:
                target *= -1
                sim.set_base_velocity(v_linear=-5.0, omega=-30)
            elif target < 0 and current_position < target:
                target *= -1
                sim.set_base_velocity(v_linear=5.0, omega=30)

    except KeyboardInterrupt:
        sim.stop()
