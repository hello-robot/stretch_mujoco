import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

try:
    # Some machines seem to need this for matplotlib to work.
    matplotlib.use("TkAgg")
except: ...

def show_laser_scan(scan_data: np.ndarray):

    lower_bound = 0.2
    upper_bound = 5

    mask_lower = scan_data >= lower_bound
    mask_upper = scan_data <= upper_bound

    filtered_distance = scan_data[mask_lower & mask_upper]

    if len(filtered_distance) == 0:
        return time.sleep(1 / 15)
    
    degrees = np.array(range(len(scan_data)))
    degrees = np.radians(degrees)
    
    degrees_filtered = degrees[mask_lower & mask_upper]

    x = filtered_distance * np.cos(degrees_filtered) * -1
    y = filtered_distance * np.sin(degrees_filtered) * -1

    degrees_filtered = np.rad2deg(degrees_filtered)
    front_idx = (degrees_filtered >= 150) & (degrees_filtered <= 210) # ~180
    back_idx = (degrees_filtered >= 330) | (degrees_filtered <= 30) # ~0
    right_idx= (degrees_filtered >= 60) & (degrees_filtered <= 120) # ~90
    left_idx = (degrees_filtered >= 240) & (degrees_filtered <= 300) # ~270

    plt.scatter(x, y, color="r", s=5)
    plt.scatter(x[front_idx], y[front_idx], color="g", s=5)
    plt.scatter(x[back_idx], y[back_idx], color="b", s=5)
    plt.scatter(x[left_idx], y[left_idx], color="k", s=5)
    plt.scatter(x[right_idx], y[right_idx], color="c", s=5)
    max_x = np.abs(x).max()
    max_y = np.abs(y).max()
    plt.xlim([-max_x-1, max_x+1])
    plt.ylim([-max_y-1, max_y+1])
    plt.legend(["All", "Front", "Back", "Left", "Right"])

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

            try:
                show_laser_scan(scan_data=sensor_data.get_data(StretchSensors.base_lidar))
            except: ...

            current_position = status.base.x

            if target > 0 and current_position > target:
                target *= -1
                sim.set_base_velocity(v_linear=-5.0, omega=-30)
            elif target < 0 and current_position < target:
                target *= -1
                sim.set_base_velocity(v_linear=5.0, omega=30)

    except KeyboardInterrupt:
        sim.stop()
