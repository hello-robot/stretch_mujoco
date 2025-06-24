from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.stretch_sensors import StretchSensors

import zmq
import time
import math
import numpy as np
np.set_printoptions(precision=3, linewidth=100, suppress=True)
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Constants
DMAX = 4000
SEEK = 12

# Program
prev = time.time()
dock_angle = math.pi
prev_heading = 0.0

# Networking
ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.setsockopt(zmq.SNDHWM, 1)
sock.setsockopt(zmq.RCVHWM, 1)
sock.connect(f"tcp://127.0.0.1:8080")


def rotation_3x3_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta),  0],
                     [0            , 0,              1]])


def to_cartesian(arr):
    x = arr[:,1] * np.cos(arr[:,0])
    y = arr[:,1] * np.sin(arr[:,0])
    xy = np.stack((x, y), axis=1)
    return xy


def wrap_slice(arr, start, stop):
    n = len(arr)
    start %= n
    stop %= n
    if start < stop:
        return arr[start:stop]
    else:
        return np.concatenate((arr[start:], arr[:stop]), axis=0)


def fit_line(points):
    """Returns (centroid, direction, RMS fit error)"""
    if len(points) < 2:
        return None, None, np.inf
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    direction = vh[0]  # First principal component is the line direction
    # Line is: (x, y) = centroid + t * direction

    # Project points onto the direction/slope
    proj_lengths = np.dot(centered_points, direction)  # shape (N,)
    projections = np.outer(proj_lengths, direction)  # shape (N, 2)
    # Orthogonal vectors = points - projections
    orthogonal_vecs = centered_points - projections
    # Distances from points to the line (Euclidean norms)
    distances = np.linalg.norm(orthogonal_vecs, axis=1)  # shape (N,)
    # RMS error
    rms_error = np.sqrt(np.mean(distances**2))

    return (centroid, direction, rms_error)


def prepare_scan(sim):
    sensor_data = sim.pull_sensor_data()
    scan_data = sensor_data.get_data(StretchSensors.base_lidar)

    lower_bound = 0.05
    upper_bound = 5

    mask_lower = scan_data >= lower_bound
    mask_upper = scan_data <= upper_bound

    filtered_distance = scan_data[mask_lower & mask_upper] * 1000
    if len(filtered_distance) == 0:
        return np.empty((0, 2))

    degrees = np.array(range(len(scan_data)))
    degrees = (-1*degrees + 180) % (360)
    angles = np.radians(degrees)
    angles_filtered = angles[mask_lower & mask_upper]

    scan = np.stack([angles_filtered, filtered_distance], axis=1)
    return scan


def update(sim):
    global prev, dock_angle, prev_heading

    # Heading update
    curr_heading = sim.pull_status().base.theta
    delta_heading = curr_heading - prev_heading
    delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi
    if np.abs(delta_heading) > 1e-3:
        dock_angle += delta_heading
    dock_angle = dock_angle % (2 * math.pi)
    prev_heading = curr_heading

    # Get Scan
    scan = prepare_scan(sim)

    # Log rate
    now = time.time()
    delta = now - prev
    # print(f'Rate: {1/delta:.2f} Hz, Scan size: {scan.shape}')
    prev = now

    # Polar & cartesian update
    polar_offsets = scan
    cartesian_offsets = to_cartesian(polar_offsets)

    # Compute seek
    idx = (np.abs(scan[:,0] - dock_angle)).argmin()
    seek_right = wrap_slice(scan, idx-SEEK, idx+1)
    seek_left = wrap_slice(scan, idx, idx+SEEK+1)
    both_sides = np.vstack([seek_right, seek_left])

    # Segment dock from wall
    xy = to_cartesian(both_sides)
    # 1. Choose x→y or y→x depending on wall orientation
    if np.var(xy[:, 0]) >= np.var(xy[:, 1]):       # not near-vertical
        X, y = xy[:, 0].reshape(-1, 1), xy[:, 1]
    else:                                          # near-vertical line
        X, y = xy[:, 1].reshape(-1, 1), xy[:, 0]

    # 2. Fit a robust line with RANSAC
    ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=2,
                residual_threshold=10, # ≈ 1–3× range noise
                max_trials=250,
                random_state=42)
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    wall_pts = both_sides[inlier_mask]
    dock_pts = both_sides[~inlier_mask]
    wall_centroid, wall_direction, _ = fit_line(to_cartesian(wall_pts))
    dock_centroid, dock_direction, _ = fit_line(to_cartesian(dock_pts))
    if dock_centroid is None or wall_direction is None:
        return

    # Plot line
    t_vals = np.linspace(-DMAX, DMAX, 100)
    line_points = dock_centroid[None, :] + t_vals[:, None] * wall_direction[None, :]

    sock.send_pyobj({
        'polar_offsets': polar_offsets,
        'cartesian_offsets': cartesian_offsets,
        'cut_both_sides': dock_pts,
        'both_sides': both_sides,
        'line_points': line_points,
    })

    # Compute servoing target
    normal = np.array([-dock_direction[1], dock_direction[0]])
    normal /= np.linalg.norm(normal)
    normal = -normal
    target_t = math.atan2(-normal[1], -normal[0])
    target_xy = (dock_centroid/1000) + 0.3 * normal
    print("target", target_xy, target_t)

    # Back out target into world frame
    b = sim.pull_status().base
    currx, curry, currt = (b.x, b.y, b.theta)
    errx, erry, errt = (target_xy[0], target_xy[1], target_t)
    Sb = rotation_3x3_matrix(currt) @ np.array([errx, erry, errt])
    errx_wrt_world = currx + Sb[0]
    erry_wrt_world = curry + Sb[1]
    errt_wrt_world = currt + Sb[2]
    sim.add_world_frame((errx_wrt_world, erry_wrt_world, 0.0), (0.0, 0.0, errt_wrt_world))

    sim.set_base_velocity(0.0, 0.4)


if __name__ == "__main__":
    sim = StretchMujocoSimulator()
    sim.start()
    print("Start!")

    while True:
        update(sim)
        time.sleep(0.1)

    print('Done!')
    sim.stop()

