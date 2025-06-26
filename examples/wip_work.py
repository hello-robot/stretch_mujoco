from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.stretch_sensors import StretchSensors

import zmq
import time
import math
import numpy as np
np.set_printoptions(precision=3, linewidth=100, suppress=True)
from scipy.special import comb
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Constants
DMAX = 4000
SEEK = 25

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


def bezier_SE2(p0, th0, p3, th3, frac=0.8):
    """
    Cubic Bézier through two SE(2) poses that never goes past either pose.
    - `frac` (0-1) tells how far *toward* the intersection we put the handle.
       0.8 is a good all-rounder; 1.0 puts it exactly at the intersection.
    """
    v0  = np.array([np.cos(th0), np.sin(th0)])
    v3  = np.array([np.cos(th3), np.sin(th3)])          # fwd direction at goal
    d   = p3 - p0

    # Solve p0 + a·v0  ==  p3 - b·v3   (a,b ≥ 0 give intersection of the two rays)
    A = np.column_stack((v0, v3))                       # [v0  v3]
    try:
        a, b = np.linalg.lstsq(A, d, rcond=None)[0]
    except np.linalg.LinAlgError:                       # nearly parallel headings
        a = b = np.inf

    # Limit handles:  ‖p1-p0‖ ≤ a,  ‖p3-p2‖ ≤ b
    chord = np.linalg.norm(d)
    L1 = frac * min(max(a, 0), chord/3)                 # fall back to chord/3
    L2 = frac * min(max(b, 0), chord/3)

    p1 = p0 + L1 * v0
    p2 = p3 - L2 * v3

    def curve(t):
        B = np.vstack([(comb(3,i)*(t**i)*(1-t)**(3-i)) for i in range(4)]).T
        return B @ np.vstack([p0, p1, p2, p3])

    return curve


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
    if np.dot(direction, np.array([0, 1])) < 0: # keep +y “up”
        direction = -direction
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

    # Compute target
    normal = np.array([wall_direction[1], wall_direction[0]])
    normal /= np.linalg.norm(normal)
    target_t = math.atan2(normal[1], normal[0])
    dock_centroid[1] = -1*dock_centroid[1]
    target_x, target_y = (dock_centroid/1000) + 0.625 * normal

    # Compute bezier path
    curve = bezier_SE2(np.array([0.0, 0.0]), math.pi, np.array([target_x, target_y]), target_t)
    path = curve(np.linspace(0, 1, 20))

    # Viz in matplotlib
    path_viz = []
    for p in path:
        b = sim.pull_status().base
        Sb = rotation_3x3_matrix(b.theta) @ np.array((p[0], p[1], 0.0))
        path_viz.append([Sb[0], Sb[1]])
    path_viz = np.array(path_viz) * 1000

    sock.send_pyobj({
        'polar_offsets': polar_offsets,
        'cartesian_offsets': cartesian_offsets,
        'cut_both_sides': dock_pts,
        'both_sides': both_sides,
        'line_points': line_points,
        'path': path_viz,
    })


if __name__ == "__main__":
    sim = StretchMujocoSimulator()
    sim.start()
    print("Start!")

    while True:
        update(sim)
        time.sleep(0.1)

    print('Done!')
    sim.stop()

