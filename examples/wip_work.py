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
SEEK = 25

# Program
prev = time.time()
dock_angle = math.pi
prev_heading = 0.0
servo_done = False

# Networking
ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.setsockopt(zmq.SNDHWM, 1)
sock.setsockopt(zmq.RCVHWM, 1)
sock.connect(f"tcp://127.0.0.1:8080")

# Controller parameters
k_rho: float   = 1.2      # >0  (distance gain)
k_alpha: float = 2.5      # >k_rho (bearing gain)
k_theta: float = -1.6     # <0  (heading gain)
v_max: float = 0.6        # m/s
w_max: float = 2.5        # rad/s
rho_tol: float = 0.01     # m (10mm)
theta_tol: float = 0.02   # rad (1.14deg)

def wrap(angle):
    """Wrap to (-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def compute_cmd(x_g: float,
                y_g: float,
                theta_g: float,
                force_backwards: bool = False):
    """
    Returns (v, w, arrived).
    arrived=True when both distance ≤ rho_tol and heading error ≤ theta_tol.

    • If force_backwards == True the robot always tries to drive in reverse.
    • Otherwise it chooses the direction that minimises how much it must turn:
        * drive forward when |α| ≤ π/2
        * drive backward when |α| > π/2
    """
    # --- 1. basic geometry in the robot frame --------------------------------
    rho        = math.hypot(x_g, y_g)               # distance to goal  (≥ 0)
    alpha      = math.atan2(y_g, x_g)               # bearing to goal   (−π, π]
    theta_err  = wrap(theta_g)                      # wanted final yaw

    # --- 2. choose a driving direction ---------------------------------------
    drive_dir = -1 if force_backwards else ( -1 if abs(alpha) > math.pi/2 else 1 )

    #  When driving backward we redefine α so that it is measured w.r.t
    #  the *rear* of the robot.  This keeps α in (−π/2, π/2) and prevents the
    #  controller from trying to spin the base around.
    if drive_dir == -1:
        alpha = wrap(alpha - math.copysign(math.pi, alpha))

    # --- 3. control law -------------------------------------------------------
    v = drive_dir * k_rho * rho                      # ← may be negative
    w = k_alpha * alpha + k_theta * theta_err  # unchanged sign law

    # --- 4. saturation --------------------------------------------------------
    v = max(min(v,  v_max), -v_max)
    w = max(min(w,  w_max), -w_max)

    # --- 5. stopping logic ----------------------------------------------------
    arrived = False
    if rho < rho_tol:          # close enough in position
        v = 0.0
        if abs(theta_err) < theta_tol:
            w = 0.0
            arrived = True

    return v, w, arrived


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
    global prev, dock_angle, prev_heading, servo_done

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
    normal = np.array([wall_direction[1], wall_direction[0]])
    normal /= np.linalg.norm(normal)
    target_t = math.atan2(normal[1], normal[0])
    dock_centroid[1] = -1*dock_centroid[1]
    target_x, target_y = (dock_centroid/1000) + 0.625 * normal

    # servo!
    v, w, arrived = compute_cmd(target_x, target_y, target_t)
    print(f"Cmd: {np.array([v, w])}, Curr: {np.array([sim.pull_status().base.x, sim.pull_status().base.y, sim.pull_status().base.theta])}")
    if arrived and not servo_done:
        servo_done = True
        print("servo done")
        sim.move_by('base_translate', 0.0)
    if not servo_done:
        sim.set_base_velocity(v, w)


if __name__ == "__main__":
    sim = StretchMujocoSimulator()
    sim.start()
    print("Start!")

    while True:
        update(sim)
        time.sleep(0.1)

    print('Done!')
    sim.stop()

