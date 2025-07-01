from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.utils import diff_drive_inv_kinematics

import zmq
import time
import math
import numpy as np
np.set_printoptions(precision=3, linewidth=100, suppress=True)
from scipy.optimize import minimize
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Constants
DMAX = 4000
SEEK = 12
MAX_WHEEL_SPEED = 6.0
ARC_COST_THRES = 1e-7
TURN_MAX_SPEED = 0.15

# Program
prev = time.time()
dock_angle = math.pi
prev_heading = 0.0
needs_turning = None
turn_offset = 0.0

# Networking
ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.setsockopt(zmq.SNDHWM, 1)
sock.setsockopt(zmq.RCVHWM, 1)
sock.connect(f"tcp://127.0.0.1:8080")


def follow_arc(R, dtheta):
    vs = [-0.3, -0.2, -0.1, -0.05, -0.01]
    for v in vs:
        w = v / abs(R) * np.sign(dtheta)
        w_left, w_right = diff_drive_inv_kinematics(v, w)
        if abs(w_left) < MAX_WHEEL_SPEED and abs(w_right) < MAX_WHEEL_SPEED:
            break
    return v, w


def convert_to_0to2pi(theta):
    """Converts [-pi, pi] to [0, 2pi]"""
    return theta if theta >= 0 else theta + 2 * math.pi


def arc_endpoint(R, delta_theta):
    x = R * np.sin(delta_theta)
    y = R * (1 - np.cos(delta_theta))
    return x, y, delta_theta


def cost(params, x_target, y_target, theta_target):
    R, delta_theta = params
    if np.abs(R) < 1e-4:  # avoid near-zero radius
        return 1e6
    x, y, heading = arc_endpoint(R, delta_theta)
    dx = x - x_target
    dy = y - y_target
    dtheta = ((heading - theta_target + np.pi) % (2*np.pi)) - np.pi
    return dx**2 + dy**2 + (dtheta**2)


def plan_arc(target_x, target_y, target_t):
    r_bounds = (0.01, None) if target_t < 0 else (None, -0.01)
    res = minimize(
        cost,
        x0=np.array([0.0, 0.0]), # (radius, dTheta)
        args=(target_x, target_y, target_t),
        bounds=[r_bounds, (-2*np.pi, 2*np.pi)]
    )
    if not res.success:
        return
    R_opt, dTheta_opt = res.x
    return (R_opt, dTheta_opt, res.fun)

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
    global prev, dock_angle, prev_heading, needs_turning, turn_offset

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
        print('Failed to find fits')
        return

    # Compute target
    normal = np.array([wall_direction[1], -wall_direction[0]])
    normal /= np.linalg.norm(normal)
    target_t = math.atan2(normal[1], normal[0])
    target_x, target_y = (dock_centroid/1000) + 0.625 * normal

    # Plot line
    t_vals = np.linspace(-DMAX, DMAX, 100)
    line_points = dock_centroid[None, :] + t_vals[:, None] * wall_direction[None, :]
    more_line_points = np.array([np.array([target_x, target_y]) - np.random.uniform(0.0, 0.625) * normal for i in range(100)]) * 1000
    line_points = np.vstack([line_points, more_line_points])

    # Compute arc path
    ret = plan_arc(target_x, target_y, target_t)
    if ret is None:
        print("No arc path found")
        return
    R_opt, dTheta_opt, arc_cost = ret
    is_optimal = arc_cost < ARC_COST_THRES
    if not is_optimal and needs_turning is None:
        needs_turning = True
        angle_to_target = math.atan2(target_y, target_x)
        turn_offset = np.sign(angle_to_target)
    if needs_turning and is_optimal:
        needs_turning = False
        sim.set_base_velocity(0.0, 0.0)
        time.sleep(0.1)

    if needs_turning:
        path = np.zeros_like(line_points)

        # Turn to feasible arc
        angle_to_target = math.atan2(target_y, target_x)
        k = 0.3
        w = k * ((math.pi + turn_offset) - convert_to_0to2pi(angle_to_target))
        w = np.clip(w, -TURN_MAX_SPEED, TURN_MAX_SPEED)
        print(f"Turn: {w:.4f} Turn target: {math.pi + turn_offset:.4f} Turn current: {convert_to_0to2pi(angle_to_target):.4f} Arc Cost: {arc_cost:.10f}")
        sim.set_base_velocity(0.0, w)
    else:
        # arc_len = R_opt * abs(dTheta_opt)
        # print(f"Found arc: Radius = {R_opt:.3f}, Angle = {dTheta_opt:.3f} rad, Arc length = {arc_len:.3f}")

        # Viz path
        angles = np.linspace(0, dTheta_opt, 100)
        x_arc = R_opt * np.sin(angles)
        y_arc = R_opt * (1 - np.cos(angles))
        path = np.stack([x_arc, y_arc], axis=1)
        path *= 1000

        # Execute path
        v, w = follow_arc(R_opt, dTheta_opt)
        print(f"{v=:.1f} {w=:.3f} {sim.pull_status().base.theta_vel=:.3f}")
        sim.set_base_velocity(v, w)

    sock.send_pyobj({
        'polar_offsets': polar_offsets,
        'cartesian_offsets': cartesian_offsets,
        'cut_both_sides': dock_pts,
        'both_sides': both_sides,
        'line_points': line_points,
        'path': path,
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

