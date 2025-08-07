from stretch_mujoco.mujoco_server import MujocoServer
from stretch_mujoco.mujoco_server import MujocoServerProxies

import cv2
import math
import mujoco
import signal
import threading
import numpy as np
from pathlib import Path
from multiprocessing import Manager
from scipy.optimize import minimize
np.set_printoptions(
    precision=3,      # two decimals
    suppress=True,    # never use scientific notation
    floatmode='fixed' # always show exactly `precision` decimals
)


# Setup server
_manager = Manager()
data_proxies = MujocoServerProxies.default(_manager)

event = threading.Event()
signal.signal(signal.SIGTERM, lambda num, frame: event.set())
signal.signal(signal.SIGINT, lambda num, frame: event.set())

scene_xml = str(Path.cwd() / "stretch_mujoco" / "models" / "dock_pen.xml")
server = MujocoServer(scene_xml_path=scene_xml, model=None, stop_mujoco_process_event=event, data_proxies=data_proxies)

# Setup orthographic rendering
h, w = (640, 640)
server.mjmodel.vis.global_.offheight = h
server.mjmodel.vis.global_.offwidth = w
server.mjmodel.vis.global_.orthographic = True
server.mjmodel.vis.global_.fovy = 3.1 * server.mjmodel.stat.extent
renderer = mujoco.Renderer(server.mjmodel, height=h, width=w)
renderer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False # Disables the lidar yellow lines
scale = h / (server.mjmodel.vis.global_.fovy) # pixels per meter

# Setup top-down camera
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam) # inits cam with defaults
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.azimuth = 0
cam.elevation = -90 # look straight down
cam.lookat[:] = server.mjmodel.stat.center
# cam.distance = 10 * server.mjmodel.stat.extent # higher up is closer to light, makes scene brighter

# Set the robot's joints to stowed
stow_config = {'joint_lift': 0.4, "joint_wrist_yaw": 3.4, "joint_wrist_pitch": -0.5}
for joint, pos in stow_config.items():
    joint_id = mujoco.mj_name2id(server.mjmodel, mujoco.mjtObj.mjOBJ_JOINT, joint)
    server.mjdata.qpos[server.mjmodel.jnt_qposadr[joint_id]] = pos
    server.mjdata.qvel[server.mjmodel.jnt_dofadr[joint_id]] = 0.0 # zero out velocity

# Set the robot to a specified base pose
base_jid = mujoco.mj_name2id(server.mjmodel, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
qpos_addr = server.mjmodel.jnt_qposadr[base_jid]
dof_addr = server.mjmodel.jnt_dofadr[base_jid]
translation = [0.0, -0.375, 0.0]
euler = np.array([0.0, 0.0, -1.5708], dtype=np.float64) # rpy
quat = np.zeros(4, dtype=np.float64)
mujoco.mju_euler2Quat(quat, euler, 'xyz') # 'xyz' is intrinsic rotation, 'XYZ' is extrinsic rotation
server.mjdata.qpos[qpos_addr:qpos_addr+7] = np.hstack([translation, quat])
server.mjdata.qvel[dof_addr:dof_addr+6] = np.zeros(6)
mujoco.mj_forward(server.mjmodel, server.mjdata)

# Compute target_x,y,t
t_robot_wrt_world = translation
r_robot_wrt_world = quat
t_world_wrt_robot = np.zeros(3, dtype=np.float64)
r_world_wrt_robot = np.zeros(4, dtype=np.float64)
mujoco.mju_negPose(t_world_wrt_robot, r_world_wrt_robot, t_robot_wrt_world, r_robot_wrt_world)
t_dock_wrt_world = [-1.0+0.35, 0.0, 0.0] # the scoot target is a bit in front of the dock
r_dock_wrt_world = [1.0, 0.0, 0.0, 0.0]
t_dock_wrt_robot  = np.zeros(3, dtype=np.float64)
r_dock_wrt_robot = np.zeros(4, dtype=np.float64)
mujoco.mju_mulPose(t_dock_wrt_robot, r_dock_wrt_robot, t_world_wrt_robot, r_world_wrt_robot, t_dock_wrt_world, r_dock_wrt_world)
target_x = t_dock_wrt_robot[0]
target_y = t_dock_wrt_robot[1]
def rot_about_z(_quat):
    w, x, y, z = _quat
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
target_t = rot_about_z(r_dock_wrt_robot)
print("Target: ", (target_x, target_y, target_t))

# Plan arc
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

ret = plan_arc(target_x, target_y, target_t)
if ret is None:
    print("No arc path found")
R_opt, dTheta_opt, arc_cost = ret
print(f"Arc: ({arc_cost=}, {R_opt=}, {dTheta_opt=})")

# Viz arc's underlying circle
base_id = mujoco.mj_name2id(server.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "base_link")
base_pos = server.mjdata.xpos[base_id]
base_rot = server.mjdata.xmat[base_id].reshape(3,3)
cx, cy = w/2, h/2
def point_in_robot_frame_to_pixel(x, y):
    p_r = np.array([x, y, 0.0])          # point lies in the ground plane
    p_w = base_rot @ p_r + base_pos      # rotate & translate
    dx = p_w[0] - cam.lookat[0]
    dy = p_w[1] - cam.lookat[1]
    # y increases left, while u increases to the right
    # similarly, x increases up, while v increases down
    u = int(-dy * scale + cx)
    v = int(-dx * scale + cy)
    return (u, v)
circle_x, circle_y = (0, R_opt)
circle_uv = point_in_robot_frame_to_pixel(circle_x, circle_y)
circle_radius = int(scale * abs(R_opt))

# Viz arc
angles = np.linspace(0, dTheta_opt, 100)
x_arc = R_opt * np.sin(angles)
y_arc = R_opt * (1 - np.cos(angles))
pixel_pts = []
for x_r, y_r in zip(x_arc, y_arc):
    pixel_pts.append(point_in_robot_frame_to_pixel(x_r, y_r))

# Viz target
px, py = point_in_robot_frame_to_pixel(target_x, target_y)
target_t_in_world = target_t + euler[2]
rect_w, rect_h = 40, 20
angle_deg = np.degrees(target_t_in_world)
rot_rect = ((int(px), int(py)), (rect_w, rect_h), -angle_deg)
box = cv2.boxPoints(rot_rect)
box = np.vstack([box, (int(px), int(py)), box[0]])
box = np.int32(box)

# Render scene
mujoco.mj_forward(server.mjmodel, server.mjdata)
renderer.update_scene(server.mjdata, camera=cam)
img = renderer.render()
cv2.circle(
    img,
    circle_uv,
    circle_radius,
    (255, 192, 203),      # pink
    thickness=1,
    lineType=cv2.LINE_AA  # optional: anti-aliased
)
cv2.polylines(
    img,
    [np.array(pixel_pts, dtype=np.int32)],
    isClosed=False,
    color=(255, 0, 0),    # RGB red
    thickness=2,
    lineType=cv2.LINE_AA
)
cv2.polylines(
    img,
    [box],
    isClosed=True,
    color=(0, 0, 255),    # RGB blue
    thickness=2,
    lineType=cv2.LINE_AA
)
cv2.putText(img, f'{arc_cost=}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite('test.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
