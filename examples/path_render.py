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
translation = [0.0, -0.6, 0.0]
euler = np.array([0.0, 0.0, -0.3], dtype=np.float64) # rpy
quat = np.zeros(4, dtype=np.float64)
mujoco.mju_euler2Quat(quat, euler, 'xyz') # 'xyz' is intrinsic rotation, 'XYZ' is extrinsic rotation
server.mjdata.qpos[qpos_addr:qpos_addr+7] = np.hstack([translation, quat])
server.mjdata.qvel[dof_addr:dof_addr+6] = np.zeros(6)

# Compute target_x,y,t
t_robot_wrt_world = translation
r_robot_wrt_world = quat
t_world_wrt_robot = np.zeros(3, dtype=np.float64)
r_world_wrt_robot = np.zeros(4, dtype=np.float64)
mujoco.mju_negPose(t_world_wrt_robot, r_world_wrt_robot, t_robot_wrt_world, r_robot_wrt_world)
t_dock_wrt_world = [-1.0+0.625, 0.0, 0.0] # the scoot target is 0.625m in front of the dock
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

# TODO: Viz arc
angles = np.linspace(0, dTheta_opt, 100)
x_arc = R_opt * np.sin(angles)
y_arc = R_opt * (1 - np.cos(angles))
print("arc_cost: ", arc_cost)
print(x_arc)

# Render scene
mujoco.mj_forward(server.mjmodel, server.mjdata)
renderer.update_scene(server.mjdata, camera=cam)
img = renderer.render()
cv2.putText(img, '', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite('test.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
