from stretch_mujoco.mujoco_server import MujocoServer
from stretch_mujoco.mujoco_server import MujocoServerProxies

import cv2
import mujoco
import signal
import threading
import numpy as np
from pathlib import Path
from multiprocessing import Manager

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
t_dock_wrt_world = [-1.0, 0.0, 0.0]
r_dock_wrt_world = [1.0, 0.0, 0.0, 0.0]
t_dock_wrt_robot  = np.zeros(3, dtype=np.float64)
r_dock_wrt_robot = np.zeros(4, dtype=np.float64)
mujoco.mju_mulPose(t_dock_wrt_robot, r_dock_wrt_robot, t_world_wrt_robot, r_world_wrt_robot, t_dock_wrt_world, r_dock_wrt_world)
print(t_dock_wrt_robot)
print(r_dock_wrt_robot)
print()
mujoco.mj_forward(server.mjmodel, server.mjdata)

# Verify target
def quat_conjugate(q):
    """Return the conjugate (inverse for unit quaternions)."""
    # q = [w, x, y, z]
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
def rel_pose_quat(model, data, linkA, linkB):
    # look up IDs
    idA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, linkA)
    idB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, linkB)

    # get world poses: note .xipos/.xiquat not .body_xpos/.body_xquat
    posA, quatA = data.xpos[idA], data.xquat[idA]
    posB, quatB = data.xpos[idB], data.xquat[idB]
    print(linkA, posA)
    print(linkB, posB)

    # inverse (conjugate) of A’s quaternion
    qA_inv = quat_conjugate(quatA)

    # vector from A to B, in world
    d = posB - posA

    # rotate d into A’s frame
    d_rel = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(d_rel, d, qA_inv)

    # quaternion multiply: qA_inv ⊗ qB
    q_rel = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(q_rel, qA_inv, quatB)

    return d_rel, q_rel
H_dock_wrt_robot = rel_pose_quat(server.mjmodel, server.mjdata, "base_link", "link_docking_station")
print(H_dock_wrt_robot)
"""
example output:
[-1.13264861  0.27768169  0.        ]
[0.98877108 0.         0.         0.14943813]

base_link [ 0.  -0.6  0. ]
link_docking_station [-1.    0.    0.01]
(array([-1.13264861,  0.27768169,  0.01      ]), array([0.98877108, 0.        , 0.        , 0.14943813]))
"""

# Render scene
mujoco.mj_forward(server.mjmodel, server.mjdata)
renderer.update_scene(server.mjdata, camera=cam)
img = renderer.render()
cv2.putText(img, '', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite('test.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
