from stretch_mujoco.mujoco_server import MujocoServer
from stretch_mujoco.mujoco_server import MujocoServerProxies

import cv2
import mujoco
import signal
import threading
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

# Setup rendering
h, w = (640, 640)
server.mjmodel.vis.global_.offheight = h
server.mjmodel.vis.global_.offwidth = w
renderer = mujoco.Renderer(server.mjmodel, height=h, width=w)

# Setup orthographic top-down camera
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam) # inits cam with defaults
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.orthographic = True
cam.azimuth = 0
cam.elevation = -90 # look straight down
cam.lookat[:] = server.mjmodel.stat.center
cam.distance = 3.5*server.mjmodel.stat.extent  # zoom so whole scene fits

# Set the robot's joints to stowed
stow_config = {'joint_lift': 0.4, "joint_wrist_yaw": 3.4, "joint_wrist_pitch": -0.5}
for joint, pos in stow_config.items():
    joint_id = mujoco.mj_name2id(server.mjmodel, mujoco.mjtObj.mjOBJ_JOINT, joint)
    server.mjdata.qpos[server.mjmodel.jnt_qposadr[joint_id]] = pos
    server.mjdata.qvel[server.mjmodel.jnt_dofadr[joint_id]] = 0.0 # zero out velocity

# Render scene
mujoco.mj_forward(server.mjmodel, server.mjdata)
renderer.update_scene(server.mjdata, camera=cam)
img = renderer.render()
cv2.imwrite('test.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
