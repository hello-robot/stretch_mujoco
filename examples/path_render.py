from stretch_mujoco.mujoco_server import MujocoServer
from stretch_mujoco.mujoco_server import MujocoServerProxies

import cv2
import mujoco
import signal
import threading
from multiprocessing import Manager

# Setup server
_manager = Manager()
data_proxies = MujocoServerProxies.default(_manager)

event = threading.Event()
signal.signal(signal.SIGTERM, lambda num, frame: event.set())
signal.signal(signal.SIGINT, lambda num, frame: event.set())

server = MujocoServer(scene_xml_path=None, model=None, stop_mujoco_process_event=event, data_proxies=data_proxies)

# Setup rendering
h, w = (480, 640)
server.mjmodel.vis.global_.offheight = h
server.mjmodel.vis.global_.offwidth = w
renderer = mujoco.Renderer(server.mjmodel, height=h, width=w)

# Set to home keyframe
for i in range(server.mjmodel.nkey):
    name = mujoco.mj_id2name(server.mjmodel, mujoco.mjtObj.mjOBJ_KEY, i)
    if name == "home":
        mujoco.mj_resetDataKeyframe(server.mjmodel, server.mjdata, i)
        break
while server.mjdata.time < 1.0:
    mujoco.mj_step(server.mjmodel, server.mjdata)

# Render scene
mujoco.mj_forward(server.mjmodel, server.mjdata)
renderer.update_scene(server.mjdata)
img = renderer.render()
cv2.imwrite('test.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
