from stretch_mujoco.mujoco_server import MujocoServer
from stretch_mujoco.mujoco_server import MujocoServerProxies

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

# Render scene
mujoco.mj_forward(server.mjmodel, server.mjdata)
renderer.update_scene(server.mjdata)
img = renderer.render()
print(img.shape)
print(img)
print(type(img))