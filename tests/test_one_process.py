from multiprocessing import Manager
import signal
import threading

from stretch_mujoco.mujoco_server import MujocoServer

from stretch_mujoco.mujoco_server import MujocoServerProxies

_manager = Manager()
data_proxies = MujocoServerProxies.default(_manager)

event = threading.Event()
signal.signal(signal.SIGTERM, lambda num, frame: event.set())
signal.signal(signal.SIGINT, lambda num, frame: event.set())

MujocoServer.launch_server(
    scene_xml_path=None, 
    model=None, 
    camera_hz=30, 
    show_viewer_ui=True,
    stop_mujoco_process_event=event, 
    data_proxies=data_proxies,
    cameras_to_use=[]
)
