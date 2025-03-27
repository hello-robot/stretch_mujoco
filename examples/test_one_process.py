import signal
import threading

from stretch_mujoco.mujoco_server import MujocoServer

command = {"val": {}}
status = {"val": {}}
imagery = {"val": {}}

event = threading.Event()
signal.signal(signal.SIGTERM, lambda num, frame: event.set())
signal.signal(signal.SIGINT, lambda num, frame: event.set())

MujocoServer.launch_server(
    scene_xml_path=None, 
    model=None, 
    camera_hz=30, 
    show_viewer_ui=True,
    stop_mujoco_process_event=event, 
    command=command, 
    status=status, 
    imagery=imagery,
    cameras_to_use=[]
)
