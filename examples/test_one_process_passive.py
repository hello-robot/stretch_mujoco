from multiprocessing import Manager
import signal
import threading

from stretch_mujoco.mujoco_server import MujocoServerProxies
from stretch_mujoco.mujoco_server_passive import MujocoServerPassive
from stretch_mujoco.status import CommandStatus, StretchCameraStatus, StretchStatus

_manager = Manager()
data_proxies = MujocoServerProxies(
            _command=_manager.dict({"val": CommandStatus.default()}),
            _status=_manager.dict({"val": StretchStatus.default()}),
            _cameras=_manager.dict({"val": StretchCameraStatus.default()}),
        )

event = threading.Event()
signal.signal(signal.SIGTERM, lambda num, frame: event.set())
signal.signal(signal.SIGINT, lambda num, frame: event.set())

MujocoServerPassive.launch_server(
    scene_xml_path=None, 
    model=None, 
    camera_hz=30, 
    show_viewer_ui=True,
    stop_mujoco_process_event=event, 
    data_proxies=data_proxies,
    cameras_to_use=[]
    )
