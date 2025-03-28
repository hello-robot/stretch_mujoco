from multiprocessing import Manager
import signal
import threading

from stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from stretch_mujoco.mujoco_server import MujocoServer

from stretch_mujoco.mujoco_server import MujocoServerProxies
from stretch_mujoco.datamodels.status_command import StatusCommand

_manager = Manager()
data_proxies = MujocoServerProxies(
            _command=_manager.dict({"val": StatusCommand.default()}),
            _status=_manager.dict({"val": StatusStretchJoints.default()}),
            _cameras=_manager.dict({"val": StatusStretchCameras.default()}),
        )


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
