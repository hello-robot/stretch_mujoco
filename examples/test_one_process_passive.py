import threading
from stretch_mujoco.config import CameraRates
from stretch_mujoco.stretch_mujoco import launch_server

command = {'val': {}}
status = {'val': {}}
imagery = {'val': {}}
launch_server(
    None, 
    None, 
    CameraRates.hundredHz, 
    True, 
    False, 
    threading.Event(), 
    command, 
    status, 
    imagery,
    cameras_to_use=[]
    )
