import threading
from stretch_mujoco.stretch_mujoco import launch_server

command = {"val": {}}
status = {"val": {}}
imagery = {"val": {}}
launch_server(
    None, None, 20, False, False, threading.Event(), command, status, imagery, []
)
