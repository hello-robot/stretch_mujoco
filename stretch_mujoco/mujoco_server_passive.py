import threading
import time
from typing import override

import mujoco
import mujoco._functions
from stretch_mujoco.mujoco_server import MujocoServer
from stretch_mujoco.utils import FpsCounter

class MujocoServerPassive(MujocoServer):
    """
    A MujocoServer flavor that uses the mujoco passive viewer. 

    Use `MujocoServerPassive.launch_server()` to start the simulator.
    
    On MacOS, this needs to be started with `mjpython`.

    https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer 
    """
    def _do_physics(self, viewer):
        fps = FpsCounter()
        while viewer.is_running() and not self.stop_event.is_set():
            fps.tick()
            print(f"Physics thread: {fps.fps=}, {self.simulation_fps_counter.fps=}")
            start_ts = time.perf_counter()

            with viewer.lock():
                mujoco._functions.mj_step(self.mjmodel, self.mjdata)
                
                self._ctrl_callback(self.mjmodel, self.mjdata)


            time_until_next_step = self.mjmodel.opt.timestep - (time.time() - start_ts)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        

    @override
    def _run(self, show_viewer_ui: bool):
        with self.viewer.launch_passive(self.mjmodel, self.mjdata, show_left_ui=show_viewer_ui, show_right_ui=show_viewer_ui) as viewer:
            
            physics_thread = threading.Thread(target=self._do_physics, args=(viewer,))
            physics_thread.start()

            fps = FpsCounter()

            while viewer.is_running() and not self.stop_event.is_set():
                fps.tick()
                print(f"UI thread: {fps.fps=}, {self.simulation_fps_counter.fps=}")

                viewer.sync()

            physics_thread.join()