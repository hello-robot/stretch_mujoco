import time
from typing import override

import mujoco
import mujoco._functions
from stretch_mujoco.mujoco_server import MujocoServer


class MujocoServerPassive(MujocoServer):
    @override
    def _run(self, show_viewer_ui: bool):
        with self.viewer.launch_passive(self.mjmodel, self.mjdata, show_left_ui=show_viewer_ui, show_right_ui=show_viewer_ui) as viewer:
            
            while viewer.is_running() and not self.stop_event.is_set():

                start_ts = time.perf_counter()

                mujoco._functions.mj_step(self.mjmodel, self.mjdata)
                self._ctrl_callback(self.mjmodel, self.mjdata)

                viewer.sync()

                elapsed = time.perf_counter() - start_ts
                if elapsed < self.mjmodel.opt.timestep:
                    time.sleep(self.mjmodel.opt.timestep - elapsed)