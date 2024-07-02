import time
import threading
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData
import cv2
import numpy as np


class RealsenseMujocoSimulator:
    """
    StretchMujocoSimulator sample class for simulating Stretch robot in Mujoco
    """

    def __init__(self, scene_xml_path: str = "./floating_realsense.xml"):
        self.mjmodel = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.mjdata = mujoco.MjData(self.mjmodel)

        self.rgb_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer.enable_depth_rendering()

    def pull_camera_data(self) -> dict:
        """
        Pull camera data from the simulator
        """
        data = {}
        data["time"] = self.mjdata.time
        self.rgb_renderer.update_scene(self.mjdata, "d435i_camera_rgb")
        self.depth_renderer.update_scene(self.mjdata, "d435i_camera_rgb")
        data["cam_d435i_rgb"] = cv2.rotate(
            cv2.cvtColor(self.rgb_renderer.render(), cv2.COLOR_RGB2BGR),
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )
        data["cam_d435i_depth"] = cv2.rotate(
            self.depth_renderer.render(), cv2.ROTATE_90_COUNTERCLOCKWISE
        )

        return data

    def __ctrl_callback(self, model: MjModel, data: MjData) -> None:
        """
        Callback function that gets executed with mj_step
        """
        self.mjdata = data
        self.mjmodel = model

    def __run(self) -> None:
        mujoco.set_mjcb_control(self.__ctrl_callback)
        mujoco.viewer.launch(self.mjmodel)

    def start(self) -> None:
        """
        Start the simulator in a using blocking Managed-vieiwer for precise timing. And user code
        is looped through callback. Some projects might need non-blocking Passive-vieiwer.
        For more info visit: https://mujoco.readthedocs.io/en/stable/python.html#managed-viewer
        """
        threading.Thread(target=self.__run).start()
        time.sleep(0.5)


if __name__ == "__main__":
    sim = RealsenseMujocoSimulator()
    sim.start()
    sim.mjmodel.body('d435i').pos = np.array([0., 0., 0.5])
    sim.mjmodel.body('d435i').quat = np.array([0., 0., -0.71, 0.71])
    # display camera feeds
    while True:
        camera_data = sim.pull_camera_data()
        cv2.imshow("cam_d435i_rgb", camera_data["cam_d435i_rgb"])
        cv2.imshow("cam_d435i_depth", camera_data["cam_d435i_depth"])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
