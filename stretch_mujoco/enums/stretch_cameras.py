from enum import Enum
from typing import Callable

import numpy as np

from stretch_mujoco import config, utils


class StretchCameras(Enum):
    """
    An enum of the camera's available to the simulation.
    """
    
    cam_d405_rgb = 0
    cam_d405_depth = 1

    cam_d435i_rgb = 2
    cam_d435i_depth = 3

    cam_nav_rgb = 4

    def get_render_params(self):
        return (self.camera_name_in_scene, self.name, self.post_processing_callback)

    @staticmethod
    def all() -> list["StretchCameras"]:
        """
        Returns all the available cameras
        """
        return [camera for camera in StretchCameras]

    @staticmethod
    def none() -> list["StretchCameras"]:
        """
        Short-hand for not using any cameras.
        """
        return []

    @staticmethod
    def rgb() -> list["StretchCameras"]:
        """
        Returns the RGB camera's only
        """
        return [
            StretchCameras.cam_d405_rgb,
            StretchCameras.cam_d435i_rgb,
            StretchCameras.cam_nav_rgb,
        ]
    @staticmethod
    def depth() -> list["StretchCameras"]:
        """
        Returns the Depth camera's only
        """
        return [
            StretchCameras.cam_d405_depth,
            StretchCameras.cam_d435i_depth
        ]

    @property
    def camera_name_in_scene(self) -> str:
        if self == StretchCameras.cam_d405_depth or self == StretchCameras.cam_d405_rgb:
            return "d405_rgb"
        if self == StretchCameras.cam_d435i_depth or self == StretchCameras.cam_d435i_rgb:
            return "d435i_camera_rgb"
        if self == StretchCameras.cam_nav_rgb:
            return "nav_camera_rgb"

        raise NotImplementedError(f"Camera {self} camera_name_in_scene is not implemented")

    @property
    def is_depth(self) -> bool:
        if self == StretchCameras.cam_d405_depth or self == StretchCameras.cam_d435i_depth:
            return True
        if (
            self == StretchCameras.cam_d405_rgb
            or self == StretchCameras.cam_d435i_rgb
            or self == StretchCameras.cam_nav_rgb
        ):
            return False

        raise NotImplementedError(f"Camera {self} is_depth is not implemented")

    @property
    def post_processing_callback(self) -> Callable[[np.ndarray], np.ndarray] | None:

        if self == StretchCameras.cam_d405_depth:
            return lambda render: utils.limit_depth_distance(render, config.depth_limits["d405"])

        if self == StretchCameras.cam_d435i_depth:
            return lambda render: utils.limit_depth_distance(render, config.depth_limits["d435i"])

        if (
            self == StretchCameras.cam_d405_rgb
            or self == StretchCameras.cam_d435i_rgb
            or self == StretchCameras.cam_nav_rgb
        ):
            return None

        raise NotImplementedError(f"Camera {self} post_processing_callback is not implemented")
