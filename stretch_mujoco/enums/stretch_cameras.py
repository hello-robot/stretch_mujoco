from enum import Enum
from typing import Callable

import numpy as np

from stretch_mujoco import config, utils


class StretchCamera(Enum):
    cam_d405_rgb = 0
    cam_d405_depth = 1

    cam_d435i_rgb = 2
    cam_d435i_depth = 3

    cam_nav_rgb = 4

    def get_render_params(self):
        return (self.camera_name_in_scene, self.name, self.post_processing_callback)

    @staticmethod
    def all() -> list["StretchCamera"]:
        """
        Returns all the available cameras
        """
        return [camera for camera in StretchCamera]

    @staticmethod
    def none() -> list["StretchCamera"]:
        """
        Short-hand for not using any cameras.
        """
        return []

    @staticmethod
    def rgb() -> list["StretchCamera"]:
        """
        Returns the RGB camera's only
        """
        return [
            StretchCamera.cam_d405_rgb,
            StretchCamera.cam_d435i_rgb,
            StretchCamera.cam_d435i_rgb,
        ]
    @staticmethod
    def depth() -> list["StretchCamera"]:
        """
        Returns the Depth camera's only
        """
        return [
            StretchCamera.cam_d405_depth,
            StretchCamera.cam_d435i_depth
        ]

    @property
    def camera_name_in_scene(self) -> str:
        if self == StretchCamera.cam_d405_depth or self == StretchCamera.cam_d405_rgb:
            return "d405_rgb"
        if self == StretchCamera.cam_d435i_depth or self == StretchCamera.cam_d435i_rgb:
            return "d435i_camera_rgb"
        if self == StretchCamera.cam_nav_rgb:
            return "nav_camera_rgb"

        raise NotImplementedError(f"Camera {self} camera_name_in_scene is not implemented")

    @property
    def is_depth(self) -> bool:
        if self == StretchCamera.cam_d405_depth or self == StretchCamera.cam_d435i_depth:
            return True
        if (
            self == StretchCamera.cam_d405_rgb
            or self == StretchCamera.cam_d435i_rgb
            or self == StretchCamera.cam_nav_rgb
        ):
            return False

        raise NotImplementedError(f"Camera {self} is_depth is not implemented")

    @property
    def post_processing_callback(self) -> Callable[[np.ndarray], np.ndarray] | None:

        if self == StretchCamera.cam_d405_depth:
            return lambda render: utils.limit_depth_distance(render, config.depth_limits["d405"])

        if self == StretchCamera.cam_d435i_depth:
            return lambda render: utils.limit_depth_distance(render, config.depth_limits["d435i"])

        if (
            self == StretchCamera.cam_d405_rgb
            or self == StretchCamera.cam_d435i_rgb
            or self == StretchCamera.cam_nav_rgb
        ):
            return None

        raise NotImplementedError(f"Camera {self} post_processing_callback is not implemented")
