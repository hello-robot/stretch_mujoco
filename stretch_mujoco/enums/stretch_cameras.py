from dataclasses import dataclass
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
        return (self.camera_name_in_mjcf, self.name, self.post_processing_callback)

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
        return [StretchCameras.cam_d405_depth, StretchCameras.cam_d435i_depth]

    @property
    def camera_name_in_mjcf(self) -> str:
        if self == StretchCameras.cam_d405_rgb:
            return "d405_rgb"
        if self == StretchCameras.cam_d405_depth:
            return "d405_depth"
        if self == StretchCameras.cam_d435i_rgb:
            return "d435i_camera_rgb"
        if self == StretchCameras.cam_d435i_depth:
            return "d435i_camera_depth"
        if self == StretchCameras.cam_nav_rgb:
            return "nav_camera_rgb"

        raise NotImplementedError(f"Camera {self} camera_name_in_mjcf is not implemented")

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

    @property
    def initial_camera_settings(self):

        if self == StretchCameras.cam_d405_rgb:
            return CameraSettings(
                field_of_view_vertical_in_degrees=58,  # from spec
                focal=(242.56, 242.34),  # from calibration on SE3-3044
                width=480,  # from webteleop
                height=270,  # from webteleop
                crop=CameraCrop(y_min=0, y_max=270, x_min=125, x_max=395),  # from webteleop
                sensor_resolution=(1280, 720),  # from ov9782 spec
                # sensor_pixel_size_micrometers=3.0 # from ov9782 spec
            )

        if self == StretchCameras.cam_d405_depth:
            # Stereo camera, we just use a depth camera camera in mujoco:
            return StretchCameras.cam_d405_rgb.initial_camera_settings

        if self == StretchCameras.cam_d435i_rgb:
            return CameraSettings(
                field_of_view_vertical_in_degrees=42,  # from spec
                focal=(304.24, 304.07),  # from calibration on SE3-3044
                width=424,  # from webteleop
                height=240,  # from webteleop
                sensor_resolution=(1920, 1080),  # from ov2740 spec
                # sensor_pixel_size_micrometers=1.4 # from ov2740 spec
            )

        if self == StretchCameras.cam_d435i_depth:
            return StretchCameras.cam_d435i_rgb.initial_camera_settings
            # TODO: To use these values, depth disparity must be corrected:
        #     return CameraSettings(
        #         field_of_view_vertical_in_degrees=58,  # 58 from spec
        #         focal=(212.31, 212.31),  # from calibration on SE3-3044
        #         width=424,  # from webteleop
        #         height=240,  # from webteleop
        #     )

        if self == StretchCameras.cam_nav_rgb:
            # Arducam B0385 - 70 degrees FOV-X from spec, converted to FOV-Y by field_of_view_vertical_from_horizontal()
            field_of_view_vertical_in_degrees = (
                CameraSettings.field_of_view_vertical_from_horizontal(70, 1280, 720)
            )
            return CameraSettings(
                field_of_view_vertical_in_degrees=field_of_view_vertical_in_degrees,
                focal=(0.0, 0.0),  # TODO We don't have calibrated values.
                width=800,  # from webteleop
                height=600,  # from webteleop
                sensor_resolution=(1280, 720),  # from ov9782 spec
                # sensor_pixel_size_micrometers=3.0 # from ov9782 spec, note: enabling this will not work with 0 `focal`
            )

        raise NotImplementedError(f"Camera {self} initial settings are not implemented")


@dataclass
class CameraCrop:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def x_offset(self):
        return self.x_min

    @property
    def y_offset(self):
        return self.y_min

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def height(self):
        return self.y_max - self.y_min


@dataclass
class CameraSettings:
    field_of_view_vertical_in_degrees: int
    """Vertical FOV for the camera in degrees."""
    focal: tuple[float, float]
    """(x,y) Focal lengths in mm."""
    width: int
    """Width of the rendered image - this is different from `sensor_resolution` which is the max resolution"""
    height: int
    """Height of the rendered image - this is different from `sensor_resolution` which is the max resolution"""
    sensor_resolution: tuple[float, float] | None = None
    """The resolution of the image sensor."""
    sensor_pixel_size_micrometers: float | None = None
    """The size of a single pixel in µm"""
    sensor_size_millimeters: tuple[float, float] | None = None
    """Optional, sensor_size() can calculate this if you specify `sensor_pixel_size_micrometers` and `sensor_resolution`"""
    crop: CameraCrop | None = None
    """This is currently being used in Stretch Web Teleop to crop a ROI"""
    distortion_params: tuple | None = None
    """Specify this if they are available. Zeros will be used in `get_distortion_params_d()` otherwise."""

    @property
    def sensor_size(self) -> tuple[float, float] | None:
        """
        Returns the `sensor_size_millimeters` property if it is not None.
        Otherwise, calculated the sensor size if `sensor_pixel_size_micrometers` and `sensor_resolution` are given.
        Otherwise, returns None.
        The dimensions of the camera sensor can be calculated from the sensor's resolution (width and height) multiplied with its pixel size, if they are known.
        """
        if self.sensor_size_millimeters is not None:
            return self.sensor_size_millimeters

        if self.sensor_pixel_size_micrometers is None or self.sensor_resolution is None:
            return None

        return (
            self.sensor_pixel_size_micrometers * self.sensor_resolution[0] / 1000,
            self.sensor_pixel_size_micrometers * self.sensor_resolution[1] / 1000,
        )  # mm

    @staticmethod
    def field_of_view_vertical_from_horizontal(
        fov_horizontal_degrees: int, width: int, height: int
    ) -> int:
        """Calculates vertical FOV from horizontal using aspect ratio."""
        horizontal_fov = np.radians(fov_horizontal_degrees)
        aspect_ratio = width / height
        vertical_fov = np.rad2deg(2 * np.arctan(np.tan(horizontal_fov / 2) * aspect_ratio))
        return int(abs(vertical_fov))

    def get_distortion_params_d(self):
        """
        Distortion Parameters (D):
        D is an array of floating-point numbers representing the camera's distortion coefficients.
        These coefficients describe how the camera lens distorts the image.
        The number of parameters and their interpretation depend on the distortion_model field.
        For the common "plumb_bob" model, D contains five parameters: (k1, k2, t1, t2, k3), representing radial and tangential distortion.
        k1, k2, and k3 are radial distortion coefficients.
        t1 and t2 are tangential distortion coefficients.
        """
        return self.distortion_params or [0.0] * 5

    def get_intrinsic_params_k(self):
        """
        Intrinsic Camera Matrix (K):
        K is a 3x3 matrix describing the camera's intrinsic parameters: focal lengths and principal point.
        It represents the transformation from normalized camera coordinates to pixel coordinates.
        The matrix has the following form:
        K = [fx 0 cx]
            [0 fy cy]
            [0  0  1]
        fx and fy are the focal lengths in pixels along the x and y axes, respectively.
        cx and cy are the coordinates of the principal point (center of the image) in pixels.
        """
        return [
            self.focal[0],
            0.0,
            self.width / 2,
            0.0,
            self.focal[1],
            self.height / 2,
            0.0,
            0.0,
            1.0,
        ]

    def get_projection_matrix_p(self):
        """
        P is a 3x4 projection matrix that projects 3D points in the camera coordinate frame onto the 2D image plane.
        It's typically derived from the camera's intrinsic matrix (K) and may include additional transformations like rotation and translation.
        The matrix has the following form:
        P = [fx' 0 cx' Tx]
            [0 fy' cy' Ty]
            [0  0  1   0]
        fx' and fy' are the focal lengths in pixels for the rectified image.
        cx' and cy' represent the principal point (center of the image) in pixels.
        Tx and Ty are used in stereo setups to represent the translation of the second camera relative to the first. For monocular cameras, Tx and Ty are typically 0.

        """
        return [
            self.focal[0],
            0.0,
            self.width / 2,
            0.0,
            0.0,
            self.focal[1],
            self.height / 2,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
