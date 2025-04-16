import copy
from dataclasses import asdict, dataclass
import numpy as np

from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.utils import dataclass_from_dict


@dataclass
class StatusStretchCameras:
    """
    A dataclass and helper methods to pack camera data. 
    """
    time: float
    fps: float

    cam_d405_rgb: np.ndarray|None = None
    cam_d405_depth:np.ndarray|None = None
    cam_d405_K: np.ndarray|None = None

    cam_d435i_rgb:np.ndarray|None = None
    cam_d435i_depth:np.ndarray|None = None
    cam_d435i_K: np.ndarray|None = None

    cam_nav_rgb: np.ndarray|None = None

    def get_all(self, auto_rotate: bool = True)-> dict[StretchCameras, np.ndarray]:
        """Returns the camera `{StretchCameras: pixels}` that are available (not None).

        `auto_rotate` will correct the rotation of the cam_d435i_rgb and cam_d435i_depth from their innately rotated optical frame.

        Note: This get the values inside this dataclass; it does not poll from the simulator.
        
        Note: Alternatively, use `get_camera_data()` to get a specific camera's data.
        """
        data: dict[StretchCameras, np.ndarray] = {}
        for camera in StretchCameras.all():
            try:
                data[camera] = self.get_camera_data(camera=camera, auto_rotate=auto_rotate)
            except ValueError: ... # get_camera_data throws a ValueError when the value is None or doesn't exist.

        return data
    
    def get_camera_data(self, camera:StretchCameras, auto_rotate: bool = True) -> np.ndarray:
        """
        Use this to get the camera data (pixels) using a StretchCameras instance.
        
        Throws a ValueError if the data is None.

        `auto_rotate` will correct the rotation of the cam_d435i_rgb and cam_d435i_depth from their innately rotated optical frame.

        Note: This get the values inside this dataclass; it does not poll from the simulator.
        """
        data:np.ndarray|None = None
        if camera == StretchCameras.cam_d405_rgb:
            data = self.cam_d405_rgb
        elif camera == StretchCameras.cam_d405_depth:
            data = self.cam_d405_depth
        elif camera == StretchCameras.cam_d435i_rgb and self.cam_d435i_rgb is not None:
            data = np.rot90(self.cam_d435i_rgb, -1) if auto_rotate else self.cam_d435i_rgb
        elif camera == StretchCameras.cam_d435i_depth and self.cam_d435i_depth is not None:
            data = np.rot90(self.cam_d435i_depth, -1) if auto_rotate else self.cam_d435i_depth
        elif camera == StretchCameras.cam_nav_rgb:
            data = self.cam_nav_rgb

        if data is None:
            raise ValueError(f"Tried to get {camera} data, but it is empty or not implemented.")
        
        return data
    
    def set_camera_data(self, camera:StretchCameras, data:np.ndarray):
        """
        Use this to match a StretchCameras enum instance with its property in StatusStretchCameras dataclass, to set the camera data property within this dataclass.

        Note: This sets the values inside this dataclass; it does not send data to the simulator.
        """
        if camera == StretchCameras.cam_d405_rgb:
            self.cam_d405_rgb = data
            return
        if camera == StretchCameras.cam_d405_depth:
            self.cam_d405_depth = data
            return
        if camera == StretchCameras.cam_d435i_rgb:
            self.cam_d435i_rgb = data
            return
        if camera == StretchCameras.cam_d435i_depth:
            self.cam_d435i_depth = data
            return
        if camera == StretchCameras.cam_nav_rgb:
            self.cam_nav_rgb = data
            return
        
        raise NotImplementedError(f"Camera {camera} is not implemented.")

    @staticmethod
    def default():
        """
        Returns an empty instance with None or zeros for properties.
        """
        return StatusStretchCameras(time=0, fps=0)
    
    def to_dict(self):
        return asdict(self)
    
    def copy(self):
        return StatusStretchCameras.from_dict(copy.copy(self.to_dict()))
    
    @staticmethod
    def from_dict(dict_data:dict)-> "StatusStretchCameras": 
        return dataclass_from_dict(StatusStretchCameras, dict_data) #type: ignore
    
    
