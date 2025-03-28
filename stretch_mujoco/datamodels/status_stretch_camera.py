import copy
from dataclasses import asdict, dataclass, fields
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

    def get_all(self)-> list[tuple[str, np.ndarray]]:
        """Returns the camera `(name, pixels)` that are available (not None).

        Note: Alternatively, use `get_camera_data()` to get a specific camera's data.
        """
        return [(name, data) for name, data in self.to_dict().items() if isinstance(data, np.ndarray) and "_K" not in name ]

    def get_camera_data(self, camera:StretchCameras) -> np.ndarray:
        """
        Use this to match a StretchCameras enum instance with its property in StatusStretchCameras dataclass, to get the camera data property.
        """
        data:np.ndarray|None = None
        if camera == StretchCameras.cam_d405_rgb:
            data = self.cam_d405_rgb
        if camera == StretchCameras.cam_d405_depth:
            data = self.cam_d405_depth
        if camera == StretchCameras.cam_d435i_rgb:
            data = self.cam_d435i_rgb
        if camera == StretchCameras.cam_d435i_depth:
            data = self.cam_d435i_depth
        if camera == StretchCameras.cam_nav_rgb:
            data = self.cam_nav_rgb

        if data is None:
            raise ValueError(f"Tried to get {camera} data, but it is empty.")
        
        return data
    
    def set_camera_data(self, camera:StretchCameras, data:np.ndarray):
        """
        Use this to match a StretchCameras enum instance with its property in StatusStretchCameras dataclass, to set the camera data property.
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
    
    
