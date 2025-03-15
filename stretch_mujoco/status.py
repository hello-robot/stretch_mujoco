from dataclasses import asdict, dataclass
import numpy as np

from stretch_mujoco.enums.stretch_cameras import StretchCamera
from stretch_mujoco.utils import dataclass_from_dict

@dataclass
class PositionVelocity:
    pos: float
    vel: float

    @staticmethod
    def default():
        return PositionVelocity(0, 0)

@dataclass
class BaseStatus:
    x:float
    y:float
    theta:float
    x_vel:float
    theta_vel:float

    @staticmethod
    def default():
        return BaseStatus(0, 0, 0,0,0)

@dataclass
class StretchStatus:
    time: float
    fps:float
    base:BaseStatus
    lift: PositionVelocity
    arm: PositionVelocity
    head_pan: PositionVelocity
    head_tilt: PositionVelocity
    wrist_yaw: PositionVelocity
    wrist_pitch: PositionVelocity
    wrist_roll: PositionVelocity
    gripper: PositionVelocity

    def __getitem__(self, name:str):
        # Backward compatibility
        return getattr(self, name)

    @staticmethod
    def default():
        return StretchStatus(
            0,
            0,
            BaseStatus.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default()
        )
    
    def to_dict(self):
        return asdict(self)
    
    @staticmethod
    def from_dict(dict_data:dict)-> "StretchStatus": 
        return dataclass_from_dict(StretchStatus, dict_data) #type: ignore


@dataclass
class StretchCameraStatus:
    time: float
    fps: float
    cam_d405_rgb: np.ndarray|None = None
    cam_d405_depth:np.ndarray|None = None
    cam_d405_K: np.ndarray|None = None
    cam_d435i_rgb:np.ndarray|None = None
    cam_d435i_depth:np.ndarray|None = None
    cam_d435i_K: np.ndarray|None = None
    cam_nav_rgb: np.ndarray|None = None

    def to_dict(self):
        return asdict(self)
    
    def set_camera_data(self, camera:StretchCamera, data:np.ndarray):
        if camera == StretchCamera.cam_d405_rgb:
            self.cam_d405_rgb = data
            return
        if camera == StretchCamera.cam_d405_depth:
            self.cam_d405_depth = data
            return
        if camera == StretchCamera.cam_d435i_rgb:
            self.cam_d435i_rgb = data
            return
        if camera == StretchCamera.cam_d435i_depth:
            self.cam_d435i_depth = data
            return
        if camera == StretchCamera.cam_nav_rgb:
            self.cam_nav_rgb = data
            return
        
        raise NotImplementedError(f"Camera {camera} is not implemented.")

    def get_camera_data(self, camera:StretchCamera) -> np.ndarray:
        data:np.ndarray|None = None
        if camera == StretchCamera.cam_d405_rgb:
            data = self.cam_d405_rgb
        if camera == StretchCamera.cam_d405_depth:
            data = self.cam_d405_depth
        if camera == StretchCamera.cam_d435i_rgb:
            data = self.cam_d435i_rgb
        if camera == StretchCamera.cam_d435i_depth:
            data = self.cam_d435i_depth
        if camera == StretchCamera.cam_nav_rgb:
            data = self.cam_nav_rgb

        if data is None:
            raise ValueError(f"Tried to get {camera} data, but it is empty.")
        
        return data
    
    @staticmethod
    def default():
        return StretchCameraStatus(0, 0)
    