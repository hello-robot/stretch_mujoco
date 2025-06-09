import copy
from dataclasses import asdict, dataclass
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
class StatusStretchJoints:
    time: float
    fps:float
    sim_to_real_time_ratio_msg: str
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
        """For backward compatibility: allows access with the square brackets []"""
        return getattr(self, name)
    
    def to_dict(self):
        return asdict(self)
    
    def copy(self):
        return StatusStretchJoints.from_dict(copy.copy(self.to_dict()))
    
    @staticmethod
    def from_dict(dict_data:dict)-> "StatusStretchJoints": 
        return dataclass_from_dict(StatusStretchJoints, dict_data) #type: ignore


    @staticmethod
    def default():
        """
        Returns an empty instance with None or zeros for properties.
        """
        return StatusStretchJoints(
            0,
            0,
            "",
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
