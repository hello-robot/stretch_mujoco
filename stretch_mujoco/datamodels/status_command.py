"""
Dataclasses that communicate movement commands to Mujoco.
"""
from dataclasses import dataclass

@dataclass
class CommandMove:
    actuator_name: str
    trigger:bool
    pos: float

@dataclass
class CommandBaseVelocity:
    v_linear: float
    omega: float
    trigger:bool

@dataclass
class CommandKeyframe:
    name: str
    trigger: bool

@dataclass
class StatusCommand:
    """
    A dataclass to ferry movement commands to the Mujoco server.
    """
    move_to:list[CommandMove]|None = None
    move_by:list[CommandMove]|None = None
    set_base_velocity:CommandBaseVelocity|None = None
    keyframe:CommandKeyframe|None = None


    @staticmethod
    def default():
        """
        Returns an empty instance with None or zeros for properties.
        """
        return StatusCommand()
