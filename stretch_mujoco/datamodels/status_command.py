"""
Dataclasses that communicate movement commands to Mujoco.
"""
from dataclasses import dataclass, field

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
    move_to:list[CommandMove]|None = field(default_factory=None)
    move_by:list[CommandMove]|None = field(default_factory=None)
    set_base_velocity:CommandBaseVelocity|None = field(default_factory=None)
    keyframe:CommandKeyframe|None = field(default_factory=None)


    @staticmethod
    def default():
        """
        Returns an empty instance with None or zeros for properties.
        """
        return StatusCommand()
