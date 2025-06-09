"""
Dataclasses that communicate movement commands to Mujoco.
"""

import copy
from dataclasses import asdict, dataclass, field

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.utils import dataclass_from_dict


@dataclass
class CommandMove:
    actuator_name: str
    trigger: bool
    pos: float


@dataclass
class CommandBaseVelocity:
    v_linear: float
    omega: float
    trigger: bool


@dataclass
class CommandKeyframe:
    name: str
    trigger: bool


@dataclass
class CommandCoordinateFrameArrowsViz:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    trigger: bool


@dataclass
class StatusCommand:
    """
    A dataclass to ferry movement commands to the Mujoco server.
    """

    move_to: dict[str, CommandMove] = field(default_factory=dict)
    move_by: dict[str, CommandMove] = field(default_factory=dict)
    base_velocity: CommandBaseVelocity = field(default_factory=lambda:CommandBaseVelocity(0, 0, False))
    keyframe: CommandKeyframe = field(default_factory=lambda:CommandKeyframe("", False))
    coordinate_frame_arrows_viz: list[CommandCoordinateFrameArrowsViz] = field(default_factory=list)



    def set_move_to(self, command: CommandMove):
        """Sends a move_to command and removes the move_by command."""
        self.move_to[command.actuator_name] = command

        self.move_by.pop(command.actuator_name, None)

    def set_move_by(self, command: CommandMove):
        """Sends a move_by command and removes the move_to command."""
        self.move_by[command.actuator_name] = command

        self.move_to.pop(command.actuator_name, None)

    def set_base_velocity(self, command: CommandBaseVelocity):
        """Sends the velocity command and removes the move_to and move_by commands."""
        self.base_velocity = command

        for actuator in [
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
            Actuators.base_rotate,
            Actuators.base_translate,
        ]:
            self.move_to.pop(actuator.name, None)
            self.move_by.pop(actuator.name, None)

    def to_dict(self):
        return asdict(self)

    def copy(self):
        return StatusCommand.from_dict(copy.copy(self.to_dict()))

    @staticmethod
    def from_dict(dict_data: dict) -> "StatusCommand":
        command: StatusCommand = dataclass_from_dict(StatusCommand, dict_data)  # type: ignore

        command.move_to = {
            key: dataclass_from_dict(CommandMove, val) for key, val in command.move_to.items()  # type: ignore
        }
        command.move_by = {
            key: dataclass_from_dict(CommandMove, val) for key, val in command.move_by.items()  # type: ignore
        }
        return command

    @staticmethod
    def default():
        """
        Returns an empty instance with None or zeros for properties.
        """
        return StatusCommand()
