from enum import Enum

from stretch_mujoco.status import StretchStatus


class Actuators(Enum):
    """
    An enum for the joints defined in the URDF.
    """

    arm = 0
    gripper = 1
    head_pan = 2
    head_tilt = 3
    lift = 4
    wrist_pitch = 5
    wrist_roll = 6
    wrist_yaw = 7
    base_rotate = 8
    base_translate = 9
    left_wheel_vel = 10
    right_wheel_vel = 11

    def _get_status_attribute(self, is_position: bool, status: StretchStatus) -> float:
        attribute_name = "pos" if is_position else "vel"
        if self == Actuators.arm:
            return getattr(status.arm, attribute_name)
        if self == Actuators.gripper:
            return getattr(status.gripper, attribute_name)
        if self == Actuators.head_pan:
            return getattr(status.head_pan, attribute_name)
        if self == Actuators.head_tilt:
            return getattr(status.head_tilt, attribute_name)
        if self == Actuators.lift:
            return getattr(status.lift, attribute_name)
        if self == Actuators.wrist_pitch:
            return getattr(status.wrist_pitch, attribute_name)
        if self == Actuators.wrist_roll:
            return getattr(status.wrist_roll, attribute_name)
        if self == Actuators.wrist_yaw:
            return getattr(status.wrist_yaw, attribute_name)

        raise NotImplementedError(
            f"Get {'Position' if is_position else 'Velocity'} for {self.name} is not implemented."
        )

    def _get_base_status_attribute(
        self, is_position: bool, status: StretchStatus
    ) -> tuple[float, float, float]:
        x = "x" if is_position else "x_vel"
        y = "y" if is_position else "y_vel"
        theta = "theta" if is_position else "theta_vel"
        if self in [Actuators.base_rotate, Actuators.base_translate]:
            return (
                getattr(status.base, x),
                getattr(status.base, y),
                getattr(status.base, theta),
            )

        raise NotImplementedError(
            f"Get {'Position' if is_position else 'Velocity'}  for {self.name} is not implemented."
        )

    def get_position(self, status: StretchStatus) -> float:
        if self in [Actuators.base_rotate, Actuators.base_translate]:
            raise Exception(f"Please use `get_position_relative()` for {self.name}")
        return self._get_status_attribute(True, status)

    def get_position_relative(self, status: StretchStatus) -> tuple[float, float, float]:
        if self not in [Actuators.base_rotate, Actuators.base_translate]:
            raise Exception(f"Please use `get_position()` for {self.name}")
        return self._get_base_status_attribute(True, status)

    def get_velocity(self, status: StretchStatus) -> float:
        if self in [Actuators.base_rotate, Actuators.base_translate]:
            raise Exception(f"Please use `get_velocity_relative()` for {self.name}")
        return self._get_status_attribute(False, status)

    def get_velocity_relative(self, status: StretchStatus) -> tuple[float, float, float]:
        if self not in [Actuators.base_rotate, Actuators.base_translate]:
            raise Exception(f"Please use `get_velocity()` for {self.name}")
        return self._get_base_status_attribute(False, status)
