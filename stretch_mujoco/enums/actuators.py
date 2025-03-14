
from enum import Enum


class Actuators(Enum):
    arm = 0
    gripper = 1
    head_pan = 2
    head_tilt = 3
    left_wheel_vel = 4
    lift = 5
    right_wheel_vel = 6
    wrist_pitch = 7
    wrist_roll = 8
    wrist_yaw = 9
    base_rotate = 10
    base_translate = 11

    @property
    def is_position_actuator(self):
        if self == Actuators.left_wheel_vel or self == Actuators.right_wheel_vel:
            return False
        return True

    @staticmethod
    def position_actuators():
        return [
            actuator
            for actuator in Actuators
            if actuator != Actuators.left_wheel_vel and actuator != Actuators.right_wheel_vel
        ]

