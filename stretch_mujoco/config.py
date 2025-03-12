from enum import Enum


camera_settings = {
    "d405_rgb": {"fovy": 50, "width": 640, "height": 480},
    "d405_depth": {"fovy": 50, "width": 640, "height": 480},
    "d435i_camera_rgb": {"fovy": 62, "width": 640, "height": 480},
    "d435i_camera_depth": {"fovy": 62, "width": 640, "height": 480},
}


robot_settings = {
    "wheel_diameter": 0.1016,
    "wheel_separation": 0.3153,
    "gripper_min_max": (-0.376, 0.56),
    "sim_gripper_min_max": (-0.02, 0.04),
}

depth_limits = {"d405": 1, "d435i": 10}


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


base_motion = {"timeout": 15, "default_x_vel": 0.3, "default_r_vel": 1.0}

# TODO: Add params to tune joints response motion profiles
