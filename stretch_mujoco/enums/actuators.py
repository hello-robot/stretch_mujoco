from enum import Enum
from functools import cache

from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints


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
    gripper_left_finger = 12
    gripper_right_finger = 13


    def get_joint_names_in_mjcf(self) -> list[str]:
        """
        An actuator may have multiple joints in the MJCF. Return their names here. Useful for querying positions from Mujoco.

        Opposite mapping to get_actuator_by_joint_names_in_mjcf()
        """
        if self == Actuators.left_wheel_vel:
            return ["joint_left_wheel"]
        if self == Actuators.right_wheel_vel:
            return ["joint_right_wheel"]
        if self == Actuators.lift:
            return ["joint_lift"]
        if self == Actuators.arm:
            return ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]
        if self == Actuators.wrist_yaw:
            return ["joint_wrist_yaw"]
        if self == Actuators.wrist_pitch:
            return ["joint_wrist_pitch"]
        if self == Actuators.wrist_roll:
            return ["joint_wrist_roll"]
        if self == Actuators.gripper:
            return ["joint_gripper_slide"]
        if self == Actuators.gripper_left_finger:
            return ["joint_gripper_finger_left_open"]
        if self == Actuators.gripper_right_finger:
            return ["joint_gripper_finger_right_open"]
        if self == Actuators.head_pan:
            return ["joint_head_pan"]
        if self == Actuators.head_tilt:
            return ["joint_head_tilt"]

        raise NotImplementedError(f"Joint names for {self} are not defined.")
    
    @staticmethod
    @cache
    def get_actuator_by_joint_names_in_mjcf(joint_name: str) -> "Actuators":
        """
        Joint names defined in the mjcf, return their Actuator here.

        Opposite mapping to get_joint_names_in_mjcf()
        """
        # Actual from MJCF:
        # joint_right_wheel: Limits = [0. 0.]
        # joint_left_wheel: Limits = [0. 0.]
        # joint_lift: Limits = [0.  1.1]
        # joint_arm_l3: Limits = [0.   0.13]
        # joint_arm_l2: Limits = [0.   0.13]
        # joint_arm_l1: Limits = [0.   0.13]
        # joint_arm_l0: Limits = [0.   0.13]
        # joint_wrist_yaw: Limits = [-1.39  4.42]
        # joint_wrist_pitch: Limits = [-1.57  0.56]
        # joint_wrist_roll: Limits = [-3.14  3.14]
        # joint_gripper_slide: Limits = [-0.02  0.04]
        # joint_gripper_finger_left_open: Limits = [-0.6  0.6]
        # rubber_left_x: Limits = [0. 0.]
        # rubber_left_y: Limits = [0. 0.]
        # joint_gripper_finger_right_open: Limits = [-0.6  0.6]
        # rubber_right_x: Limits = [0. 0.]
        # rubber_right_y: Limits = [0. 0.]
        # joint_head_pan: Limits = [-4.04  1.73]
        # joint_head_tilt: Limits = [-1.53  0.79]
        # joint_head_nav_cam: Limits = [-1.53  0.79]
        if joint_name == "joint_left_wheel":
            return Actuators.left_wheel_vel
        if joint_name == "joint_right_wheel":
            return Actuators.right_wheel_vel
        if joint_name == 'translate_mobile_base' or joint_name == 'position':
            return Actuators.base_translate
        if joint_name == 'rotate_mobile_base':
            return Actuators.base_rotate
        
        if joint_name == "joint_lift":
            return Actuators.lift
        if "joint_arm" in joint_name:
            return Actuators.arm
        if joint_name == "joint_wrist_yaw":
            return Actuators.wrist_yaw
        if joint_name == "joint_wrist_pitch":
            return Actuators.wrist_pitch
        if joint_name == "joint_wrist_roll":
            return Actuators.wrist_roll
        if joint_name == "joint_gripper_slide" or joint_name == "gripper_aperture":
            return Actuators.gripper
        if "joint_gripper_finger_left" in joint_name:
            return Actuators.gripper_left_finger
        if "joint_gripper_finger_right" in joint_name:
            return Actuators.gripper_right_finger
        if joint_name == "joint_head_pan":
            return Actuators.head_pan
        if joint_name == "joint_head_tilt":
            return Actuators.head_tilt

        raise NotImplementedError(f"Actuator for {joint_name} is not defined.")



    def _get_status_attribute(self, is_position: bool, status: StatusStretchJoints) -> float:
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
        self, is_position: bool, status: StatusStretchJoints
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

    def get_position(self, status: StatusStretchJoints) -> float:
        if self in [Actuators.base_rotate, Actuators.base_translate]:
            raise Exception(f"Please use `get_position_relative()` for {self.name}")
        return self._get_status_attribute(True, status)

    def get_position_relative(self, status: StatusStretchJoints) -> tuple[float, float, float]:
        if self not in [Actuators.base_rotate, Actuators.base_translate]:
            raise Exception(f"Please use `get_position()` for {self.name}")
        return self._get_base_status_attribute(True, status)

    def get_velocity(self, status: StatusStretchJoints) -> float:
        if self in [Actuators.base_rotate, Actuators.base_translate]:
            raise Exception(f"Please use `get_velocity_relative()` for {self.name}")
        return self._get_status_attribute(False, status)

    def get_velocity_relative(self, status: StatusStretchJoints) -> tuple[float, float, float]:
        if self not in [Actuators.base_rotate, Actuators.base_translate]:
            raise Exception(f"Please use `get_velocity()` for {self.name}")
        return self._get_base_status_attribute(False, status)
