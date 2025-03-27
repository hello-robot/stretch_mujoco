from enum import Enum

from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints


class Actuators(Enum):
    """
    An enum for the joints defined in the MJCF (stretch.xml).
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
    
    @classmethod
    def get_arm_joints(cls) -> list["Actuators"]:
        return [
            cls.arm,
            cls.wrist_pitch,
            cls.wrist_roll,
            cls.wrist_yaw,
        ]
    
    @classmethod
    def get_actuated_joints(cls) -> list["Actuators"]:
        return [actuator for actuator in cls if actuator != cls.base_rotate and actuator!= cls.base_translate]
    
    def get_joint_names_in_mjcf(self):
        """
        An actuator may have multiple joints. Return their names here.
        """
        if self == Actuators.left_wheel_vel: return [ "joint_left_wheel"]
        if self == Actuators.right_wheel_vel: return  ["joint_right_wheel"]
        if self == Actuators.lift: return  ["joint_lift"]
        if self == Actuators.arm: return ['joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3']
        if self == Actuators.wrist_yaw: return  ["joint_wrist_yaw"]
        if self == Actuators.wrist_pitch: return [ "joint_wrist_pitch"]
        if self == Actuators.wrist_roll: return  ["joint_wrist_roll"]
        if self == Actuators.gripper: return  ["joint_gripper_slide"]
        if self == Actuators.head_pan: return [ "joint_head_pan"]
        if self == Actuators.head_tilt: return  ["joint_head_tilt"]

        raise NotImplementedError(f"Joint names for {self} are not defined.")

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
