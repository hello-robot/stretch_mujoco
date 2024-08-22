import copy
import importlib.resources as importlib_resources
from typing import Tuple

import numpy as np
from pytransform3d.urdf import UrdfTransformManager

pkg_path = str(importlib_resources.files("stretch_urdf"))
model_name = "SE3"  # RE1V0, RE2V0, SE3
tool_name = "eoa_wrist_dw3_tool_sg3"  # eoa_wrist_dw3_tool_sg3, tool_stretch_gripper, etc
urdf_file_path = pkg_path + f"/{model_name}/stretch_description_{model_name}_{tool_name}.urdf"
mesh_files_directory_path = pkg_path + f"/{model_name}/meshes"


def load_urdf(
    urdf_file_path: str, mesh_files_directory_path: str, whitelist: list = None
) -> UrdfTransformManager:
    # read the URDF file and convert it to a string
    with open(urdf_file_path, "r") as urdf_file:
        STRETCH_URDF = urdf_file.read()

    # replace the following pattern in the URDF string
    STRETCH_URDF = STRETCH_URDF.replace("./meshes/", "")

    tm = UrdfTransformManager()
    tm.load_urdf(STRETCH_URDF, mesh_path=mesh_files_directory_path)
    return clean_urdf(tm)


def clean_urdf(tm: UrdfTransformManager) -> UrdfTransformManager:
    _tm = copy.deepcopy(tm)
    for k in tm.transforms:
        r = False
        for e in k:
            if "visual" in e or "collision" in e or "inertia" in e:
                r = True
        if r:
            _tm.remove_transform(k[0], k[1])
    tm = copy.deepcopy(_tm)
    return tm


def get_stretch_3_urdf():
    return load_urdf(urdf_file_path, mesh_files_directory_path)


def decompose_homogeneous_matrix(homogeneous_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes a 4x4 homogeneous transformation matrix into its rotation matrix and translation vector components.

    Args:
        homogeneous_matrix (numpy.ndarray): A 4x4 matrix representing a homogeneous transformation.

    Returns:
        tuple: A tuple containing:
            - rotation_matrix : A 3x3 matrix representing the rotation component.
            - translation_vector : A 1D array of length 3 representing the translation component.
    """
    if homogeneous_matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")
    rotation_matrix = homogeneous_matrix[:3, :3]
    translation_vector = homogeneous_matrix[:3, 3]
    return rotation_matrix, translation_vector


class RobotModel:
    def __init__(self) -> None:
        self.urdf = get_stretch_3_urdf()
        self.joints_names = [
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            "joint_lift",
            "joint_arm_l0",
            "joint_arm_l1",
            "joint_arm_l2",
            "joint_arm_l3",
            "joint_head_pan",
            "joint_head_tilt",
            "joint_gripper_finger_left",
        ]

    def set_config(self, cfg: dict) -> None:
        self.urdf.set_joint("joint_wrist_yaw", cfg["wrist_yaw"])
        self.urdf.set_joint("joint_wrist_pitch", cfg["wrist_pitch"])
        self.urdf.set_joint("joint_wrist_roll", cfg["wrist_roll"])
        self.urdf.set_joint("joint_lift", cfg["lift"])
        self.urdf.set_joint("joint_arm_l0", cfg["arm"] / 4)
        self.urdf.set_joint("joint_arm_l1", cfg["arm"] / 4)
        self.urdf.set_joint("joint_arm_l2", cfg["arm"] / 4)
        self.urdf.set_joint("joint_arm_l3", cfg["arm"] / 4)
        self.urdf.set_joint("joint_head_pan", cfg["head_pan"])
        self.urdf.set_joint("joint_head_tilt", cfg["head_tilt"])
        if "gripper" in cfg.keys():
            self.urdf.set_joint("joint_gripper_finger_left", cfg["gripper"])
            self.urdf.set_joint("joint_gripper_finger_right", cfg["gripper"])

    def get_transform(self, link1: str, link2: str, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transformation matrix between two links given a configuration.
        Args:
            link1 (str): The name of the first link.
            link2 (str): The name of the second link.
            cfg (dict): The configuration of the robot."""

        self.set_config(cfg)
        return decompose_homogeneous_matrix(self.urdf.get_transform(link1, link2))


if __name__ == "__main__":
    robot = RobotModel()
    cfg = {
        "wrist_yaw": 0.0,
        "wrist_pitch": 0.0,
        "wrist_roll": 0.0,
        "lift": 0.6,
        "arm": 0.0,
        "head_pan": 0.0,
        "head_tilt": 0.0,
    }
    R, T = robot.get_transform("link_grasp_center", "base_link", cfg)
    print(R)
    print(T)
