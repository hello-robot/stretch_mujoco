import copy
import importlib.resources as importlib_resources

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

    def get_transform(self, link1: str, link2: str, cfg: dict):
        self.set_config(cfg)
        return self.urdf.get_transform(link1, link2)


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
    T = robot.get_transform("base_link", "link_grasp_center", cfg)
    print(T)
