import math
import re
import xml.etree.ElementTree as ET
from typing import Tuple

import cv2
import numpy as np
import importlib.resources
import urchin as urdf_loader

models_path = str(importlib.resources.files('stretch_mujoco') / 'models')
default_scene_xml_path = models_path + "/scene.xml"
default_robot_xml_path = models_path + "/stretch.xml"

pkg_path = str(importlib.resources.files("stretch_urdf"))
model_name = "SE3"  # RE1V0, RE2V0, SE3
tool_name = "eoa_wrist_dw3_tool_sg3"  # eoa_wrist_dw3_tool_sg3, tool_stretch_gripper, etc
urdf_file_path = pkg_path + f"/{model_name}/stretch_description_{model_name}_{tool_name}.urdf"
mesh_files_directory_path = pkg_path + f"/{model_name}/meshes"


def compute_K(fovy: float, width: int, height: int) -> np.ndarray:
    """
    Compute camera intrinsic matrix
    """
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    return np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))


def Rz(theta):
    """
    Rotation matrix about z-axis
    """
    return np.matrix(
        [[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
    )


def limit_depth_distance(depth_image_meters: np.ndarray, max_depth: float) -> np.ndarray:
    """
    Limit depth distance
    """
    return np.where(depth_image_meters > max_depth, 0, depth_image_meters)


class URDFmodel:
    def __init__(self) -> None:
        """
        Load URDF model
        """
        self.urdf = urdf_loader.URDF.load(urdf_file_path, lazy_load_meshes=True)
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

    def get_transform(self, cfg: dict, link_name: str) -> np.ndarray:
        """
        Get transformation matrix of the link w.r.t. the base_link
        """
        lk_cfg = {
            "joint_wrist_yaw": cfg["wrist_yaw"],
            "joint_wrist_pitch": cfg["wrist_pitch"],
            "joint_wrist_roll": cfg["wrist_roll"],
            "joint_lift": cfg["lift"],
            "joint_arm_l0": cfg["arm"] / 4,
            "joint_arm_l1": cfg["arm"] / 4,
            "joint_arm_l2": cfg["arm"] / 4,
            "joint_arm_l3": cfg["arm"] / 4,
            "joint_head_pan": cfg["head_pan"],
            "joint_head_tilt": cfg["head_tilt"],
        }
        if "gripper" in cfg.keys():
            lk_cfg["joint_gripper_finger_left"] = cfg["gripper"]
            lk_cfg["joint_gripper_finger_right"] = cfg["gripper"]
        return self.urdf.link_fk(lk_cfg, link=link_name)


def replace_xml_tag_value(xml_str: str, tag: str, attribute: str, pattern: str, value: str) -> str:
    """
    Replace value of a specific tag in an XML string
    Args:
        xml_str: XML string
        tag: Tag name
        attribute: Attribute name
        pattern: Pattern to match
        value: Value to replace with
    Returns:
        str: Modified XML string
    """
    root = ET.fromstring(xml_str)
    tree = ET.ElementTree(root)
    for elem in tree.iter(tag):
        if attribute in elem.attrib.keys():
            if pattern == elem.attrib[attribute]:
                elem.attrib[attribute] = value
    return ET.tostring(root, encoding="unicode")


def xml_remove_subelement(xml_str: str, subelement: str) -> str:
    """
    Remove actuator subelement from MuJoCo XML string
    Args:
        xml_str: MuJoCo XML string
    Returns:
        str: Modified MuJoCo XML string
    """
    root = ET.fromstring(xml_str)
    tree = ET.ElementTree(root)
    for elem in tree.iter(subelement):
        root.remove(elem)
    return ET.tostring(root, encoding="unicode")


def xml_remove_tag_by_name(xml_string: str, tag: str, name: str) -> Tuple[str, dict]:
    """
    Remove a subelement from an XML string with a specified tag and name attribute
    """
    # Parse the XML string into an ElementTree
    root = ET.fromstring(xml_string)

    # Find the parent element of the subelement to be removed
    parent_map = {c: p for p in root.iter() for c in p}

    removed_body_attrib = None
    # Iterate through the subelements to find the one with the specified tag and name attribute
    for elem in root.iter(tag):
        if elem.get("name") == name:
            # Remove the matching subelement
            removed_body_attrib = elem.attrib
            parent = parent_map[elem]
            parent.remove(elem)
            break

    # Convert the ElementTree back to a string
    return ET.tostring(root, encoding="unicode"), removed_body_attrib


def xml_modify_body_pos(
    xml_string: str, tag: str, name: str, pos: np.ndarray, quat: np.ndarray
) -> str:
    """
    Modify the position attribute of a tag with a specified name
    """
    # Parse the XML string into an ElementTree
    root = ET.fromstring(xml_string)
    # Find the element with body tag and name attribute
    for elem in root.iter(tag):
        if elem.get("name") == name:
            elem.set("pos", " ".join(map(str, pos)))
            elem.set("quat", " ".join(map(str, quat)))
    return ET.tostring(root, encoding="unicode")


def insert_line_after_mujoco_tag(xml_string: str, line_to_insert: str) -> str:
    """
    Insert a new line after the mujoco tag in the XML string
    """
    # Define the pattern to match the mujoco tag
    pattern = r'(<mujoco\s+model="base"\s*>)'

    # Use re.sub to insert the new line after the matched tag
    modified_xml = re.sub(pattern, f"\\1\n    {line_to_insert}", xml_string, count=1)

    return modified_xml


def get_absolute_path_stretch_xml(robot_pose_attrib: dict = None) -> str:
    """
    Generates Robot XML with absolute path to mesh files
    Args:
        robot_pose_attrib: Robot pose attributes in form {"pos": "x y z", "quat": "x y z w"}
    Returns:
        str: Path to the generated XML file
    """
    print("DEFAULT XML: {}".format(default_robot_xml_path))

    with open(default_robot_xml_path, "r") as f:
        default_robot_xml = f.read()

    default_robot_xml = re.sub(
        'assetdir="assets"', f'assetdir="{models_path + "/assets"}"', default_robot_xml
    )

    # find all the line which has the pattrn {file="something.type"}
    # and replace the file path with the absolute path
    pattern = r'file="(.+?)"'
    for match in re.finditer(pattern, default_robot_xml):
        file_path = match.group(1)
        default_robot_xml = default_robot_xml.replace(
            file_path, models_path + "/assets/" + file_path
        )

    if robot_pose_attrib is not None:
        pos = f'pos="{robot_pose_attrib["pos"]}" quat="{robot_pose_attrib["quat"]}"'
        default_robot_xml = re.sub(
            '<body name="base_link" childclass="stretch">',
            f'<body name="base_link" childclass="stretch" {pos}>',
            default_robot_xml,
        )

    # Absosolute path converted streth xml
    with open(models_path + "/stretch_temp_abs.xml", "w") as f:
        f.write(default_robot_xml)
    print("Saving temp abs path xml: {}".format(models_path + "/stretch_temp_abs.xml"))
    return models_path + "/stretch_temp_abs.xml"


def map_between_ranges(
    value: float, from_min_max: Tuple[float, float], to_min_max: Tuple[float, float]
) -> float:
    """
    Map a value from one range to another
    """
    return (value - from_min_max[0]) * (to_min_max[1] - to_min_max[0]) / (
        from_min_max[1] - from_min_max[0]
    ) + to_min_max[0]


def get_depth_color_map(depth_image, clor_map=cv2.COLORMAP_JET):
    """
    Get depth color map
    """
    depth_min = np.min(depth_image)
    depth_max = np.max(depth_image)

    normalized_depth = (depth_image - depth_min) / (depth_max - depth_min)
    depth_8bit = ((1 - normalized_depth) * 255).astype(np.uint8)
    depth_8bit = cv2.applyColorMap(depth_8bit, clor_map)
    return depth_8bit
