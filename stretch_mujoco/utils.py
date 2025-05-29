import dataclasses
import math
import re
import time
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Callable, Tuple

import cv2
import numpy as np
import importlib.resources
import urchin as urdf_loader


from functools import wraps

import mujoco
import mujoco._functions
import mujoco._callbacks
import mujoco._render
import mujoco._enums
import mujoco.viewer
import numpy as np
from mujoco._structs import MjModel
from mujoco.glfw import GLContext as GlFwContext

import stretch_mujoco.config as config

if TYPE_CHECKING:
    from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


models_path = str(importlib.resources.files("stretch_mujoco") / "models")
default_scene_xml_path = models_path + "/scene.xml"
default_robot_xml_path = models_path + "/stretch.xml"

pkg_path = str(importlib.resources.files("stretch_urdf"))
model_name = "SE3"  # RE1V0, RE2V0, SE3
tool_name = "eoa_wrist_dw3_tool_sg3"  # eoa_wrist_dw3_tool_sg3, tool_stretch_gripper, etc
urdf_file_path = pkg_path + f"/{model_name}/stretch_description_{model_name}_{tool_name}.urdf"
mesh_files_directory_path = pkg_path + f"/{model_name}/meshes"


def require_connection(function):
    """Wraps class methods that need self"""

    def wrapper_function(self:"StretchMujocoSimulator", *args, **kwargs):
        if not self.is_running():
            raise ConnectionError(
                "The Stretch Mujoco Simulator is not running. Use the start() method to start it."
            )
        return function(self, *args, **kwargs)

    return wrapper_function


def compute_K(fovy: float, width: int, height: int) -> np.ndarray:
    """
    Compute camera intrinsic matrix
    """
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    return np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))


def Rx(theta):
    """
    Rotation matrix about x-axis
    """
    return np.matrix(
        [[1,0,0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)], ]
    )
def Ry(theta):
    """
    Rotation matrix about y-axis
    """
    return np.matrix(
        [[math.cos(theta), 0, math.sin(theta)],  [0, 1, 0],[-math.sin(theta), 0, math.cos(theta)],]
    )
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


def diff_drive_fwd_kinematics(w_left: float, w_right: float) -> tuple:
    """
    Calculate the linear and angular velocity of a differential drive robot.
    """
    wheel_diameter = config.robot_settings["wheel_diameter"]
    wheel_separation = config.robot_settings["wheel_separation"]
    R = wheel_diameter / 2
    L = wheel_separation
    if R <= 0:
        raise ValueError("Radius must be greater than zero.")
    if L <= 0:
        raise ValueError("Distance between wheels must be greater than zero.")

    # Linear velocity (V) is the average of the linear velocities of the two wheels
    V = R * (w_left + w_right) / 2.0

    # Angular velocity (omega) is the difference in linear velocities divided by the distance
    # between the wheels
    omega = R * (w_right - w_left) / L

    return (V, omega)


def diff_drive_inv_kinematics(V: float, omega: float) -> tuple:
    """
    Calculate the rotational velocities of the left and right wheels for a
    differential drive robot.
    """
    wheel_diameter = config.robot_settings["wheel_diameter"]
    wheel_separation = config.robot_settings["wheel_separation"]
    R = wheel_diameter / 2
    L = wheel_separation
    if R <= 0:
        raise ValueError("Radius must be greater than zero.")
    if L <= 0:
        raise ValueError("Distance between wheels must be greater than zero.")

    # Calculate the rotational velocities of the wheels
    w_left = (V - (omega * L / 2)) / R
    w_right = (V + (omega * L / 2)) / R

    return (w_left, w_right)


class FpsCounter:
    def __init__(self):
        self._fps_counter = 0
        self._wall_time = time.perf_counter()

        self.fps = 0
        """The actual fps count"""

        self.sim_to_real_ratio:float|None = None
        """Sim time compared with real time"""

        self._last_sim_time = 0


    def tick(self, sim_time:float|None = None):
        """
        Call this during step() to update the fps counter. 

        Pass sim_time to calculate sim-to-real time.
        """
        self._fps_counter += 1

        elapsed = time.perf_counter() - self._wall_time
        # When one second has passed, count:
        if elapsed >= 1.0:
            new_wall_time = time.perf_counter()
            
            if sim_time:
                self.sim_to_real_ratio = (sim_time - self._last_sim_time)/(new_wall_time - self._wall_time)
                self._last_sim_time = sim_time

            self.fps = self._fps_counter / elapsed
            self._wall_time = new_wall_time
            self._fps_counter = 0

        
    @property
    def sim_to_real_time_ratio_msg(self): 
        if self.sim_to_real_ratio is None:
            return "sim_to_real_ratio is not set. Call `tick(sim_time=)` with the sim_time to calculate it."
        return f"Sim is running {self.sim_to_real_ratio:.3f}x as fast as realtime"

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
        return self.urdf.link_fk(lk_cfg, link=link_name)  # type: ignore


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


def xml_remove_tag_by_name(xml_string: str, tag: str, name: str) -> Tuple[str, dict | None]:
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


def get_absolute_path_stretch_xml(robot_pose_attrib: dict | None = None) -> str:
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


def dataclass_from_dict(klass, dict_data: dict):
    # references https://stackoverflow.com/a/54769644
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{f: dataclass_from_dict(fieldtypes[f], dict_data[f]) for f in dict_data})
    except:
        return dict_data  # Not a dataclass field


def block_until_check_succeeds(
    wait_timeout: float|None, check: Callable[[], bool], is_alive: Callable[[], bool]
) -> bool:
    """Blocks until the check callback succeeds"""

    if wait_timeout is None:
        while is_alive():
            if check():
                return True
        return False
    
    start_time = time.time()

    while time.time() - start_time < wait_timeout:
        if not is_alive():
            return False
        if check():
            return True

    return False


def switch_to_glfw_renderer(mjmodel: MjModel, renderer: mujoco.Renderer):
    """
    On Darwin, the default renderer in `mujoco/gl_context.py` is CGL, which is not compatible with offscreen rendering.

    This function frees the initial display context and creates a new one with GLFW.
    """
    if renderer._gl_context:
        renderer._gl_context.free()
    if renderer._mjr_context:
        renderer._mjr_context.free()

    renderer._gl_context = GlFwContext(480, 640)

    renderer._gl_context.make_current()

    renderer._mjr_context = mujoco._render.MjrContext(
        mjmodel, mujoco._enums.mjtFontScale.mjFONTSCALE_150.value
    )
    mujoco._render.mjr_setBuffer(
        mujoco._enums.mjtFramebuffer.mjFB_OFFSCREEN.value, renderer._mjr_context
    )
    renderer._mjr_context.readDepthMap = mujoco._enums.mjtDepthMap.mjDEPTH_ZEROFAR


try:
    # Only Python >12 has override.
    override = __import__("typing").override
except:  # noqa
    def override(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper