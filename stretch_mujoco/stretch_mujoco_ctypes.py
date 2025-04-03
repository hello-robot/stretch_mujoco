import ctypes
import os

import stretch_mujoco.utils as utils

# Adjust these names and paths for your environment:
LIB_NAME = utils.lib_path  # or "libstretch_mujoco.dylib" / ".dll"

if not os.path.isfile(LIB_NAME):
    raise OSError(f"Cannot find shared library {LIB_NAME}")

lib = ctypes.CDLL(LIB_NAME)

# -------------------  Define C structs in Python  -------------------
# We'll replicate the C structs for Command, JointStates, CameraFrame

class Command(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),      # CommandType
        ("actuator", ctypes.c_int),  # ActuatorID
        ("value", ctypes.c_double),
        ("value2", ctypes.c_double),
        ("trigger", ctypes.c_bool),
        ("keyframe_name", ctypes.c_char * 32)
    ]

class JointStates(ctypes.Structure):
    _fields_ = [
        ("time", ctypes.c_double),
        ("lift_pos", ctypes.c_double),
        ("lift_vel", ctypes.c_double),
        ("arm_pos", ctypes.c_double),
        ("arm_vel", ctypes.c_double),
        ("head_pan_pos", ctypes.c_double),
        ("head_pan_vel", ctypes.c_double),
        ("head_tilt_pos", ctypes.c_double),
        ("head_tilt_vel", ctypes.c_double),
        ("wrist_yaw_pos", ctypes.c_double),
        ("wrist_yaw_vel", ctypes.c_double),
        ("wrist_pitch_pos", ctypes.c_double),
        ("wrist_pitch_vel", ctypes.c_double),
        ("wrist_roll_pos", ctypes.c_double),
        ("wrist_roll_vel", ctypes.c_double),
        ("gripper_pos", ctypes.c_double),
        ("gripper_vel", ctypes.c_double),
        ("base_x", ctypes.c_double),
        ("base_y", ctypes.c_double),
        ("base_theta", ctypes.c_double),
        ("base_linear_vel", ctypes.c_double),
        ("base_angular_vel", ctypes.c_double),
    ]

class CameraFrame(ctypes.Structure):
    _fields_ = [
        ("time", ctypes.c_double),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        # We have a 320x240 RGBA => 320*240*4 = 307,200
        ("pixels", ctypes.c_ubyte * (320*240*4)),
        ("valid", ctypes.c_bool),
    ]

# Enum definitions, must match C++ CommandType / ActuatorID
CMD_NONE       = 0
CMD_MOVE_TO    = 1
CMD_MOVE_BY    = 2
CMD_SET_BASE_VEL = 3
CMD_KEYFRAME   = 4

ACT_NONE           = 0
ACT_LIFT           = 1
ACT_ARM            = 2
ACT_HEAD_PAN       = 3
ACT_HEAD_TILT      = 4
ACT_WRIST_YAW      = 5
ACT_WRIST_PITCH    = 6
ACT_WRIST_ROLL     = 7
ACT_GRIPPER        = 8
ACT_LEFT_WHEEL_VEL = 9
ACT_RIGHT_WHEEL_VEL= 10
ACT_BASE_TRANSLATE = 11
ACT_BASE_ROTATE    = 12

# -------------------  ctypes function signatures  -------------------

# void mj_sim_initialize(const char* mujoco_key_path, const char* xml_path);
lib.mj_sim_initialize.argtypes = [ctypes.c_char_p]
lib.mj_sim_initialize.restype  = None

def mj_sim_initialize(scene_xml_path: str):
    lib.mj_sim_initialize(scene_xml_path.encode('utf-8'))

# void mj_sim_shutdown();
lib.mj_sim_shutdown.argtypes = []
lib.mj_sim_shutdown.restype  = None

def mj_sim_shutdown():
    lib.mj_sim_shutdown()

# void mj_sim_send_command(const Command* cmd);
lib.mj_sim_send_command.argtypes = [ctypes.POINTER(Command)]
lib.mj_sim_send_command.restype  = None

def mj_sim_send_command(cmd: Command):
    lib.mj_sim_send_command(ctypes.byref(cmd))

# bool mj_sim_get_joint_states(JointStates* out_states);
lib.mj_sim_get_joint_states.argtypes = [ctypes.POINTER(JointStates)]
lib.mj_sim_get_joint_states.restype  = ctypes.c_bool

def mj_sim_get_joint_states() -> JointStates:
    js = JointStates()
    success = lib.mj_sim_get_joint_states(ctypes.byref(js))
    if not success:
        return None
    return js

# bool mj_sim_get_camera_frame(CameraFrame* out_frame);
lib.mj_sim_get_camera_frame.argtypes = [ctypes.POINTER(CameraFrame)]
lib.mj_sim_get_camera_frame.restype  = ctypes.c_bool

def mj_sim_get_camera_frame() -> CameraFrame:
    cf = CameraFrame()
    success = lib.mj_sim_get_camera_frame(ctypes.byref(cf))
    if not success:
        return None
    return cf
