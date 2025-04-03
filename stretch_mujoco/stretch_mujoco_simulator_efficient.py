import time
import signal
import atexit
import threading
import numpy as np
import click

import stretch_mujoco.stretch_mujoco_ctypes as simlib
import stretch_mujoco.utils as utils

# We'll re-import the Command struct and constants
from stretch_mujoco.stretch_mujoco_ctypes import (
    Command, JointStates, CameraFrame,
    CMD_MOVE_TO, CMD_MOVE_BY, CMD_SET_BASE_VEL, CMD_KEYFRAME,
    ACT_LIFT, ACT_ARM, ACT_HEAD_PAN, ACT_HEAD_TILT, ACT_WRIST_YAW,
    ACT_WRIST_PITCH, ACT_WRIST_ROLL, ACT_GRIPPER,
    ACT_LEFT_WHEEL_VEL, ACT_RIGHT_WHEEL_VEL,
)

class StretchMujocoSimulatorEfficient:
    def __init__(self, scene_xml_path: str | None = None):
        if scene_xml_path is None:
            self.scene_xml_path = utils.default_scene_xml_path
        self._is_running = False
        self.is_stop_called = False

    def start(self):
        simlib.mj_sim_initialize(self.scene_xml_path)
        self._is_running = True

        # Hook signals
        signal.signal(signal.SIGTERM, lambda a,b: self.stop())
        signal.signal(signal.SIGINT, lambda a,b: self.stop())
        atexit.register(self.stop)

        click.secho("Stretch Mujoco Simulator started in C++ thread.", fg="green")

    def stop(self):
        if not self._is_running:
            return
        if self.is_stop_called:
            return
        self.is_stop_called = True

        click.secho("Stopping Stretch Mujoco Simulator...", fg="red")
        simlib.mj_sim_shutdown()
        self._is_running = False
        click.secho("Done.", fg="red")

    def is_running(self):
        return self._is_running

    # ------------- send commands -------------
    def move_to(self, actuator_id: int, pos: float):
        cmd = Command()
        cmd.type = CMD_MOVE_TO
        cmd.actuator = actuator_id
        cmd.value = pos
        cmd.trigger = True
        simlib.mj_sim_send_command(cmd)

    def move_by(self, actuator_id: int, delta: float):
        cmd = Command()
        cmd.type = CMD_MOVE_BY
        cmd.actuator = actuator_id
        cmd.value = delta
        cmd.trigger = True
        simlib.mj_sim_send_command(cmd)

    def set_base_velocity(self, v_linear: float, omega: float):
        cmd = Command()
        cmd.type = CMD_SET_BASE_VEL
        cmd.actuator = 0  # not used
        cmd.value = v_linear
        cmd.value2 = omega
        cmd.trigger = True
        simlib.mj_sim_send_command(cmd)

    def keyframe(self, frame_name: str):
        cmd = Command()
        cmd.type = CMD_KEYFRAME
        cmd.actuator = 0
        cmd.trigger = True
        cmd.keyframe_name = frame_name.encode('utf-8')[:31]  # store up to 31 chars
        simlib.mj_sim_send_command(cmd)

    # ------------- read data -------------
    def pull_status(self) -> dict:
        """
        Return a dict with joint positions, base pose, etc.
        """
        js = simlib.mj_sim_get_joint_states()
        if not js:
            return {}
        return {
            "time": js.time,
            "lift_pos": js.lift_pos,
            "lift_vel": js.lift_vel,
            "arm_pos": js.arm_pos,
            "arm_vel": js.arm_vel,
            "head_pan_pos": js.head_pan_pos,
            "head_pan_vel": js.head_pan_vel,
            "head_tilt_pos": js.head_tilt_pos,
            "head_tilt_vel": js.head_tilt_vel,
            "wrist_yaw_pos": js.wrist_yaw_pos,
            "wrist_yaw_vel": js.wrist_yaw_vel,
            "wrist_pitch_pos": js.wrist_pitch_pos,
            "wrist_pitch_vel": js.wrist_pitch_vel,
            "wrist_roll_pos": js.wrist_roll_pos,
            "wrist_roll_vel": js.wrist_roll_vel,
            "gripper_pos": js.gripper_pos,
            "gripper_vel": js.gripper_vel,
            "base_x": js.base_x,
            "base_y": js.base_y,
            "base_theta": js.base_theta,
            "base_x_vel": js.base_linear_vel,
            "base_theta_vel": js.base_angular_vel,
        }

    def pull_camera_data(self) -> np.ndarray:
        """
        Get the latest 320x240 RGBA camera frame as a numpy array shape=(240,320,4).
        """
        cf = simlib.mj_sim_get_camera_frame()
        if not cf:
            return None
        if not cf.valid:
            return None
        w, h = cf.width, cf.height
        # Convert from the flat pixel array to an H,W,4 shape
        data = np.frombuffer(cf.pixels, dtype=np.uint8)
        data = data.reshape((h, w, 4))
        return data

    # Some convenience methods:
    def home(self):
        self.keyframe("home")

    def stow(self):
        self.keyframe("stow")
