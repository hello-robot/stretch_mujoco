from typing import Any, Dict, Optional, Union, override

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco._structs import MjModel
from mujoco_playground._src import mjx_env

from stretch_mujoco.enums.actuators import Actuators


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.04,
  )
class StretchMjxBase(mjx_env.MjxEnv):
  """
  Base environment for Hello Robot Stretch.
  
  References https://github.com/google-deepmind/mujoco_playground/tree/main/mujoco_playground/_src/manipulation/franka_emika_panda 
  """

  def __init__(
      self,
      xml_path: epath.Path,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    self._xml_path = xml_path.as_posix()
    mj_model = MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = self.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model)
    self._action_scale = config.action_scale

  def _post_init(self, obj_name: str, keyframe: str):
    all_joints = [act.name for act in Actuators]
    self._robot_arm_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j.name).id]
        for j in Actuators.get_arm_joints()
    ])
    self._robot_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in all_joints
    ])
    self._gripper_site = self._mj_model.body("link_grasp_center").id
    self._left_finger_geom = self._mj_model.body("link_gripper_finger_left").id
    self._right_finger_geom = self._mj_model.body("link_gripper_finger_right").id
    self._hand_geom = self._mj_model.body("link_gripper_slider").id
    self._obj_body = self._mj_model.body(obj_name).id
    self._obj_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body(obj_name).jntadr[0]
    ]
    self._mocap_target = self._mj_model.body("mocap_target1").mocapid
    self._floor_geom = self._mj_model.geom("floor").id
    self._init_q = self._mj_model.keyframe(keyframe).qpos
    self._init_obj_pos = jp.array(
        self._init_q[self._obj_qposadr : self._obj_qposadr + 3],
        dtype=jp.float32,
    )
    self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

  @property
  @override
  def xml_path(self) -> str:
    raise self._xml_path #type: ignore

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
