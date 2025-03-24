import importlib.resources
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
from etils import epath
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State

from stretch_mujoco.mjx.stretch import StretchMjxBase


def default_config() -> config_dict.ConfigDict:
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Gripper goes to the box.
                gripper_box=4.0,
                # Box goes to the target mocap.
                box_target=8.0,
                # Do not collide the gripper with the floor.
                no_floor_collision=0.25,
                # Arm stays close to target pose.
                robot_target_qpos=0.3,
            )
        ),
    )
    return config


class StretchMjxPickCube(StretchMjxBase):
    """Bring a box to a target.

    References https://github.com/google-deepmind/mujoco_playground/tree/main/mujoco_playground/_src/manipulation/franka_emika_panda
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        sample_orientation: bool = False,
        *,
        xml_path: epath.Path|None = None,
    ):
        models_path = str(importlib.resources.files("stretch_mujoco") / "mjx" / "models")
        _xml_path = xml_path or epath.Path(models_path + "/scene-mjx.xml")
        super().__init__(
            _xml_path,
            config,
            config_overrides,
        )
        self._post_init(obj_name="object2", keyframe="home")
        self._sample_orientation = sample_orientation

    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_target = jax.random.split(rng, 3)

        # intialize box position
        box_pos = (
            jax.random.uniform(
                rng_box,
                (3,),
                minval=jp.array([-0.2, -0.2, 0.0]),
                maxval=jp.array([0.2, 0.2, 0.0]),
            )
            + self._init_obj_pos
        )

        # initialize target position
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([-0.2, -0.2, 0.2]),
                maxval=jp.array([0.2, 0.2, 0.4]),
            )
            + self._init_obj_pos
        )

        target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if self._sample_orientation:
            # sample a random direction
            rng, rng_axis, rng_theta = jax.random.split(rng, 3)
            perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
            perturb_axis = perturb_axis / math.norm(perturb_axis)
            perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
            target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

        # initialize data
        init_q = jp.array(self._init_q).at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)
        data = mjx_env.init(
            self._mjx_model,
            init_q,
            jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
        )

        # set target mocap position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
        )

        # initialize env state and info
        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }
        info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        state = State(data, obs, reward, done, metrics, info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        raw_rewards = self._get_reward(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        state.metrics.update(**raw_rewards, out_of_bounds=out_of_bounds.astype(float))

        obs = self._get_obs(data, state.info)
        state = State(data, obs, reward, done, state.metrics, state.info)

        return state

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        target_pos = info["target_pos"]
        box_pos = data.xpos[self._obj_body]
        gripper_pos = data.site_xpos[self._gripper_site]
        pos_err = jp.linalg.norm(target_pos - box_pos)
        box_mat = data.xmat[self._obj_body]
        target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
        rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

        box_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
        gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
        robot_target_qpos = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr] - self._init_q[self._robot_arm_qposadr]
            )
        )

        # Check for collisions with the floor
        hand_floor_collision = [
            collision.geoms_colliding(data, self._floor_geom, g)
            for g in [
                self._left_finger_geom,
                self._right_finger_geom,
                self._hand_geom,
            ]
        ]
        floor_collision = sum(hand_floor_collision) > 0
        no_floor_collision = float(1 - floor_collision)

        info["reached_box"] = 1.0 * jp.maximum(
            info["reached_box"],
            (jp.linalg.norm(box_pos - gripper_pos) < 0.012),
        )

        rewards = {
            "gripper_box": gripper_box,
            "box_target": box_target * info["reached_box"],
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }
        return rewards

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        gripper_pos = data.site_xpos[self._gripper_site]
        gripper_mat = data.site_xmat[self._gripper_site].ravel()
        target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
        obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
                gripper_pos,
                gripper_mat[3:],
                data.xmat[self._obj_body].ravel()[3:],
                data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
                info["target_pos"] - data.xpos[self._obj_body],
                target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
                data.ctrl - data.qpos[self._robot_qposadr[:-1]],
            ]
        )

        return obs


class StretchMjxPickCubeOrientation(StretchMjxPickCube):
    """Bring a box to a target and orientation."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        *,
        xml_path: epath.Path|None = None,
    ):
        super().__init__(config, config_overrides, sample_orientation=True, xml_path=xml_path)
