# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of using python simulate UI with offscreen cameras"""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import cv2

RES_X = 1280
RES_Y = 720

def fixation_control(m, d, gl_ctx, scn, cam, vopt, pert, ctx, viewport):
  try:
    # Render the simulated camera
    mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    mujoco.mjr_render(viewport, scn, ctx)
    image = np.empty((RES_Y, RES_X, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(image, None, viewport, ctx)
    image = cv2.flip(image, 0) # OpenGL renders with inverted y axis

    # Show the simulated camera image
    cv2.imshow('fixation', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    # Threshold image and use median of detection pixels as center of target
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    thresholded_image = (  (image_hsv[:, :, 0] > 115) & (image_hsv[:, :, 0] <= 130)
                         & (image_hsv[:, :, 1] > 150) & (image_hsv[:, :, 1] <= 255)
                         & (image_hsv[:, :, 2] > 100) & (image_hsv[:, :, 2] <= 200))
    target_detections = np.where(thresholded_image)
    x = np.median(target_detections[1])
    y = np.median(target_detections[0])

    # Distance of target center from center of image normalized
    dx = (x - RES_X / 2) / (RES_X / 2)
    dy = (y - RES_Y / 2) / (RES_Y / 2)

    # Set actuator velocities
    shoulder_v_gain = 5.0
    d.actuator('shoulderv').ctrl[0] = -shoulder_v_gain * np.arctan(dx)
    elbow_v_gain = 5.0
    d.actuator('elbowv').ctrl[0] = elbow_v_gain * -np.arctan(dy)

  except Exception as e:
    print(e)
    raise e

def load_callback(m=None, d=None):
  # Clear the control callback before loading a new model
  # or a Python exception is raised
  mujoco.set_mjcb_control(None)

  m = mujoco.MjModel.from_xml_path('./fixation.xml')
  d = mujoco.MjData(m)

  if m is not None:
    # Make the windmill spin
    d.joint('windmillrotor').qvel = 1

    # Make all the things needed to render a simulated camera
    gl_ctx = mujoco.GLContext(RES_X, RES_Y)
    gl_ctx.make_current()

    scn = mujoco.MjvScene(m, maxgeom=100)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'fixater')

    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()

    ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)

    viewport = mujoco.MjrRect(0, 0, RES_X, RES_Y)

    # Set the callback and capture all variables needed for rendering
    mujoco.set_mjcb_control(
      lambda m, d: fixation_control(
        m, d, gl_ctx, scn, cam, vopt, pert, ctx, viewport))

  return m , d

if __name__ == '__main__':
  viewer.launch(loader=load_callback)
