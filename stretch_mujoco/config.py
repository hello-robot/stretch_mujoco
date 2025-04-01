camera_settings = {
    "d405_rgb": {"fovy": 50, "width": 640, "height": 480},
    "d405_depth": {"fovy": 50, "width": 640, "height": 480},
    "d435i_camera_rgb": {"fovy": 62, "width": 640, "height": 480},
    "d435i_camera_depth": {"fovy": 62, "width": 640, "height": 480},
}


robot_settings = {
    "wheel_diameter": 0.1016,
    "wheel_separation": 0.3153,
    "gripper_min_max": (-0.376, 0.56),
    "sim_gripper_min_max": (-0.02, 0.04),
}

depth_limits = {"d405": 1, "d435i": 10}


base_motion = {"timeout": 15, "default_x_vel": 0.3, "default_r_vel": 1.0}

# TODO: Add params to tune joints response motion profiles
