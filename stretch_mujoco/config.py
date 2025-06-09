robot_settings = {
    "wheel_diameter": 0.1016,
    "wheel_separation": 0.3153,
    "gripper_min_max": (-0.376, 0.56),
    "sim_gripper_min_max": (-0.02, 0.04),
}

depth_limits = {"d405": 1, "d435i": 10}


base_motion = {"timeout": 15, "default_x_vel": 0.3, "default_r_vel": 1.0}

# TODO: Add params to tune joints response motion profiles
