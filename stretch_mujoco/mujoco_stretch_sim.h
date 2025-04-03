#pragma once

#include <cstdint>
#include <array>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Enums / Constants
// ---------------------------------------------------------------------------

enum CommandType {
    CMD_NONE = 0,
    CMD_MOVE_TO = 1,
    CMD_MOVE_BY = 2,
    CMD_SET_BASE_VEL = 3,
    CMD_KEYFRAME = 4,
};

// Each joint or actuator we might command, matching the names in your .xml:
enum ActuatorID {
    ACT_NONE = 0,
    ACT_LIFT,
    ACT_ARM,
    ACT_HEAD_PAN,
    ACT_HEAD_TILT,
    ACT_WRIST_YAW,
    ACT_WRIST_PITCH,
    ACT_WRIST_ROLL,
    ACT_GRIPPER,
    ACT_LEFT_WHEEL_VEL,
    ACT_RIGHT_WHEEL_VEL,
    ACT_BASE_TRANSLATE,  // if you want a "move-by" for base
    ACT_BASE_ROTATE,
};

// ---------------------------------------------------------------------------
// Command struct: put everything needed for a single command
// ---------------------------------------------------------------------------
struct Command {
    CommandType type;      // move_to, move_by, etc.
    ActuatorID actuator;   // which joint to manipulate
    double value;          // could be "pos" or "delta" or "omega"
    double value2;         // sometimes used for second param (e.g. set_base_vel with linear & angular)
    bool trigger;          // if this is actually a triggered command
    char keyframe_name[32];// if type == CMD_KEYFRAME, store e.g. "home" or "stow"
};

// ---------------------------------------------------------------------------
// JointStates struct: published each step
// We replicate your Joints: lift, arm, head_pan, head_tilt, wrist_yaw, etc.
// plus base pose (x, y, theta).
// ---------------------------------------------------------------------------
struct JointStates {
    double time;
    double lift_pos, lift_vel;
    double arm_pos, arm_vel;
    double head_pan_pos, head_pan_vel;
    double head_tilt_pos, head_tilt_vel;
    double wrist_yaw_pos, wrist_yaw_vel;
    double wrist_pitch_pos, wrist_pitch_vel;
    double wrist_roll_pos, wrist_roll_vel;
    double gripper_pos, gripper_vel;
    double base_x, base_y, base_theta;
    double base_linear_vel, base_angular_vel;
};

// ---------------------------------------------------------------------------
// CameraFrame struct: an example offscreen render (RGB). For demonstration,
// we do a small resolution. For real usage, adjust as needed.
// ---------------------------------------------------------------------------
struct CameraFrame {
    double time;
    int width;
    int height;
    // We store raw uncompressed RGBA or RGB
    // For demonstration, let's do RGBA. Each pixel 4 bytes. 
    // size = width * height * 4 
    // We'll just store a fixed max of e.g. 320x240 for demonstration.
    std::array<unsigned char, 320*240*4> pixels;
    bool valid; // if we actually rendered
};

// ---------------------------------------------------------------------------
// extern "C" functions that we'll expose via ctypes
// ---------------------------------------------------------------------------

// 1) Start the MuJoCo simulation thread, load model from "scene.xml"
void mj_sim_initialize(const char* scene_xml_path);

// 2) Cleanly end the simulation
void mj_sim_shutdown();

// 3) Send a command (non-blocking)
void mj_sim_send_command(const Command* cmd);

// 4) Get the latest joint states (blocking). 
//    We'll always give you the freshest item in the queue if multiple are available.
bool mj_sim_get_joint_states(JointStates* out_states);

// 5) Get the latest camera frame (blocking).
bool mj_sim_get_camera_frame(CameraFrame* out_frame);

#ifdef __cplusplus
}
#endif
