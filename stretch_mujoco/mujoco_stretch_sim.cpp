#include "mujoco_stretch_sim.h"

#include <mujoco/mujoco.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cstring>  // for std::strncpy

#include "readerwriterqueue/readerwriterqueue.h"

using namespace moodycamel;

// For controlling real-time stepping
static std::atomic<bool> g_exitFlag{false};

// MuJoCo global variables
static mjModel* m = nullptr;
static mjData* d = nullptr;

// Named actuator IDs from the .xml
static int act_lift = -1;
static int act_arm = -1;
static int act_head_pan = -1;
static int act_head_tilt = -1;
static int act_wrist_yaw = -1;
static int act_wrist_pitch = -1;
static int act_wrist_roll = -1;
static int act_gripper = -1;
static int act_left_wheel = -1;
static int act_right_wheel = -1;

// For base move-by threads or concurrency
static std::atomic<bool> base_pos_motion(false);

// Threads
static std::thread simulationThread;

// Queues (now blocking):
static BlockingReaderWriterQueue<Command>     cmdQueue(100);
static BlockingReaderWriterQueue<JointStates> jointQueue(100);
static BlockingReaderWriterQueue<CameraFrame> cameraQueue(20);

// A small function to find MuJoCo IDs by name:
static int findActuatorID(const mjModel* model, const char* name) {
    int id = mj_name2id(model, mjOBJ_ACTUATOR, name);
    return id;
}

// Map from one numeric range to another
static double mapBetweenRanges(double val,
                               double in_min, double in_max,
                               double out_min, double out_max) {
    double ratio = (val - in_min) / (in_max - in_min);
    return out_min + ratio * (out_max - out_min);
}

// For gripper: real range [0.0..0.1] -> sim range [0.0..0.01]
static double toSimGripperRange(double real_pos) {
    return mapBetweenRanges(real_pos, 0.0, 0.1, 0.0, 0.01);
}

// For reading out the gripper: sim range [0.0..0.01] -> real range [0.0..0.1]
static double toRealGripperRange(double sim_pos) {
    return mapBetweenRanges(sim_pos, 0.0, 0.01, 0.0, 0.1);
}

// Diff-drive inverse kinematics
static void diffDriveInvKinematics(double v_linear, double omega,
                                   double& w_left, double& w_right) {
    double track_width = 0.5; // example half meter track
    w_left  = v_linear - (omega * track_width/2.0);
    w_right = v_linear + (omega * track_width/2.0);
}

// forward kinematics
static void diffDriveFwdKinematics(double w_left, double w_right,
                                   double& v_linear_out, double& omega_out) {
    double track_width = 0.5;
    v_linear_out = 0.5 * (w_left + w_right);
    omega_out    = (w_right - w_left) / track_width;
}

// Retrieve base pose from the "base_link" body
static void getBasePose(double& x, double& y, double& theta) {
    if (!m || !d) {
        x=0; y=0; theta=0;
        return;
    }
    int base_body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
    if (base_body_id < 0) {
        x=0; y=0; theta=0;
        return;
    }
    const double* pos3 = d->xpos + 3*base_body_id;
    x = pos3[0];
    y = pos3[1];
    const double* mat9 = d->xmat + 9*base_body_id;
    theta = std::atan2(mat9[3], mat9[0]);
}

// Offscreen rendering
static mjrContext context;
static mjvScene scene;
static mjvCamera camera;
static mjvOption vopt;       // for visualization
static mjvPerturb pert;      // if you need interactions

static void initOffscreenRender() {
    mjv_defaultScene(&scene);
    mjv_makeScene(m, &scene, 1000);

    mjr_defaultContext(&context);
    mjr_makeContext(m, &context, mjFONTSCALE_150);

    // Setup camera
    mjv_defaultCamera(&camera);
    camera.type = mjCAMERA_FREE;
    camera.distance = 2.0;
    camera.azimuth = 90;
    camera.elevation = -30;

    // Visualization options
    mjv_defaultOption(&vopt);

    // Perturb
    mjv_defaultPerturb(&pert);
}

// Renders a 320x240 image
static void renderCameraFrame() {
    int W = 320;
    int H = 240;
    std::vector<unsigned char> rgb(W * H * 3);

    mjrRect viewport = {0, 0, W, H};

    // Update scene. MuJoCo 2.3.x signature:
    // mjv_updateScene(const mjModel*, mjData*, const mjvOption*, const mjvPerturb*, mjvCamera*, int catmask, mjvScene*);
    mjv_updateScene(m, d, &vopt, nullptr, &camera, mjCAT_ALL, &scene);

    // Render offscreen
    mjr_render(viewport, &scene, &context);

    // Read 3-channel RGB
    mjr_readPixels(&rgb[0], nullptr, viewport, &context);

    // Prepare the camera frame
    CameraFrame frame;
    frame.time = d->time;
    frame.width = W;
    frame.height = H;
    frame.valid = true;

    // Convert RGB -> RGBA=4 channels
    for (int i = 0; i < W*H; i++){
        frame.pixels[4*i + 0] = rgb[3*i + 0];
        frame.pixels[4*i + 1] = rgb[3*i + 1];
        frame.pixels[4*i + 2] = rgb[3*i + 2];
        frame.pixels[4*i + 3] = 255;
    }

    cameraQueue.enqueue(frame);
}

// Process commands from cmdQueue
static void processCommands() {
    Command cmd;
    // Use try_dequeue many times in a row to drain queue
    while (cmdQueue.try_dequeue(cmd)) {
        if (!cmd.trigger)
            continue;
        switch (cmd.type) {
        case CMD_MOVE_TO:
            if (cmd.actuator == ACT_GRIPPER && act_gripper >= 0) {
                d->ctrl[act_gripper] = toSimGripperRange(cmd.value);
            }
            else if (cmd.actuator == ACT_LEFT_WHEEL_VEL && act_left_wheel >= 0) {
                d->ctrl[act_left_wheel] = cmd.value;
            }
            else if (cmd.actuator == ACT_RIGHT_WHEEL_VEL && act_right_wheel >= 0) {
                d->ctrl[act_right_wheel] = cmd.value;
            }
            else {
                if (cmd.actuator == ACT_LIFT && act_lift >=0) {
                    d->ctrl[act_lift] = cmd.value;
                } else if (cmd.actuator == ACT_ARM && act_arm>=0) {
                    d->ctrl[act_arm] = cmd.value;
                } else if (cmd.actuator == ACT_HEAD_PAN && act_head_pan>=0) {
                    d->ctrl[act_head_pan] = cmd.value;
                } else if (cmd.actuator == ACT_HEAD_TILT && act_head_tilt>=0) {
                    d->ctrl[act_head_tilt] = cmd.value;
                } else if (cmd.actuator == ACT_WRIST_YAW && act_wrist_yaw>=0) {
                    d->ctrl[act_wrist_yaw] = cmd.value;
                } else if (cmd.actuator == ACT_WRIST_PITCH && act_wrist_pitch>=0) {
                    d->ctrl[act_wrist_pitch] = cmd.value;
                } else if (cmd.actuator == ACT_WRIST_ROLL && act_wrist_roll>=0) {
                    d->ctrl[act_wrist_roll] = cmd.value;
                }
            }
            break;
        case CMD_MOVE_BY: {
            if (cmd.actuator == ACT_GRIPPER && act_gripper>=0) {
                // Convert sim->real, add delta, real->sim
                double realPos = toRealGripperRange(d->ctrl[act_gripper]);
                realPos += cmd.value;
                d->ctrl[act_gripper] = toSimGripperRange(realPos);
            }
            else if (cmd.actuator == ACT_LIFT && act_lift>=0) {
                double oldVal = d->ctrl[act_lift];
                d->ctrl[act_lift] = oldVal + cmd.value;
            }
            // etc. for other actuators
        } break;
        case CMD_SET_BASE_VEL: {
            double w_left, w_right;
            diffDriveInvKinematics(cmd.value, cmd.value2, w_left, w_right);
            if (act_left_wheel >=0)  d->ctrl[act_left_wheel] = w_left;
            if (act_right_wheel>=0) d->ctrl[act_right_wheel] = w_right;
        } break;
        case CMD_KEYFRAME: {
            if (std::strlen(cmd.keyframe_name) > 0) {
                int kf_id = mj_name2id(m, mjOBJ_KEY, cmd.keyframe_name);
                if (kf_id >=0) {
                    // keyframe has .ctrl
                    std::memcpy(d->ctrl, m->key_ctrl + kf_id*m->nu, m->nu*sizeof(mjtNum));
                }
            }
        } break;
        default:
            break;
        }
    }
}

// Publish joint states
static void publishJointStates() {
    JointStates js;
    js.time = d->time;
    if (act_lift >=0) {
        js.lift_pos = d->actuator_length[act_lift];
        js.lift_vel = d->actuator_velocity[act_lift];
    }
    if (act_arm>=0) {
        js.arm_pos = d->actuator_length[act_arm];
        js.arm_vel = d->actuator_velocity[act_arm];
    }
    if (act_head_pan>=0) {
        js.head_pan_pos = d->actuator_length[act_head_pan];
        js.head_pan_vel = d->actuator_velocity[act_head_pan];
    }
    if (act_head_tilt>=0) {
        js.head_tilt_pos = d->actuator_length[act_head_tilt];
        js.head_tilt_vel = d->actuator_velocity[act_head_tilt];
    }
    if (act_wrist_yaw>=0) {
        js.wrist_yaw_pos = d->actuator_length[act_wrist_yaw];
        js.wrist_yaw_vel = d->actuator_velocity[act_wrist_yaw];
    }
    if (act_wrist_pitch>=0) {
        js.wrist_pitch_pos = d->actuator_length[act_wrist_pitch];
        js.wrist_pitch_vel = d->actuator_velocity[act_wrist_pitch];
    }
    if (act_wrist_roll>=0) {
        js.wrist_roll_pos = d->actuator_length[act_wrist_roll];
        js.wrist_roll_vel = d->actuator_velocity[act_wrist_roll];
    }
    if (act_gripper>=0) {
        double simPos = d->actuator_length[act_gripper];
        js.gripper_pos = toRealGripperRange(simPos);
        js.gripper_vel = d->actuator_velocity[act_gripper];
    }
    if (act_left_wheel>=0 && act_right_wheel>=0) {
        double wl = d->actuator_velocity[act_left_wheel];
        double wr = d->actuator_velocity[act_right_wheel];
        double vlin, vang;
        diffDriveFwdKinematics(wl, wr, vlin, vang);
        js.base_linear_vel = vlin;
        js.base_angular_vel = vang;
    }
    getBasePose(js.base_x, js.base_y, js.base_theta);

    jointQueue.enqueue(js);
}

// The main simulation loop
static void simulationLoop() {
    auto lastWallTime = std::chrono::steady_clock::now();
    double simStep = m->opt.timestep; // e.g. 0.001

    double desiredFps = 1000.0;
    double minFrameDt = 1.0 / desiredFps;
    double cameraInterval = 1.0 / 15.0;  // 15 Hz
    double lastCamT = 0.0;

    while (!g_exitFlag) {
        auto now = std::chrono::steady_clock::now();
        double wallDt = std::chrono::duration<double>(now - lastWallTime).count();
        if (wallDt < minFrameDt) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        lastWallTime = now;

        // 1) handle commands
        processCommands();

        // 2) step simulation
        int nSubsteps = static_cast<int>(wallDt / simStep);
        if (nSubsteps < 1) nSubsteps=1;
        if (nSubsteps>10) nSubsteps=10;
        for (int i=0; i<nSubsteps; i++) {
            mj_step(m, d);
        }

        // 3) publish joint states
        publishJointStates();

        // // 4) render camera ~15 Hz | TODO: re-enable this
        // double sim_time = d->time;
        // if ((sim_time - lastCamT) >= cameraInterval) {
        //     renderCameraFrame();
        //     lastCamT = sim_time;
        // }
    }

    // Clean up
    if (d) {
        mj_deleteData(d);
        d = nullptr;
    }
    if (m) {
        mj_deleteModel(m);
        m = nullptr;
    }
    mjr_freeContext(&context);
    mjv_freeScene(&scene);
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------
extern "C" {

void mj_sim_initialize(const char* scene_xml_path) {
    if (mjVERSION_HEADER!=mj_version()) {
        std::cerr << "ERROR: must use mujoco v3.3.0" << std::endl;
        return;
    }

    char error[1000] = {0};
    m = mj_loadXML(scene_xml_path, nullptr, error, 1000);
    if (!m) {
        std::cerr << "MuJoCo model load error: " << error << std::endl;
        return;
    }
    d = mj_makeData(m);

    // Find named actuators
    act_lift        = findActuatorID(m, "lift");
    act_arm         = findActuatorID(m, "arm");
    act_head_pan    = findActuatorID(m, "head_pan");
    act_head_tilt   = findActuatorID(m, "head_tilt");
    act_wrist_yaw   = findActuatorID(m, "wrist_yaw");
    act_wrist_pitch = findActuatorID(m, "wrist_pitch");
    act_wrist_roll  = findActuatorID(m, "wrist_roll");
    act_gripper     = findActuatorID(m, "gripper");
    act_left_wheel  = findActuatorID(m, "left_wheel_vel");
    act_right_wheel = findActuatorID(m, "right_wheel_vel");

    // initOffscreenRender(); // TODO: re-enable this

    g_exitFlag = false;
    simulationThread = std::thread(simulationLoop);
}

void mj_sim_shutdown() {
    g_exitFlag = true;
    if (simulationThread.joinable()) {
        simulationThread.join();
    }
    std::cout << "Shut down MuJoCo simulation.\n";
}

void mj_sim_send_command(const Command* cmd) {
    if (!cmd) return;
    cmdQueue.enqueue(*cmd);
}

bool mj_sim_get_joint_states(JointStates* out_states) {
    if (!out_states) return false;
    // We do a blocking dequeue:
    JointStates temp;
    jointQueue.wait_dequeue(temp);
    *out_states = temp;
    return true;
}

bool mj_sim_get_camera_frame(CameraFrame* out_frame) {
    if (!out_frame) return false;
    CameraFrame temp;
    cameraQueue.wait_dequeue(temp);
    *out_frame = temp;
    return true;
}

} // end extern "C"
