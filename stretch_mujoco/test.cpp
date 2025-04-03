#include <iostream>
#include <string>
#include <chrono>
#include <mujoco/mujoco.h>

// MuJoCo global variables
mjModel* m = nullptr;
mjData* d = nullptr;
mjrContext context;
mjvScene scene;
mjvCamera camera;
mjvOption vopt;
mjvPerturb pert;

// simple controller applying damping to each dof
void mycontroller(const mjModel* m, mjData* d) {
    if (m->nu == m->nv) {
        mju_scl(d->ctrl, d->qvel, -0.1, m->nv);
    }
}


int main() {
    std::cout << "mjVERSION_HEADER: " << mjVERSION_HEADER << ", mj_version(): " << mj_version() << std::endl;
    if (mjVERSION_HEADER!=mj_version()) {
        std::cerr << "ERROR: must use MuJoCo v3.3.0" << std::endl;
        return 1;
    }

    // load scene
    // char error[1000] = {0};
    // m = mj_loadXML("./models/scene.xml", nullptr, error, 1000);
    // if (!m) {
    //     std::cerr << "ERROR: MuJoCo model load error: " << error << std::endl;
    //     return 1;
    // }
    m = mj_loadModel("./models/scene.mjb", NULL);
    d = mj_makeData(m);

    // setup rendering
    mjr_defaultContext(&context);
    mjv_defaultCamera(&camera);
    mjv_defaultOption(&vopt);
    mjv_defaultPerturb(&pert);
    mjv_makeScene(m, &scene, 1000);
    mjr_makeContext(m, &context, mjFONTSCALE_100);

    // print out model info
    std::cout << "Is forward and inverse solutions enabled? " << mjENABLED(mjENBL_FWDINV) << std::endl;
    std::cout << "Is energy enabled? "<< mjENABLED(mjENBL_ENERGY) << std::endl;
    std::string integrator = (m->opt.integrator == mjINT_IMPLICITFAST) ? "mjINT_IMPLICITFAST" : "idk";
    std::cout << "Integrator: " << integrator << std::endl;
    std::cout << "Timestep: " << m->opt.timestep << "s" << std::endl;
    printf("Before maxuse_arena: %.4fMb\n", (double)d->maxuse_arena / 1000000);
    printf("\n");

    // simulate until t=10s
    std::cout << "Starting sim..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    mjcb_control = mycontroller;
    while (d->time < 10) {
        mj_step(m, d);

        // clear control input
        mju_zero(d->ctrl, m->nu);
        mju_zero(d->qfrc_applied, m->nv);
        mju_zero(d->xfrc_applied, 6*m->nbody);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double elapsed_time = elapsed.count();
    printf("Sim took %.2fs\n", elapsed_time);
    printf("After maxuse_arena: %.4fMb\n", (double)d->maxuse_arena / 1000000);

    // deallocate model/data
    mj_deleteModel(m);
    mj_deleteData(d);

    std::cout << "Exiting." << std::endl;
    return 0;
}