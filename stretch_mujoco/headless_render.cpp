#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

// Include EGL headers
#include <EGL/egl.h>
#include <EGL/eglext.h>
// Use desktop OpenGL header rather than GLES2 for full OpenGL support.
#include <GL/gl.h>

// Include MuJoCo headers (ensure the include path is correct for your installation)
#include <mujoco/mujoco.h>

//------------------------------------------------------------------------------
// Initialize an offscreen EGL context with a pbuffer surface.
//------------------------------------------------------------------------------
EGLDisplay initEGL(EGLContext &eglContext, EGLSurface &eglSurface, int width, int height) {    
    // 1. Get the default display.
    EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDisplay == EGL_NO_DISPLAY) {
        std::cerr << "Unable to get EGL display" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 2. Initialize EGL.
    EGLint major, minor;
    if (!eglInitialize(eglDisplay, &major, &minor)) {
        std::cerr << "Unable to initialize EGL" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "EGL initialized, version " << major << "." << minor << std::endl;

    const GLubyte* version = glGetString(GL_VERSION);
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* extensions = glGetString(GL_EXTENSIONS);
    std::cout << "OpenGL Version: " << version << std::endl;
    std::cout << "Renderer: " << renderer << std::endl;
    // Optionally, search the extensions string for "GL_ARB_framebuffer_object"
    if (extensions && strstr(reinterpret_cast<const char*>(extensions), "GL_ARB_framebuffer_object"))
        std::cout << "GL_ARB_framebuffer_object supported" << std::endl;
    else
        std::cout << "GL_ARB_framebuffer_object NOT supported" << std::endl;

    // 3. Choose an appropriate EGL config.
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,  // Request desktop OpenGL.
        EGL_RED_SIZE,        8,
        EGL_GREEN_SIZE,      8,
        EGL_BLUE_SIZE,       8,
        EGL_ALPHA_SIZE,      8,
        EGL_DEPTH_SIZE,      24,
        EGL_NONE
    };
    EGLConfig eglConfig;
    EGLint numConfigs;
    if (!eglChooseConfig(eglDisplay, configAttribs, &eglConfig, 1, &numConfigs) || numConfigs < 1) {
        std::cerr << "No matching EGL configs found" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 4. Create a pbuffer (offscreen) surface.
    EGLint pbufferAttribs[] = {
        EGL_WIDTH,  width,
        EGL_HEIGHT, height,
        EGL_NONE,
    };
    eglSurface = eglCreatePbufferSurface(eglDisplay, eglConfig, pbufferAttribs);
    if (eglSurface == EGL_NO_SURFACE) {
        std::cerr << "Failed to create EGL surface" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 5. Bind the OpenGL API.
    if (!eglBindAPI(EGL_OPENGL_API)) {
        std::cerr << "Failed to bind OpenGL API" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 6. Create an EGL rendering context with a core profile.
    EGLint contextAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,    // Request OpenGL 3.x
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    eglContext = eglCreateContext(eglDisplay, eglConfig, EGL_NO_CONTEXT, contextAttribs);
    if (eglContext == EGL_NO_CONTEXT) {
        std::cerr << "Failed to create EGL context" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 7. Make the context current.
    if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
        std::cerr << "Failed to make EGL context current" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return eglDisplay;
}

//------------------------------------------------------------------------------
// Main: load MuJoCo model, create offscreen EGL+MuJoCo contexts, and render each
// fixed camera defined in the model.
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Define offscreen render dimensions.
    const int width  = 800;
    const int height = 600;
    EGLContext eglContext;
    EGLSurface eglSurface;
    EGLDisplay eglDisplay = initEGL(eglContext, eglSurface, width, height);

    // Load MuJoCo model (ensure "./models/scene.xml" is a valid model file path)
    const char* modelPath = "./models/scene.xml";
    char error[1000] = "Could not load model";
    mjModel* m = mj_loadXML(modelPath, nullptr, error, 1000);
    if (!m) {
        std::cerr << "Error loading model: " << error << std::endl;
        return 1;
    }
    mjData* d = mj_makeData(m);

    // Create MuJoCo visualization objects.
    mjvCamera cam;
    mjv_defaultCamera(&cam);
    // Use fixed-camera mode; we will change fixedcamid for each camera.
    cam.type = mjCAMERA_FIXED;

    mjvOption vopt;
    mjv_defaultOption(&vopt);

    mjvPerturb perturb;
    mjv_defaultPerturb(&perturb);

    mjvScene scn;
    mjv_defaultScene(&scn);
    mjv_makeScene(m, &scn, 1000);

    // Create MuJoCo offscreen rendering context.
    mjrContext con;
    mjr_defaultContext(&con);
    mjr_makeContext(m, &con, 50);

    // Loop over each camera defined in the model and render its view.
    for (int camid = 0; camid < m->ncam; camid++) {
        std::cout << "Rendering view from camera " << camid << std::endl;

        // Set the fixed camera ID (MuJoCo uses the fixed camera id to select a model-defined camera)
        cam.fixedcamid = camid;

        // Update the scene with the current camera view.
        mjv_updateScene(m, d, &vopt, &perturb, &cam, mjCAT_ALL, &scn);

        // Define the viewport for rendering.
        mjrRect viewport = {0, 0, width, height};

        // Render the scene offscreen.
        mjr_render(viewport, &scn, &con);

        // Read back the rendered pixels from the framebuffer.
        std::vector<unsigned char> pixels(width * height * 3);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        std::cout << "Camera " << camid << " rendered; image buffer has " 
                  << pixels.size() << " bytes." << std::endl;
    }

    // Clean up MuJoCo and EGL objects.
    mjr_freeContext(&con);
    mjv_freeScene(&scn);
    mj_deleteData(d);
    mj_deleteModel(m);

    eglDestroyContext(eglDisplay, eglContext);
    eglDestroySurface(eglDisplay, eglSurface);
    eglTerminate(eglDisplay);

    return 0;
}
