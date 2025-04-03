#include <iostream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <EGL/egl.h>
#include <GL/gl.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Mujoco model and data
mjModel* model = nullptr;
mjData* data = nullptr;

// EGL variables
EGLDisplay egl_dpy = EGL_NO_DISPLAY;
EGLContext egl_ctx = EGL_NO_CONTEXT;
EGLSurface egl_surf = EGL_NO_SURFACE;

// Mujoco visualization
mjvScene scn;
mjvCamera cam;
mjvOption opt;
mjrContext con;

// Function to initialize EGL context for NVIDIA GPU
bool initEGL() {
    // Get EGL display
    egl_dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_dpy == EGL_NO_DISPLAY) {
        std::cerr << "Error: Could not get EGL display" << std::endl;
        return false;
    }

    // Initialize EGL
    EGLint major, minor;
    if (!eglInitialize(egl_dpy, &major, &minor)) {
        std::cerr << "Error: Could not initialize EGL" << std::endl;
        return false;
    }
    std::cout << "EGL initialized with version " << major << "." << minor << std::endl;

    // Configure EGL - Using a more compatible configuration for NVIDIA GPUs
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    // Print available configurations and extensions (for debugging)
    const char* extensions = eglQueryString(egl_dpy, EGL_EXTENSIONS);
    std::cout << "EGL Extensions: " << extensions << std::endl;

    // Find matching configuration
    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(egl_dpy, config_attribs, &config, 1, &num_configs)) {
        std::cerr << "Error: Could not choose EGL config" << std::endl;
        return false;
    }
    
    if (num_configs < 1) {
        std::cerr << "Error: No matching EGL configurations found" << std::endl;
        return false;
    }
    
    std::cout << "Found " << num_configs << " matching EGL configurations" << std::endl;

    // Create offscreen surface first
    EGLint surface_attribs[] = {
        EGL_WIDTH, 640,  // Using larger dimensions
        EGL_HEIGHT, 480,
        EGL_NONE
    };
    egl_surf = eglCreatePbufferSurface(egl_dpy, config, surface_attribs);
    if (egl_surf == EGL_NO_SURFACE) {
        std::cerr << "Error: Could not create EGL surface" << std::endl;
        return false;
    }

    // Create EGL context with specific OpenGL version (3.3 core profile)
    eglBindAPI(EGL_OPENGL_API);
    
    // For NVIDIA GPUs, let's try a more compatible context configuration
    // 3.3 core profile is a good balance for compatibility and features
    EGLint context_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    
    egl_ctx = eglCreateContext(egl_dpy, config, EGL_NO_CONTEXT, context_attribs);
    if (egl_ctx == EGL_NO_CONTEXT) {
        std::cerr << "Error: Could not create EGL context with OpenGL 3.3 Core" << std::endl;
        
        // Fall back to OpenGL 3.0 compatibility profile if core profile fails
        EGLint compat_context_attribs[] = {
            EGL_CONTEXT_MAJOR_VERSION, 3,
            EGL_CONTEXT_MINOR_VERSION, 0,
            EGL_NONE
        };
        
        egl_ctx = eglCreateContext(egl_dpy, config, EGL_NO_CONTEXT, compat_context_attribs);
        if (egl_ctx == EGL_NO_CONTEXT) {
            std::cerr << "Error: Could not create fallback EGL context" << std::endl;
            return false;
        }
        std::cout << "Created OpenGL 3.0 context (fallback)" << std::endl;
    } else {
        std::cout << "Created OpenGL 3.3 Core context" << std::endl;
    }

    // Make context current
    if (!eglMakeCurrent(egl_dpy, egl_surf, egl_surf, egl_ctx)) {
        std::cerr << "Error: Could not make EGL context current" << std::endl;
        return false;
    }
    
    // Print OpenGL information
    const GLubyte* version = glGetString(GL_VERSION);
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* vendor = glGetString(GL_VENDOR);
    const GLubyte* glsl = glGetString(GL_SHADING_LANGUAGE_VERSION);
    
    std::cout << "OpenGL Version: " << (version ? (const char*)version : "Unknown") << std::endl;
    std::cout << "OpenGL Renderer: " << (renderer ? (const char*)renderer : "Unknown") << std::endl;
    std::cout << "OpenGL Vendor: " << (vendor ? (const char*)vendor : "Unknown") << std::endl;
    std::cout << "GLSL Version: " << (glsl ? (const char*)glsl : "Unknown") << std::endl;
    
    // Check for required OpenGL extensions
    const GLubyte* extensions_gl = glGetString(GL_EXTENSIONS);
    if (extensions_gl) {
        std::string ext_str((const char*)extensions_gl);
        if (ext_str.find("GL_ARB_framebuffer_object") == std::string::npos) {
            std::cerr << "Warning: GL_ARB_framebuffer_object extension not found" << std::endl;
            std::cerr << "This may cause problems with Mujoco rendering" << std::endl;
        } else {
            std::cout << "Found GL_ARB_framebuffer_object extension" << std::endl;
        }
    }

    return true;
}

// Function to clean up EGL resources
void cleanupEGL() {
    if (egl_dpy != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (egl_surf != EGL_NO_SURFACE) {
            eglDestroySurface(egl_dpy, egl_surf);
        }
        if (egl_ctx != EGL_NO_CONTEXT) {
            eglDestroyContext(egl_dpy, egl_ctx);
        }
        eglTerminate(egl_dpy);
    }
}

// Function to initialize Mujoco
bool initMujoco(const std::string& model_path) {
    // Load model
    char error[1000] = "Could not load model";
    model = mj_loadXML(model_path.c_str(), 0, error, 1000);
    if (!model) {
        std::cerr << "Error loading model: " << error << std::endl;
        return false;
    }

    // Make data
    data = mj_makeData(model);
    if (!data) {
        std::cerr << "Error making mjData" << std::endl;
        return false;
    }

    // Initialize visualization structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    
    // Initialize scene with appropriate size for the model
    mjv_defaultScene(&scn);
    mjv_makeScene(model, &scn, 2000);
    
    // Try to initialize rendering context with more verbose error handling
    mjr_defaultContext(&con);
    
    try {
        mjr_makeContext(model, &con, mjFONTSCALE_150);
    } catch (const std::exception& e) {
        std::cerr << "Exception in mjr_makeContext: " << e.what() << std::endl;
        return false;
    }
    
    // Verify that the context was created successfully
    if (con.currentBuffer != mjFB_WINDOW) {
        std::cout << "Mujoco context initialized successfully" << std::endl;
    } else {
        std::cerr << "Warning: Mujoco context may not be properly initialized" << std::endl;
    }

    return true;
}

// Function to clean up Mujoco resources
void cleanupMujoco() {
    if (model) {
        mjr_freeContext(&con);
        mjv_freeScene(&scn);
        mj_deleteData(data);
        mj_deleteModel(model);
    }
}

// Function to render from a specific camera and save the image
bool renderCamera(int cam_id, const std::string& filename) {
    if (cam_id < 0 || cam_id >= model->ncam) {
        std::cerr << "Invalid camera ID: " << cam_id << std::endl;
        return false;
    }

    // Get camera parameters from the model
    mjvCamera render_cam;
    mjv_defaultCamera(&render_cam);
    
    // Set camera parameters from the model's camera
    render_cam.type = mjCAMERA_FIXED;
    render_cam.fixedcamid = cam_id;

    // Update scene with current model state
    mjv_updateScene(model, data, &opt, nullptr, &render_cam, mjCAT_ALL, &scn);

    // Get camera resolution
    int width = model->cam_resolution[2*cam_id];
    int height = model->cam_resolution[2*cam_id+1];
    
    std::cout << "Rendering camera " << cam_id << " at resolution " << width << "x" << height << std::endl;
    
    // Create framebuffer for offscreen rendering
    mjrRect viewport = {0, 0, width, height};
    unsigned char* rgb = new unsigned char[3 * width * height];

    // Render scene to RGB array
    mjr_setBuffer(mjFB_OFFSCREEN, &con);
    
    // Check if we need to resize the framebuffer
    if (con.offWidth != width || con.offHeight != height) {
        std::cout << "Resizing offscreen buffer from " << con.offWidth << "x" << con.offHeight 
                 << " to " << width << "x" << height << std::endl;
                 
        // Free existing context and recreate with new size
        mjr_freeContext(&con);
        mjr_defaultContext(&con);
        
        // Set desired offscreen buffer size before making context
        con.offWidth = width;
        con.offHeight = height;
        
        mjr_makeContext(model, &con, mjFONTSCALE_150);
    }
    
    // Try to render
    mjr_render(viewport, &scn, &con);
    
    // Read pixels
    mjr_readPixels(rgb, nullptr, viewport, &con);

    // Save the image (using stb_image_write)
    // Note: OpenGL renders from bottom to top, we need to flip it
    unsigned char* flipped = new unsigned char[3 * width * height];
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int k = 0; k < 3; k++) {
                flipped[3*(row*width+col)+k] = rgb[3*((height-1-row)*width+col)+k];
            }
        }
    }
    
    bool success = stbi_write_png(filename.c_str(), width, height, 3, flipped, width * 3);
    
    // Clean up
    delete[] rgb;
    delete[] flipped;
    
    if (!success) {
        std::cerr << "Failed to write image: " << filename << std::endl;
        return false;
    }
    
    std::cout << "Saved camera " << cam_id << " image to: " << filename << std::endl;
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path.xml>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    // Initialize EGL
    if (!initEGL()) {
        std::cerr << "Failed to initialize EGL" << std::endl;
        cleanupEGL();
        return 1;
    }
    
    // Initialize Mujoco
    if (!initMujoco(model_path)) {
        std::cerr << "Failed to initialize Mujoco" << std::endl;
        cleanupMujoco();
        cleanupEGL();
        return 1;
    }
    
    // Reset simulation
    mj_resetData(model, data);
    
    // Forward dynamics for a few steps to settle the model
    for (int i = 0; i < 10; i++) {
        mj_step(model, data);
    }
    
    // Render each camera
    bool any_success = false;
    for (int i = 0; i < model->ncam; i++) {
        std::string filename = "camera_" + std::to_string(i) + ".png";
        if (renderCamera(i, filename)) {
            any_success = true;
        }
    }
    
    // Cleanup
    cleanupMujoco();
    cleanupEGL();
    
    if (!any_success) {
        std::cerr << "Failed to render any cameras successfully" << std::endl;
        return 1;
    }
    
    return 0;
}