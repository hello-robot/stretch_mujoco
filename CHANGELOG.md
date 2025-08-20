# Changelog

The changes between releases of Stretch Mujoco are documented here.

## [0.5.0](https://github.com/hello-robot/stretch_mujoco/releases/tag/v0.5.0) - June 10, 2025
Here’s what’s new with the release:

 - Runs on Ubuntu, MacOS, and Windows
 - Simulates Stretch 3 in real-time
   - Depth and Color cameras, with calibrated intrinsics, and support for hardware rendering
   - 2D spinning lidar
   - Python & ROS2 library, with position & velocity control
 - Can generate 100s of permutations of Robocasa kitchen-style environments
   - or add any custom environments you create
 - Can run headless, easy to use in Google Colab T4 or CPU instances
 - Supports ROS2
   - Lidar-SLAM & Nav2
   - Generates point clouds
   - Tested with Stretch Web Teleop
 - Includes tutorials & docs
   - Examples for keyboard teleop, lidar scanning, camera feeds, etc.
 - Advanced API for deformables, procedural model generation, SDF collisions, cloth simulation, and more

This release focused on bringing the behavior between the simulated and real robots closer, so that software written against simulation can translate to a real Stretch. We characterized and matched camera intrinsics, depth range, laser-scan ordering, motion quality, etc. and tested against existing applications like Stretch Web Teleop to compare against expected behavior. We hope this release makes writing new code, testing it against a plethora of environments, and deploying on the robot easier!
