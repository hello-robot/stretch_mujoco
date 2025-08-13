# Releasing to PyPi

Usually the simulation is run by cloning the code locally, which works well if you're changing code within the repo itself. However, if you're looking to use Stretch Mujoco as a library within your own codebase, it can be convenient to be able to `pip install` the project. This guide covers how I put Stretch Mujoco on PyPi and how you can release new versions of the project. Users can install the library by adding "hello-robot-stretch-mujoco" to their project's dependencies or using:

```
pip install hello-robot-stretch-mujoco
```

## Steps

 1. Update version in pyproject.toml. Make code changes.

 1. update changelog

 1. Tag the release commit

 1. `uv build` and `uv publish`

## Using the library

 - Getting a specific version e.g. "v0.x.y"
 - Importing the modules
 - Accessing the mujoco xmls

## What won't work

 - robocasa