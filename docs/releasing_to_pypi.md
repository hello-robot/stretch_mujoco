# Releasing to PyPi

Usually the simulation is run by cloning the code locally, which works well if you're changing code within the repo itself. However, if you're looking to use Stretch Mujoco as a library within your own codebase, it can be convenient to be able to `pip install` the project. This guide covers how you can release new versions of the project. Users can install the library by adding "hello-robot-stretch-mujoco" to their project's dependencies or using:

```
pip install hello-robot-stretch-mujoco
```

## Steps

 1. Merge your code changes. Update version in `pyproject.toml`. We're using [semantic versioning](https://semver.org/).

 1. Update the changelog to describe what's new in this version.

 1. Merge the above changes in a release commit. Tag this commit as `v<version>` (e.g. v0.5.0). Your tag should show up here: https://github.com/hello-robot/stretch_mujoco/tags

    ```
    git add .
    git commit -m "Release commit v0.5.0"
    git tag -a v0.5.0 -m "Short description of release"
    git push
    ```

 1. Run `uv build`. It generates .whl and .tar.gz files in the `dist/` folder. Open the tar file in an file explorer to confirm it contains the modules and assets you want included.

 1. Run `uv publish`. You will need an token for PyPi to accept the release.

 1. For larger releases, consider creating a [new Github Release](https://github.com/hello-robot/stretch_mujoco/releases/new) and posting to [the forum](https://forum.hello-robot.com/).

## Using the library

### Getting a specific version

You can get a specific version e.g. "v0.x.y" by adding "hello-robot-stretch-mujoco==0.x.y" to your dependencies (e.g. using `uv add hello-robot-stretch-mujoco==0.x.y` if using UV) or using:

```
pip install hello-robot-stretch-mujoco==0.x.y
```

### Importing the modules

You can import modules from Stretch Mujoco using:

```python
from stretch_mujoco.mujoco_server import MujocoServer, MujocoServerProxies
```

### Accessing the XMLs

The XML models ship with the PyPi package. You can import them using:

```python
import importlib.resources

scene_xml = importlib.resources.files("stretch_mujoco") / "models" / "scene.xml"
server = MujocoServer(scene_xml_path=str(scene_xml), ...)
```

### What won't work

Robocasa won't work since it's not included with the PyPi release. Using the Robocasa assets or the `stretch_mujoco.robocasa_gen` module currently requires cloning the repo locally and pulling in the submodules.
