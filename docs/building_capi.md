
# Building bindings to Mujoco's C API

why? TODO

## Notes

 - https://github.com/google-deepmind/mujoco/issues/2184
 - https://github.com/google-deepmind/mujoco/issues/1778
 - https://github.com/google-deepmind/dm_control/issues/123
 - https://askubuntu.com/questions/1514352/ubuntu-24-04-with-nvidia-driver-libegl-warning-egl-failed-to-create-dri2-scre

## Install

```
sudo apt update
sudo apt install -y libgl1 libosmesa6 libglfw3 libglew-dev patchelf

wget https://github.com/google-deepmind/mujoco/releases/download/3.3.0/mujoco-3.3.0-linux-x86_64.tar.gz
untar mujoco-3.3.0-linux-x86_64.tar.gz # or use tar -xvf

sudo mv mujoco-3.3.0 /opt
sudo chown root:root -R /opt/mujoco-3.3.0/
```

Add the following to your `.bashrc`, then refresh the terminal.
```
export MUJOCO_GL=egl
export MUJOCO_PATH=/opt/mujoco-3.3.0
export LD_LIBRARY_PATH=$MUJOCO_PATH/bin:$LD_LIBRARY_PATH
```

Verify it works w/ `/opt/mujoco/bin/simulate`

## Building

```
cd stretch_mujoco
g++ -std=c++17 -fPIC -shared mujoco_stretch_sim.cpp \
    -o libstretch_mujoco.so \
    -I/opt/mujoco-3.3.0/include \
    -I../third_party \
    -L/opt/mujoco-3.3.0/lib -lmujoco -lGL -lpthread

g++ -std=c++17 test.cpp \
    -o test \
    -I/opt/mujoco-3.3.0/include \
    -L/opt/mujoco-3.3.0/lib -lmujoco -lGL -lpthread

g++ -std=c++17 headless_render.cpp \
    -o headless_render \
    -I/opt/mujoco-3.3.0/include \
    -L/opt/mujoco-3.3.0/lib -lEGL -lGLESv2 -lmujoco -ldl

g++ -std=c++17 -I/opt/mujoco-3.3.0/include -L/opt/mujoco-3.3.0/lib \
    -lmujoco -lGL -lEGL -lGLESv2 -lOSMesa -o mujoco_headless_render \
    mujoco_headless_render.cpp

g++ -o mujoco_headless_render mujoco_headless_render.cpp -I/opt/mujoco-3.3.0/include -L/opt/mujoco-3.3.0/lib -lmujoco -lEGL -lGL
```

next, setup venv

```
cd ..
uv venv
uv pip install -e .
```

then, in python (via `uv run ipython`):

```python
import stretch_mujoco.stretch_mujoco_simulator_efficient as e
sim = e.StretchMujocoSimulatorEfficient()
sim.start()
```
