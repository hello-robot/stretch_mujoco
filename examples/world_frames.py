from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


if __name__ == "__main__":

    sim = StretchMujocoSimulator()

    sim.start(headless=False)

    sim.add_world_frame((0.1,0,0  ), (0,0,0))
    sim.add_world_frame((0.2,0,0), (1.57,0,0)) # (x, y, z), (r, p, y)

    while sim.is_running(): ...