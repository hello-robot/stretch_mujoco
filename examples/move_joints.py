import random
import numpy as np

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

def lift_sequence():
    LIFT_START_POS = 0.1
    MOVE_LIFT_BY = 0.5

    sim.move_to(Actuators.lift, LIFT_START_POS)
    sim.wait_until_at_setpoint(Actuators.lift)

    start_lift_position = sim.pull_status().lift.pos

    if not np.isclose(start_lift_position, LIFT_START_POS, atol=0.05):
        print(f"The lift did not move to the starting position. Should be at {LIFT_START_POS}, but is at {start_lift_position:.2f} instead.")

    sim.move_by(Actuators.lift, MOVE_LIFT_BY)
    sim.wait_while_is_moving(Actuators.lift)

    current_lift_position = sim.pull_status().lift.pos
    
    if not np.isclose(start_lift_position, current_lift_position, atol=0.05):
        print(f"The lift did not move by the specified amount. Asked to move from {start_lift_position:.4f} by {MOVE_LIFT_BY}, but ended up at {current_lift_position:.4f}. Should be {start_lift_position + MOVE_LIFT_BY :.4f}")


if __name__ == "__main__":

    sim = StretchMujocoSimulator()

    sim.start(headless=False)

    try:
        sim.stow()

        sim.set_base_velocity(v_linear=5.0, omega=30)

        target = 1.1  # m
        while sim.is_running():
            lift_sequence()

            status = sim.pull_status()

            current_position = status.base.x

            if target > 0 and current_position > target:
                target *= -1
                sim.set_base_velocity(v_linear=-5.0, omega=-30)
            elif target < 0 and current_position < target:
                target *= -1
                sim.set_base_velocity(v_linear=5.0, omega=30)

            sim.move_to(Actuators.head_pan, random.random()-0.5)
            sim.move_to(Actuators.head_tilt, random.random()-0.5)
            sim.wait_until_at_setpoint(Actuators.head_pan)
            sim.wait_until_at_setpoint(Actuators.head_tilt)

    except KeyboardInterrupt:
        sim.stop()
