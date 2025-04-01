from pynput import keyboard
import time

import click

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators


def print_keyboard_options():
    click.secho("\n       Keyboard Controls:", fg="yellow")
    click.secho("=====================================", fg="yellow")
    print("W / A /S / D : Move BASE")
    print("U / J / H / K : Move LIFT & ARM")
    print("O / P: Move WRIST YAW")
    print("C / V: Move WRIST PITCH")
    print("T / Y: Move WRIST ROLL")
    print("N / M : Open & Close GRIPPER")
    print("Q : Stop")
    click.secho("=====================================", fg="yellow")


def keyboard_control(key: str|None, sim: StretchMujocoSimulator):
    if key == "w":
        sim.move_by(Actuators.base_translate, 0.07)
    elif key == "s":
        sim.move_by(Actuators.base_translate, -0.07)
    elif key == "a":
        sim.move_by(Actuators.base_rotate, 0.15)
    elif key == "d":
        sim.move_by(Actuators.base_rotate, -0.15)
    elif key == "u":
        sim.move_by(Actuators.lift, 0.1)
    elif key == "j":
        sim.move_by(Actuators.lift, -0.1)
    elif key == "h":
        sim.move_by(Actuators.arm, -0.05)
    elif key == "k":
        sim.move_by(Actuators.arm, 0.05)
    elif key == "o":
        sim.move_by(Actuators.wrist_yaw, 0.2)
    elif key == "p":
        sim.move_by(Actuators.wrist_yaw, -0.2)
    elif key == "c":
        sim.move_by(Actuators.wrist_pitch, 0.2)
    elif key == "v":
        sim.move_by(Actuators.wrist_pitch, -0.2)
    elif key == "t":
        sim.move_by(Actuators.wrist_roll, 0.2)
    elif key == "y":
        sim.move_by(Actuators.wrist_roll, -0.2)
    elif key == "n":
        sim.move_by(Actuators.gripper, 0.07)
    elif key == "m":
        sim.move_by(Actuators.gripper, -0.07)
    elif key == "q":
        sim.stop()
        exit()


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--robocasa-env", is_flag=True, help="Use robocasa environment")
def main(scene_xml_path: str, robocasa_env: bool):
    model = None
    if robocasa_env:
        from stretch_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard()
        sim = StretchMujocoSimulator(model=model)
    elif scene_xml_path:
        sim = StretchMujocoSimulator(scene_xml_path=scene_xml_path)
    else:
        sim = StretchMujocoSimulator()

    try:
        sim.start()

        print_keyboard_options()

        while sim.is_running():
            with keyboard.Events() as events:
                event = events.get(1.0) # Blocks
                if event is not None:
                    key = event.key
                    if isinstance(key, keyboard.KeyCode):
                        keyboard_control(key.char, sim)
                    
                    print_keyboard_options()

    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
