from time import sleep
from pynput import keyboard

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


def keyboard_control_release(key: str|None, sim: StretchMujocoSimulator):
    if key in ("w", "s", "a", "d"):
        sim.set_base_velocity(0, 0)

# Allow multiple key-presses, references https://stackoverflow.com/a/74910695
key_buffer = []

def on_press(key):
    global key_buffer
    if key not in key_buffer and len(key_buffer) < 3:
        key_buffer.append(key)
        print(key_buffer)

def on_release(key, sim: StretchMujocoSimulator):
    global key_buffer
    if(key in key_buffer):
        key_buffer.remove(key)
    if isinstance(key, keyboard.KeyCode):
        keyboard_control_release(key.char, sim)
        
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


        listener = keyboard.Listener(
            on_press=on_press,
            on_release=lambda key: on_release(key, sim)
        )

        listener.start()

        while sim.is_running():
            for key in key_buffer:
                if isinstance(key, keyboard.KeyCode):
                    keyboard_control(key.char, sim)
            sleep(0.05)

        listener.stop()
                    

    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
