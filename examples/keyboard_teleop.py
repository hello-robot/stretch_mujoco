import sys
import termios
import time
import tty

import click

from stretch_mujoco import StretchMujocoSimulator


def getch():
    """
    Get a single character from the terminal
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


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


def keyboard_control(sim):
    while True:
        print_keyboard_options()
        key = getch().lower()
        if key == "w":
            sim.move_by("base_translate", 0.07)
        elif key == "s":
            sim.move_by("base_translate", -0.07)
        elif key == "a":
            sim.move_by("base_rotate", 0.15)
        elif key == "d":
            sim.move_by("base_rotate", -0.15)
        elif key == "u":
            sim.move_by("lift", 0.1)
        elif key == "j":
            sim.move_by("lift", -0.1)
        elif key == "h":
            sim.move_by("arm", -0.05)
        elif key == "k":
            sim.move_by("arm", 0.05)
        elif key == "o":
            sim.move_by("wrist_yaw", 0.2)
        elif key == "p":
            sim.move_by("wrist_yaw", -0.2)
        elif key == "c":
            sim.move_by("wrist_pitch", 0.2)
        elif key == "v":
            sim.move_by("wrist_pitch", -0.2)
        elif key == "t":
            sim.move_by("wrist_roll", 0.2)
        elif key == "y":
            sim.move_by("wrist_roll", -0.2)
        elif key == "n":
            sim.move_by("gripper", 0.07)
        elif key == "m":
            sim.move_by("gripper", -0.07)
        elif key == "q":
            sim.stop()
            exit()
        time.sleep(0.1)


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--robocasa-env", is_flag=True, help="Use robocasa environment")
def main(scene_xml_path: str, robocasa_env: bool):
    model = None
    if robocasa_env:
        from stretch_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard()
        sim = StretchMujocoSimulator(model=model, camera_hz="off")
    elif scene_xml_path:
        sim = StretchMujocoSimulator(scene_xml_path=scene_xml_path, camera_hz="off")
    else:
        sim = StretchMujocoSimulator(camera_hz="off")
    try:
        sim.start()
        keyboard_control(sim)
    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
