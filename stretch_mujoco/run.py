import sys
import termios
import time
import tty

import click

from stretch_mujoco import StretchMujocoSimulator, default_scene_xml_path


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


def keyboard_control(robot_sim):
    click.secho("\n       Keyboard Controls:", fg="yellow")
    click.secho("=====================================", fg="yellow")
    print("W / A /S / D : Move BASE")
    print("U / J / H / K : Move LIFT & ARM")
    print("N / M : Open & Close GRIPPER")
    print("Q : Stop")
    click.secho("=====================================", fg="yellow")
    while True:
        key = getch().lower()
        if key == "w":
            robot_sim.move_by("base_translate", 0.05)
        elif key == "s":
            robot_sim.move_by("base_translate", -0.05)
        elif key == "a":
            robot_sim.move_by("base_rotate", 0.1)
        elif key == "d":
            robot_sim.move_by("base_rotate", -0.1)
        elif key == "u":
            robot_sim.move_by("lift", 0.01)
        elif key == "j":
            robot_sim.move_by("lift", -0.01)
        elif key == "h":
            robot_sim.move_by("arm", 0.01)
        elif key == "k":
            robot_sim.move_by("arm", -0.01)
        elif key == "n":
            robot_sim.move_by("gripper", 0.01)
        elif key == "m":
            robot_sim.move_by("gripper", -0.01)
        elif key == "q":
            robot_sim.stop()
        time.sleep(0.1)


@click.command()
@click.option("--scene-xml-path", default=default_scene_xml_path, help="Path to the scene xml file")
def main(scene_xml_path: str):
    robot_sim = StretchMujocoSimulator()
    robot_sim.start()
    keyboard_control(robot_sim)


if __name__ == "__main__":
    main()
