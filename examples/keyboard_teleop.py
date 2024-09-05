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
    print("N / M : Open & Close GRIPPER")
    print("Q : Stop")
    click.secho("=====================================", fg="yellow")


def keyboard_control(robot_sim):
    while True:
        print_keyboard_options()
        key = getch().lower()
        if key == "w" and not robot_sim._base_in_pos_motion:
            robot_sim.move_by("base_translate", 0.07)
        elif key == "s" and not robot_sim._base_in_pos_motion:
            robot_sim.move_by("base_translate", -0.07)
        elif key == "a" and not robot_sim._base_in_pos_motion:
            robot_sim.move_by("base_rotate", 0.15)
        elif key == "d" and not robot_sim._base_in_pos_motion:
            robot_sim.move_by("base_rotate", -0.15)
        elif key == "u":
            robot_sim.move_by("lift", 0.1)
        elif key == "j":
            robot_sim.move_by("lift", -0.1)
        elif key == "h":
            robot_sim.move_by("arm", -0.05)
        elif key == "k":
            robot_sim.move_by("arm", 0.05)
        elif key == "n":
            robot_sim.move_by("gripper", 0.007)
        elif key == "m":
            robot_sim.move_by("gripper", -0.007)
        elif key == "q":
            robot_sim.stop()
        time.sleep(0.1)


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--robocasa-env", is_flag=True, help="Use robocasa environment")
def main(scene_xml_path: str, robocasa_env: bool):
    model = None
    if robocasa_env:
        from stretch_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard()
        robot_sim = StretchMujocoSimulator(model=model)
    elif scene_xml_path:
        robot_sim = StretchMujocoSimulator(scene_xml_path=scene_xml_path)
    else:
        robot_sim = StretchMujocoSimulator()
    robot_sim.start()
    keyboard_control(robot_sim)


if __name__ == "__main__":
    main()
