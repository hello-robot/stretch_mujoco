from collections import OrderedDict
from typing import Tuple

import click
import mujoco
import mujoco.viewer
import numpy as np
import robosuite
from robocasa.models.arenas.layout_builder import STYLES
from robosuite import load_controller_config
from termcolor import colored

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.utils import (
    get_absolute_path_stretch_xml,
    insert_line_after_mujoco_tag,
    replace_xml_tag_value,
    xml_remove_subelement,
    xml_remove_tag_by_name,
)

"""
Modified version of robocasa's kitchen scene generation script
https://github.com/robocasa/robocasa/blob/main/robocasa/demos/demo_kitchen_scenes.py
"""


def choose_option(options, option_name, show_keys=False, default=None, default_message=None):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("{}s:".format(option_name.capitalize()))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except Exception:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


def model_generation_wizard(
    task: str = "PnPCounterToCab",
    layout: int = None,
    style: int = None,
    wrtie_to_file: str = None,
    robot_spawn_pose: dict = None,
) -> Tuple[mujoco.MjModel, str]:
    """
    Wizard to generate a kitchen model for a given task, layout, and style.

    Args:
        task (str): task name
        layout (int): layout id
        style (int): style id
    Returns:
        Tuple[mujoco.MjModel, str]: model and xml string
    """
    layouts = OrderedDict(
        [
            (0, "One wall"),
            (1, "One wall w/ island"),
            (2, "L-shaped"),
            (3, "L-shaped w/ island"),
            (4, "Galley"),
            (5, "U-shaped"),
            (6, "U-shaped w/ island"),
            (7, "G-shaped"),
            (8, "G-shaped (large)"),
            (9, "Wraparound"),
        ]
    )

    styles = OrderedDict()
    for k in sorted(STYLES.keys()):
        styles[k] = STYLES[k].capitalize()
    if layout is None:
        layout = choose_option(
            layouts, "kitchen layout", default=-1, default_message="random layouts"
        )
    else:
        layout = layout

    if style is None:
        style = choose_option(styles, "kitchen style", default=-1, default_message="random styles")
    else:
        style = style

    if layout == -1:
        layout = np.random.choice(range(10))
        print(colored(f"Randomly choosing layout... id: {layout}", "yellow"))
    if style == -1:
        style = np.random.choice(range(11))
        print(colored(f"Randomly choosing style... id: {style}", "yellow"))

    # Create argument configuration
    # TODO: Figure how to get an env without robot arg
    config = {
        "env_name": task,
        "robots": "PandaMobile",
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        "translucent_robot": False,
        "layout_and_style_ids": [[layout, style]],
    }

    print(colored("Initializing environment...", "yellow"))

    env = robosuite.make(
        **config,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    print(
        colored(
            f"Showing configuration:\n    Layout: {layouts[layout]}\n    Style: {styles[style]}",
            "green",
        )
    )
    print()
    print(
        colored(
            "Spawning environment...\n",
            "yellow",
        )
    )
    model = env.sim.model._model
    xml = env.sim.model.get_xml()
    xml, robot_base_fixture_pose = custom_cleanups(xml)

    if robot_spawn_pose is not None:
        robot_base_fixture_pose = robot_spawn_pose

    if wrtie_to_file is not None:
        with open(wrtie_to_file, "w") as f:
            f.write(xml)
        print(colored(f"Model saved to {wrtie_to_file}", "green"))

    # add stretch to kitchen
    xml = add_stretch_to_kitchen(xml, robot_base_fixture_pose)
    model = mujoco.MjModel.from_xml_string(xml)
    return model, xml


def custom_cleanups(xml: str) -> Tuple[str, dict]:
    """
    Custom cleanups to models from robocasa envs to support
    use with stretch_mujoco package.
    """

    # make invisible the red/blue boxes around geom/sites of interests found
    xml = replace_xml_tag_value(xml, "geom", "rgba", "0.5 0 0 0.5", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "geom", "rgba", "0.5 0 0 1", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "site", "rgba", "0.5 0 0 1", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "site", "actuator", "0.3 0.4 1 0.5", "0.3 0.4 1 0")
    # remove subelements
    xml = xml_remove_subelement(xml, "actuator")
    xml = xml_remove_subelement(xml, "sensor")

    # remove option tag element
    # xml = xml_remove_subelement(xml, "option")
    # xml = xml_remove_subelement(xml, "size")

    # remove robot
    xml, remove_robot_attrib = xml_remove_tag_by_name(xml, "body", "robot0_base")

    return xml, remove_robot_attrib


def add_stretch_to_kitchen(xml: str, robot_pose_attrib: dict) -> str:
    """
    Add stretch robot to kitchen xml
    """
    print("Adding stretch to kitchen at pose: {}".format(robot_pose_attrib))
    stretch_xml_absolute = get_absolute_path_stretch_xml(robot_pose_attrib)
    # add Stretch xml
    xml = insert_line_after_mujoco_tag(
        xml,
        f' <include file="{stretch_xml_absolute}"/>',
    )
    return xml


@click.command()
@click.option("--task", type=str, default="PnPCounterToCab", help="task")
@click.option("--layout", type=int, default=None, help="kitchen layout (choose number 0-9)")
@click.option("--style", type=int, default=None, help="kitchen style (choose number 0-11)")
@click.option("--write-to-file", type=str, default=None, help="write to file")
def main(task: str, layout: int, style: int, write_to_file: str):
    model, xml = model_generation_wizard(
        task=task,
        layout=layout,
        style=style,
        wrtie_to_file=write_to_file,
    )
    robot_sim = StretchMujocoSimulator(model=model)
    robot_sim.start()
    # display camera feeds
    # try:
    #     while robot_sim.is_running():
    #         camera_data = robot_sim.pull_camera_data()
    #         cv2.imshow("cam_d405_rgb", camera_data["cam_d405_rgb"])
    #         cv2.imshow("cam_d405_depth", camera_data["cam_d405_depth"])
    #         cv2.imshow("cam_d435i_rgb", camera_data["cam_d435i_rgb"])
    #         cv2.imshow("cam_d435i_depth", camera_data["cam_d435i_depth"])
    #         cv2.imshow("cam_nav_rgb", camera_data["cam_nav_rgb"])
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             cv2.destroyAllWindows()
    #             break
    # except KeyboardInterrupt:
    #     robot_sim.stop()
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    main()