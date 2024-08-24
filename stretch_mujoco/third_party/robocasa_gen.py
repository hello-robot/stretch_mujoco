import argparse
from collections import OrderedDict
from typing import Tuple

import mujoco
import mujoco.viewer
import numpy as np
import robosuite
from robocasa.models.arenas.layout_builder import STYLES
from robosuite import load_controller_config
from termcolor import colored


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
    task: str, layout: int = None, style: int = None
) -> Tuple[mujoco.MjModel, str]:
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

    # Create argument configuration
    config = {
        "env_name": task,
        "robots": "PandaMobile",
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        "translucent_robot": False,
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

    if args.layout is None:
        layout = choose_option(
            layouts, "kitchen layout", default=-1, default_message="random layouts"
        )
    else:
        layout = args.layout

    if style is None:
        style = choose_option(styles, "kitchen style", default=-1, default_message="random styles")
    else:
        style = style

    if layout == -1:
        layout = np.random.choice(range(10))
    if style == -1:
        style = np.random.choice(range(11))

    env.layout_and_style_ids = [[layout, style]]
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
    return model, xml


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PnPCounterToCab", help="task")
    parser.add_argument("--layout", type=int, help="kitchen layout (choose number 0-9)")
    parser.add_argument("--style", type=int, help="kitchen style (choose number 0-11)")
    args = parser.parse_args()
    model, xml = model_generation_wizard(args.task, args.layout, args.style)
    # wrtie to file
    # with open("kitchen.xml", "w") as f:
    #     f.write(xml)
    mujoco.viewer.launch(model)
