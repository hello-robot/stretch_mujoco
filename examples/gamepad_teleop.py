import threading
import time

import click
import cv2
from examples.camera_feeds import show_camera_feeds_sync
from gamepad_controller import GamePadController

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.actuators import Actuators

sim:StretchMujocoSimulator
gamepad:GamePadController

button_mapping = {
    "top_pad_pressed": ["wrist_pitch", 1, 0.05],
    "bottom_pad_pressed": ["wrist_pitch", -1, 0.05],
    "left_pad_pressed": ["wrist_roll", -1, 0.07],
    "right_pad_pressed": ["wrist_roll", 1, 0.07],
    "right_shoulder_button_pressed": ["wrist_yaw", -1, 0.2],
    "left_shoulder_button_pressed": ["wrist_yaw", 1, 0.2],
    "top_button_pressed": ["stow", 0],
    "left_button_pressed": ["dex_switch", 0],
    "right_button_pressed": ["gripper", 1, 0.1],
    "bottom_button_pressed": ["gripper", -1, 0.1],
}

stick_mapping = {
    "right_stick_x": ("arm", "inc", 0.1),
    "right_stick_y": ("lift", "inc", 0.3),
    "left_stick_x": ("turn", "scale", 1),
    "left_stick_y": ("forward", "scale", 1),
}


def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def gamepad_loop():
    global sim, gamepad
    dex_switch = False
    status = sim.pull_status()
    gripper_val = status.gripper.pos
    while True:
        time.sleep(1 / 15)
        gamepad_state = gamepad.get_state()
        for button in button_mapping.keys():
            if gamepad_state[button]:
                try:

                    actuator_name, dir, k = button_mapping[button]
                    if actuator_name != "gripper":
                        pos = status[actuator_name]["pos"] + dir * k
                        print(f"Moving {actuator_name} to {pos}")
                        sim.move_to(actuator_name, pos)
                    elif actuator_name == "gripper":
                        gripper_val = gripper_val + dir * k
                        gripper_val = max(min(gripper_val, 0.56), -0.376)
                        print(f"Moving {actuator_name} to {gripper_val}")
                        sim.move_to(Actuators[actuator_name], gripper_val)

                except Exception:
                    pass
        if gamepad_state["left_button_pressed"]:
            dex_switch = not dex_switch
            print(f"Setting dex_switch to {dex_switch}")
            if dex_switch:
                button_mapping["bottom_pad_pressed"][0] = "head_tilt"
                button_mapping["top_pad_pressed"][0] = "head_tilt"
                button_mapping["left_pad_pressed"][0] = "head_pan"
                button_mapping["right_pad_pressed"][0] = "head_pan"
                button_mapping["left_pad_pressed"][1] = -1 * button_mapping["left_pad_pressed"][1]
                button_mapping["right_pad_pressed"][1] = -1 * button_mapping["right_pad_pressed"][1]
            else:
                button_mapping["top_pad_pressed"][0] = "wrist_pitch"
                button_mapping["bottom_pad_pressed"][0] = "wrist_pitch"
                button_mapping["left_pad_pressed"][0] = "wrist_roll"
                button_mapping["right_pad_pressed"][0] = "wrist_roll"
                button_mapping["left_pad_pressed"][1] = -1 * button_mapping["left_pad_pressed"][1]
                button_mapping["right_pad_pressed"][1] = -1 * button_mapping["right_pad_pressed"][1]
        for stick in stick_mapping.keys():
            if abs(gamepad_state[stick]) > 0.001:
                actuator_name, prop, val = stick_mapping[stick]
                if prop == "inc":
                    pos = status[actuator_name]["pos"] + gamepad_state[stick] * val
                    sim.move_to(actuator_name, pos)
                    print(f"Moving {actuator_name} to {pos}")
        if (
            abs(gamepad_state["left_stick_x"]) < 0.001
            and abs(gamepad_state["left_stick_y"]) < 0.001
        ):
            sim.set_base_velocity(0, 0)
        else:
            v_linear = map_value(gamepad_state["left_stick_y"], -1, 1, -0.3, 0.3)
            omega = -map_value(gamepad_state["left_stick_x"], -1, 1, -2, 2)
            sim.set_base_velocity(v_linear, omega)
            print(f"Setting base velocity to {v_linear, omega}")

        if gamepad_state["top_button_pressed"]:
            sim.stow()


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--robocasa-env", is_flag=True, help="Use robocasa environment")
@click.option("--headless", is_flag=True, help="Run in headless mode")
def main(scene_xml_path: str, robocasa_env: bool, headless: bool):
    global sim, gamepad
    cameras_to_use = StretchCameras.all()
    if robocasa_env:
        from stretch_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard()
        # breakpoint()
        sim = StretchMujocoSimulator(model=model, cameras_to_use=cameras_to_use)
    elif scene_xml_path:
        sim = StretchMujocoSimulator(scene_xml_path=scene_xml_path, cameras_to_use=cameras_to_use)
    else:
        sim = StretchMujocoSimulator(cameras_to_use=cameras_to_use)
    gamepad = GamePadController()
    try:
        sim.start(headless=headless)
        gamepad.start()
        threading.Thread(target=gamepad_loop, daemon=True).start()
        while sim.is_running():
            show_camera_feeds_sync(sim, False)
    except KeyboardInterrupt:
        sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
