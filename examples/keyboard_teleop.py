from time import sleep
from pynput import keyboard
from pprint import pprint

import click

from examples.camera_feeds import show_camera_feeds_sync
from examples.laser_scan import show_laser_scan
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors


def print_keyboard_options():
    click.secho("\n       Keyboard Controls:", fg="yellow")
    click.secho("=====================================", fg="yellow")
    print("W / A / S / D: Move BASE")
    print("T / F / G / H: Move HEAD")
    print("I / J / K / L: Move LIFT & ARM")
    print("O / P: Move WRIST YAW")
    print("C / V: Move WRIST PITCH")
    print("E / R: Move WRIST ROLL")
    print("N / M: Open & Close GRIPPER")
    print("Z : Print status")
    print("Q : Stop")
    click.secho("=====================================", fg="yellow")


def keyboard_control(key: str | None, sim: StretchMujocoSimulator):
    # mobile base
    if key == "w":
        sim.move_by(Actuators.base_translate, 0.07)
    elif key == "s":
        sim.move_by(Actuators.base_translate, -0.07)
    elif key == "a":
        sim.move_by(Actuators.base_rotate, 0.15)
    elif key == "d":
        sim.move_by(Actuators.base_rotate, -0.15)

    # head
    elif key == "t":
        sim.move_by(Actuators.head_tilt, 0.2)
    elif key == "f":
        sim.move_by(Actuators.head_pan, 0.2)
    elif key == "g":
        sim.move_by(Actuators.head_tilt, -0.2)
    elif key == "h":
        sim.move_by(Actuators.head_pan, -0.2)

    # arm
    elif key == "i":
        sim.move_by(Actuators.lift, 0.1)
    elif key == "k":
        sim.move_by(Actuators.lift, -0.1)
    elif key == "j":
        sim.move_by(Actuators.arm, -0.05)
    elif key == "l":
        sim.move_by(Actuators.arm, 0.05)

    # wrist
    elif key == "o":
        sim.move_by(Actuators.wrist_yaw, 0.2)
    elif key == "p":
        sim.move_by(Actuators.wrist_yaw, -0.2)
    elif key == "c":
        sim.move_by(Actuators.wrist_pitch, 0.2)
    elif key == "v":
        sim.move_by(Actuators.wrist_pitch, -0.2)
    elif key == "e":
        sim.move_by(Actuators.wrist_roll, 0.2)
    elif key == "r":
        sim.move_by(Actuators.wrist_roll, -0.2)

    # gripper
    elif key == "n":
        sim.move_by(Actuators.gripper, 0.07)
    elif key == "m":
        sim.move_by(Actuators.gripper, -0.07)

    # other
    elif key == "z":
        pprint(sim.pull_status())
    elif key == "q":
        sim.stop()


def keyboard_control_release(key: str | None, sim: StretchMujocoSimulator):
    if key in ("w", "s", "a", "d"):
        sim.set_base_velocity(0, 0)


# Allow multiple key-presses, references https://stackoverflow.com/a/74910695
key_buffer = []


def on_press(key):
    global key_buffer
    if key not in key_buffer and len(key_buffer) < 3:
        key_buffer.append(key)


def on_release(key, sim: StretchMujocoSimulator):
    global key_buffer
    if key in key_buffer:
        key_buffer.remove(key)
    if isinstance(key, keyboard.KeyCode):
        keyboard_control_release(key.char, sim)


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--robocasa-env", is_flag=True, help="Use robocasa environment")
@click.option("--imagery-nav", is_flag=True, help="Show only the Navigation camera")
@click.option("--imagery", is_flag=True, help="Show all the cameras' imagery")
@click.option("--lidar", is_flag=True, help="Show the lidar scan in Matplotlib")
@click.option("--print-ratio", is_flag=True, help="Print the sim-to-real time ratio to the cli.")
def main(scene_xml_path: str|None, robocasa_env: bool, imagery_nav: bool, imagery: bool, lidar:bool, print_ratio:bool):
    cameras_to_use = StretchCameras.all() if imagery else []
    if imagery_nav:
        cameras_to_use = [StretchCameras.cam_nav_rgb]
        imagery = True
    use_imagery = imagery or imagery_nav

    model = None

    if robocasa_env:
        from stretch_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard()

    sim = StretchMujocoSimulator(
        model=model,
        scene_xml_path=scene_xml_path,
        cameras_to_use=cameras_to_use
    )

    try:
        sim.start()

        print_keyboard_options()

        listener = keyboard.Listener(on_press=on_press, on_release=lambda key: on_release(key, sim))

        listener.start()

        while sim.is_running():
            for key in key_buffer:
                if isinstance(key, keyboard.KeyCode):
                    keyboard_control(key.char, sim)

            if not lidar and not use_imagery:
                sleep(0.05)

            if print_ratio:
                print(f"{sim.pull_status().sim_to_real_time_ratio_msg}")
                
            if use_imagery:
                show_camera_feeds_sync(sim, False)

            if lidar:
                sensor_data = sim.pull_sensor_data()

                try:
                    show_laser_scan(scan_data=sensor_data.get_data(StretchSensors.base_lidar))
                except: ...


        listener.stop()

    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
