import numpy as np
from time import sleep
from pynput import keyboard
from pprint import pprint

import click
import cv2
import cv2.aruco as aruco

from examples.laser_scan import show_laser_scan
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors


def draw_detected_arucos(img, detections, border_color=(0, 255, 0)):
    assert type(img) == np.ndarray
    assert len(img.shape) == 2 or len(img.shape) == 3
    if len(img.shape) == 3:
        assert img.shape[2] == 1 or img.shape[2] == 3
    assert type(detections) == dict
    assert type(border_color) == tuple
    assert len(border_color) == 3

    # colors
    corner_color = (border_color[0], border_color[2], border_color[1])
    text_color = (255, 255, 255) # white shows better

    for mid, d in detections.items():
        # draw borders
        corners = d['corners']
        pts = np.moveaxis(corners, 0, 1).astype(int)
        cv2.polylines(img, [pts], True, border_color)

        # draw top left corner
        v1 = (pts[0][0][0] - 3, pts[0][0][1] - 3)
        v2 = (pts[0][0][0] + 3, pts[0][0][1] + 3)
        cv2.rectangle(img, v1, v2, corner_color, lineType=cv2.LINE_AA)

        # draw ID or name
        name = d['name'] if d['name'] != None else f"id={mid}"
        ptsf = np.moveaxis(corners, 0, 1)
        p1 = np.average(ptsf.reshape(ptsf.shape[0], -1), axis=0).astype(int)
        cv2.putText(img, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness=2)

    return img


class ArucoDetector:

    def __init__(self, dictionary=aruco.DICT_6X6_250, use_apriltag_refinement=False, brighten_images=False):
        # OpenCV's Aruco detection parameters are documented here:
        # https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
        aruco_detection_parameters =  aruco.DetectorParameters()
        aruco_detection_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        if use_apriltag_refinement:
            aruco_detection_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        # aruco_detection_parameters.cornerRefinementWinSize = 2
        # TODO: Investigate Aruco3
        # aruco_detection_parameters.useAruco3Detection = True
        self._detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(dictionary), aruco_detection_parameters)

        # Equalize the gray scale image to improve ArUco marker
        # detection in low exposure time images. Low exposure reduces
        # motion blur, which interferes with ArUco detecction.
        self._adaptive_equalization = None
        if brighten_images:
            self._adaptive_equalization = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        self._known_markers = {
            130: {
                'length_mm': 47.0,
                'use_rgb_only': False,
                'name': 'base_left',
                'link': 'link_aruco_left_base',
            },
            131: {
                'length_mm': 47.0,
                'use_rgb_only': False,
                'name': 'base_right',
                'link': 'link_aruco_right_base',
            },
            132: {
                'length_mm': 23.5,
                'use_rgb_only': False,
                'name': 'wrist_inside',
                'link': 'link_aruco_inner_wrist',
            },
            133: {
                'length_mm': 23.5,
                'use_rgb_only': False,
                'name': 'wrist_top',
                'link': 'link_aruco_top_wrist',
            },
            134: {
                'length_mm': 31.4,
                'use_rgb_only': False,
                'name': 'shoulder_top',
                'link': 'link_aruco_shoulder',
            },
            135: {
                'length_mm': 31.4,
                'use_rgb_only': False,
                'name': 'd405_back',
                'link': 'link_aruco_d405',
            },
            200: {
                'length_mm': 14.0,
                'use_rgb_only': True,
                'name': 'finger_left',
                'link': 'link_finger_left',
            },
            201: {
                'length_mm': 14.0,
                'use_rgb_only': True,
                'name': 'finger_right',
                'link': 'link_finger_right',
            },
            202: {
                'length_mm': 30.0,
                'use_rgb_only': True,
                'name': 'toy',
                'link': 'None',
            },
            245: {
                'length_mm': 88.0,
                'use_rgb_only': False,
                'name': 'docking_station',
                'link': None,
            },
        }

    def detect(self, rgb_image):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        if self._adaptive_equalization is not None:
            gray_image = self._adaptive_equalization.apply(gray_image)

        corners, ids, _ = self._detector.detectMarkers(gray_image)

        # return as dict
        detections = {}
        if ids is not None:
            for i, mid in enumerate(ids.reshape(ids.shape[0])):
                detections[int(mid)] = {
                    'name': self._known_markers[mid]['name'] if mid in self._known_markers else None,
                    'link': self._known_markers[mid]['link'] if mid in self._known_markers else None,
                    'corners': corners[i],
                }

        return detections

detector = ArucoDetector()

def show_camera_feeds_sync(
    sim: StretchMujocoSimulator, 
    print_fps: bool
):
    """
    Pull camera data from the simulator and display it using OpenCV.

    Call this after calling StretchMujocoSimulator::start().
    """

    camera_data = sim.pull_camera_data()

    for camera_name, pixels in camera_data.get_all(use_depth_color_map=True).items():
        detections = detector.detect(pixels)
        cv2.imshow(camera_name.name, draw_detected_arucos(pixels, detections))

    cv2.waitKey(1)

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

            if use_imagery:
                show_camera_feeds_sync(sim, print_ratio)

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
