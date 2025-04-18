import stretch_mujoco
import click
import cv2

from stretch_mujoco.enums.stretch_cameras import StretchCameras


@click.command()
@click.option("--scene-xml-path", help="Path to a scene xml file")
@click.option("--headless", is_flag=True, help="Run the simulation headless")
@click.option("--imagery", is_flag=True, help="Show the cameras' imagery")
def main(
    scene_xml_path: str,
    headless: bool,
    imagery: bool
) -> None:
    sim = stretch_mujoco.StretchMujocoSimulator(scene_xml_path)
    sim.start(headless=headless)
    cameras_to_use = StretchCameras.all() if imagery else StretchCameras.none()
    try:
        while sim.is_running():
            if imagery: # display camera feeds
                camera_data = sim.pull_camera_data()

                for camera in cameras_to_use:
                    cv2.imshow(camera.name, camera_data.get_camera_data(camera))

    except KeyboardInterrupt:
        sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
