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
    cameras_to_use = StretchCameras.all() if imagery else []
    sim = stretch_mujoco.StretchMujocoSimulator(scene_xml_path,cameras_to_use=cameras_to_use)
    sim.start(headless=headless)
    try:
        while sim.is_running():
            camera_data = sim.pull_camera_data()

            for camera in cameras_to_use:
                cv2.imshow(camera.name, camera_data.get_camera_data(camera))

            cv2.waitKey(10)

    except KeyboardInterrupt:
        sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
