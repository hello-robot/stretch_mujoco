import stretch_mujoco
import click
import cv2


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
    # display camera feeds
    try:
        while sim.is_running():
            if imagery:
                camera_data = sim.pull_camera_data()
                cv2.imshow("cam_d405_rgb", cv2.cvtColor(camera_data["cam_d405_rgb"], cv2.COLOR_RGB2BGR))
                cv2.imshow("cam_d405_depth", camera_data["cam_d405_depth"])
                cv2.imshow(
                    "cam_d435i_rgb", cv2.cvtColor(camera_data["cam_d435i_rgb"], cv2.COLOR_RGB2BGR)
                )
                cv2.imshow("cam_d435i_depth", camera_data["cam_d435i_depth"])
                cv2.imshow("cam_nav_rgb", cv2.cvtColor(camera_data["cam_nav_rgb"], cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
    except KeyboardInterrupt:
        sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
