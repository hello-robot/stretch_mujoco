import click
import cv2

from stretch_mujoco import StretchMujocoSimulator, default_scene_xml_path


@click.command()
@click.option("--scene-xml-path", default=default_scene_xml_path, help="Path to the scene xml file")
def main(scene_xml_path: str):
    robot_sim = StretchMujocoSimulator()
    robot_sim.start()
    # display camera feeds
    try:
        while robot_sim.is_running():
            camera_data = robot_sim.pull_camera_data()
            cv2.imshow("cam_d405_rgb", camera_data["cam_d405_rgb"])
            cv2.imshow("cam_d405_depth", camera_data["cam_d405_depth"])
            cv2.imshow("cam_d435i_rgb", camera_data["cam_d435i_rgb"])
            cv2.imshow("cam_d435i_depth", camera_data["cam_d435i_depth"])
            cv2.imshow("cam_nav_rgb", camera_data["cam_nav_rgb"])
            # look for keyboard input
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    except KeyboardInterrupt:
        robot_sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
