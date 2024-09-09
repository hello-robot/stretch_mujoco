import click
import cv2

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.robocasa_gen import model_generation_wizard


@click.command()
@click.option("--task", type=str, default="PnPCounterToCab", help="task")
@click.option("--layout", type=int, default=None, help="kitchen layout (choose number 0-9)")
@click.option("--style", type=int, default=None, help="kitchen style (choose number 0-11)")
@click.option("--write-to-file", type=str, default=None, help="write to file")
def main(task: str, layout: int, style: int, write_to_file):
    model, xml, objects_info = model_generation_wizard(
        task=task,
        layout=layout,
        style=style,
        write_to_file=write_to_file,
    )
    robot_sim = StretchMujocoSimulator(model=model)
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
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    except KeyboardInterrupt:
        robot_sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
