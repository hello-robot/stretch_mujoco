import click
import cv2

from examples.camera_feeds import show_camera_feeds_sync
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.robocasa_gen import model_generation_wizard


@click.command()
@click.option("--task", type=str, default="PnPCounterToCab", help="task")
@click.option("--layout", type=int, default=None, help="kitchen layout (choose number 0-9)")
@click.option("--style", type=int, default=None, help="kitchen style (choose number 0-11)")
@click.option("--write-to-file", type=str, default=None, help="write to file")
def main(task: str, layout: int, style: int, write_to_file):

    # You can use all the camera's, but it takes longer to render, and may affect the overall simulation FPS.
    # cameras_to_use = StretchCameras.all()
    cameras_to_use = [StretchCameras.cam_d405_rgb]
    
    model, xml, objects_info = model_generation_wizard(
        task=task,
        layout=layout,
        style=style,
        write_to_file=write_to_file,
    )
    sim = StretchMujocoSimulator(model=model, cameras_to_use=cameras_to_use)
    sim.start()
    # display camera feeds
    try:
        while sim.is_running():
            show_camera_feeds_sync(sim, True)
    except KeyboardInterrupt:
        sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
