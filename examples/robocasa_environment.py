import click

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.robocasa_gen import model_generation_wizard


@click.command()
@click.option("--task", type=str, default="PnPCounterToCab", help="task")
@click.option("--layout", type=int, default=None, help="kitchen layout (choose number 0-9)")
@click.option("--style", type=int, default=None, help="kitchen style (choose number 0-11)")
@click.option("--write-to-file", type=str, default=None, help="write to file")
def main(task: str, layout: int, style: int, write_to_file):
    model, xml = model_generation_wizard(
        task=task,
        layout=layout,
        style=style,
        write_to_file=write_to_file,
    )
    robot_sim = StretchMujocoSimulator(model=model)
    robot_sim.start()


if __name__ == "__main__":
    main()
