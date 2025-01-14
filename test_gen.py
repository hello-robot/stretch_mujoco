import stretch_mujoco
import click

import stretch_mujoco.robocasa_gen


@click.command()
@click.option("--task", type=str, default="PnPCounterToCab", help="task")
@click.option("--layout", type=int, default=None, help="kitchen layout (choose number 0-9)")
@click.option("--style", type=int, default=None, help="kitchen style (choose number 0-11)")
@click.option("--write-to-file", type=str, default=None, help="write to file")
def main(task: str, layout: int, style: int, write_to_file: str):
    model, xml, objects_info = stretch_mujoco.robocasa_gen.model_generation_wizard(
        task=task,
        layout=layout,
        style=style,
        write_to_file=write_to_file,
    )
    robot_sim = stretch_mujoco.StretchMujocoSimulator(model=model)
    robot_sim.start()


if __name__ == "__main__":
    main()
