from enum import Enum
from functools import cache


class StretchSensors(Enum):
    """
    An enum of the sensors available to the simulation.
    """

    base_gyro = 0
    base_accel = 1
    base_lidar = 2

    @staticmethod
    def all() -> list["StretchSensors"]:
        """
        Returns all the available sensors
        """
        return [sensor for sensor in StretchSensors]

    @staticmethod
    def none() -> list["StretchSensors"]:
        """
        Short-hand for not using any sensor.
        """
        return []

    @staticmethod
    @cache
    def lidar_names(resolution: int = 720):
        """
        Mujoco names replicated rangefinders using the base_lidar000 -> base_lidar719 nominclature. We need to poll each one individually.
        """
        num_digits = len(str(resolution))
        return [
            f"{StretchSensors.base_lidar.name}{str(i).zfill(num_digits)}" for i in range(resolution)
        ]
