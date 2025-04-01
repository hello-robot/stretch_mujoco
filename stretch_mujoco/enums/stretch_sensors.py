from enum import Enum
from functools import cache

import mujoco
import mujoco._structs


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
    
    @staticmethod
    def from_mjmodel(mjmodel: mujoco._structs.MjModel) -> "list[StretchSensors]":
        """Get all the sensors in an mjmodel. We don't have the spec, only the compiled model. We're gonna try to find all the sensors."""
        sensors: set[StretchSensors] = set()
        remaining_sensors = [s for s in StretchSensors]
        try:
            index = 0
            while True:
                # We have no way of pulling the number of sensors via API. 
                # When we exceed the sensors in mjmodel.sensor, an IndexError will be thrown.
                name = mjmodel.sensor(index).name
                index += 1
                for sensor in remaining_sensors:
                    # base_lidar is replicated, so it's called base_lidar000 -> base_lidar359 in this list, this is why we're using `sensor.name in name` below:
                    if sensor.name in name:
                        sensors.add(sensor)
                        remaining_sensors.remove(sensor)

                if len(remaining_sensors) == 0:
                    break

        except IndexError: ...

        return list(sensors)

