import threading
import time
from typing import TYPE_CHECKING
import numpy as np

from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.datamodels.status_stretch_sensors import StatusStretchSensors
from stretch_mujoco.utils import FpsCounter

if TYPE_CHECKING:
    from stretch_mujoco.mujoco_server import MujocoServer


class MujocoServerSensorManagerSync:
    """
    Handles rendering scene sensors to a buffer.

    Call `pull_sensor_data_at_sensor_rate()` from the UI thread and the sensors will be rendered at the specified `sensor_hz`.
    """

    def __init__(
        self, sensor_hz: float, sensors_to_use: list[StretchSensors], mujoco_server: "MujocoServer"
    ) -> None:

        self.mujoco_server = mujoco_server

        self.sensor_rate = 1 / sensor_hz  # Hz to seconds

        self.sensors_to_use = sensors_to_use

        self.sensor_fps_counter = FpsCounter()

        self.time_start = time.perf_counter()

        self.sensor_lock = threading.Lock()

    def is_ready_to_pull_sensor_data(self, is_sleep_until_ready: bool = False):
        """
        Checks to see if a duration of time has passed since the last call
        to this function to render sensor at the specified `self.sensor_rate`.
        """
        elapsed = time.perf_counter() - self.time_start
        if elapsed < self.sensor_rate:
            # If we're not ready to render sensor, don't render:
            if not is_sleep_until_ready:
                return False
            # sleep until ready:
            time.sleep(self.sensor_rate - elapsed)

        self.time_start = time.perf_counter()
        return True

    def pull_sensor_data_at_sensor_rate(self, is_sleep_until_ready: bool):
        """
        Call this on the UI thread to render sensor data.
        """

        if not self.is_ready_to_pull_sensor_data(is_sleep_until_ready):
            return

        self._pull_sensor_data()

        self.sensor_fps_counter.tick()

    def _pull_sensor_data(self):
        """
        Pull data from the simulator.
        """
        sensor_status = StatusStretchSensors.default()
        sensor_status.time = self.mujoco_server.mjdata.time
        sensor_status.fps = self.sensor_fps_counter.fps

        for sensor in self.sensors_to_use:
            data: np.ndarray
            if sensor == StretchSensors.base_lidar:
                # Note: the resolution here should match the resolution defined in stretch.xml. TODO: dynamically pull this value from the MJCF.
                data = np.array(
                    [
                        data
                        for lidar_name in StretchSensors.lidar_names(resolution=360)
                        for data in self.mujoco_server.mjdata.sensor(lidar_name).data
                    ]
                )
            else:
                data = self.mujoco_server.mjdata.sensor(sensor.name).data

            sensor_status.set_data(sensor, data)

        self.mujoco_server.data_proxies.set_sensors(sensor_status)


class MujocoServerSensorManagerThreaded(MujocoServerSensorManagerSync):
    """
    Starts a sensor loop on init to pull sensor data using threading.
    """

    def __init__(
        self,
        sensor_hz: float,
        sensors_to_use: list[StretchSensors],
        mujoco_server: "MujocoServer",
    ):
        """
        `use_threadpool_executor` will use a ThreadPoolExecutor to render all sensors. Setting to false will render each one synchronously.

        `use_sensor_thread` can be set to false to use the ThreadPoolExecutor without the sensor thread. `pull_sensor_data_at_sensor_rate()` must be called on the UI thread if this mode is used.
        """

        super().__init__(sensor_hz, sensors_to_use, mujoco_server)

        self.sensors_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        self.sensors_thread.start()

    def _sensor_loop(self):
        """
        This is the thread loop that handles sensor rendering.
        """

        while (
            self.mujoco_server.data_proxies.get_status().time == 0
        ) and not self.mujoco_server._is_requested_to_stop():
            # wait for sim to start
            time.sleep(0.1)

        while not self.mujoco_server._is_requested_to_stop():

            if not self.is_ready_to_pull_sensor_data(is_sleep_until_ready=True):
                return

            self._pull_sensor_data()

            self.sensor_fps_counter.tick()
