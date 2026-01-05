from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator
from scipy.spatial.transform import Rotation as R



if __name__ == "__main__":

    translation_m = [0.1, -0.2, 0]

    # Convert euler angles to quaternion
    euler_angles_degrees = [0, 0, -90] # Example: roll, pitch, yaw (x, y, z)
    rotation_obj = R.from_euler('xyz', euler_angles_degrees, degrees=True)
    rotation_quat = rotation_obj.as_quat(scalar_first=True)


    sim = StretchMujocoSimulator(
        start_translation=translation_m,
        start_rotation_quat=rotation_quat
    )

    sim.start(headless=False)

    while sim.is_running(): ...