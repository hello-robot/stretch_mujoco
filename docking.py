from stretch_mujoco import StretchMujocoSimulator
import time
import math
import numpy as np


def inverse_3x3_matrix(matrix):
    if matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix.")
    determinant = np.linalg.det(matrix)
    if determinant == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    return np.linalg.inv(matrix)

def rotation_3x3_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta),  0],
                     [0            , 0,              1]])

# ---------- 1.  global design parameters (tune only these) ----------
k_rho    = 1.5        # > 0
k_alpha  = 4.0        # > k_rho  (makes heading converge faster than position)
k_delta  = 1.0        # > 0
k_theta  = 1.0        # Worst case, π rad/s rotation
v_max    = 0.6        # [m s⁻¹]    your motor limits
omega_max = 2.5       # [rad s⁻¹]  your motor limits

RHO_STOP     = 0.005   # [m] = 0.5cm
THETA_STOP   = math.radians(0.5)  # [rad] = 0.5°

# ---------- 2.  helper functions ----------
def wrap(angle: float) -> float:
    """Wrap any angle to (–π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def saturate(x: float, lo: float, hi: float) -> float:
    """Clip x to the closed interval [lo, hi]."""
    return max(lo, min(x, hi))

# ---------- 3.  the controller itself ----------
def polar_controller(x_g, y_g, th_g):
    rho   = math.hypot(x_g, y_g)

    # ---------- stage 1 : drive to point ----------
    if rho > RHO_STOP:
        alpha = wrap(math.atan2(y_g, x_g))
        delta = wrap(alpha - th_g)

        v     = k_rho   * rho
        omega = k_alpha * alpha + k_delta * delta

        # clip
        v     = saturate(v,    -v_max,    v_max)
        omega = saturate(omega, -omega_max, omega_max)

    # ---------- stage 2 : spin in place to heading ----------
    else:
        v     = 0.0
        omega = k_theta * wrap(th_g)           # simple P term
        omega = saturate(omega, -omega_max, omega_max)

        # done?
        if abs(wrap(th_g)) < THETA_STOP:
            omega = 0.0

    return v, omega


if __name__ == "__main__":
    sim = StretchMujocoSimulator()
    sim.start()

    print("Start!")
    sim.add_world_frame((0, 0, 0))
    goalx, goaly, goalt = (0.3, 0.3, 0.4)
    sim.add_world_frame((goalx, goaly, 0.0))
    for _ in range(5000):
        # get current pose
        b = sim.pull_status().base
        currx, curry, currt = (b.x, b.y, b.theta)
        print(f"Current: ({currx:.3f}, {curry:.3f}, {currt:.3f})")

        # compute relative goal
        errx, erry, errt = inverse_3x3_matrix(rotation_3x3_matrix(currt)) @ np.array([goalx-currx, goaly-curry, wrap(goalt-currt)])
        print(f"Delta: ({errx:.3f}, {erry:.3f}, {errt:.3f})")

        # back out errpose in world frame
        Sb = rotation_3x3_matrix(currt) @ np.array([errx, erry, errt])
        errx_wrt_world = currx + Sb[0]
        erry_wrt_world = curry + Sb[1]
        errt_wrt_world = currt + Sb[2]
        print(f"Delta wrt World: ({errx_wrt_world:.3f}, {erry_wrt_world:.3f}, {errt_wrt_world:.3f})")

        # apply controller
        v, w = polar_controller(errx, erry, errt)
        print(f"Cmd: ({v:.4f}, {w:.4f})")
        sim.set_base_velocity(v, w)
        time.sleep(0.01)

    from pprint import pprint
    pprint(sim.pull_status())
    print('Done!')
    sim.stop()

