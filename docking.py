from stretch_mujoco import StretchMujocoSimulator
import time
import math

# ---------- 1.  global design parameters (tune only these) ----------
k_rho    = 1.5        # > 0
k_alpha  = 4.0        # > k_rho  (makes heading converge faster than position)
k_delta  = 1.0        # > 0
k_theta  = 1.0        # Worst case, π rad/s rotation
v_max    = 0.6        # [m s⁻¹]    your motor limits
omega_max = 2.5       # [rad s⁻¹]  your motor limits

RHO_STOP     = 0.05   # [m]
THETA_STOP   = 3*math.pi/180  # [rad] = 3°

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
    goalx, goaly, goalt = (0.5, 0.2, 0.0)
    for _ in range(5000):
        # get current pose
        b = sim.pull_status().base
        currx, curry, currt = (b.x, b.y, b.theta)
        print(f"Current: ({currx:.3f}, {curry:.3f}, {currt:.3f})")

        # compute relative goal
        dx   = goalx - currx
        dy   = goaly - curry
        sgn  = math.sin(-currt)   # pre-compute for speed
        cgn  = math.cos(-currt)
        errx =  cgn*dx + sgn*dy
        erry = -sgn*dx + cgn*dy
        errt =  wrap(goalt - currt)
        print(f"Delta: ({errx:.3f}, {erry:.3f}, {errt:.3f})")

        # apply controller
        v, w = polar_controller(errx, erry, errt)
        print(f"Cmd: ({v:.4f}, {w:.4f})")
        sim.set_base_velocity(v, w)
        time.sleep(0.01)

    from pprint import pprint
    pprint(sim.pull_status())
    print('Done!')
    sim.stop()

