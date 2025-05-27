from stretch_mujoco import StretchMujocoSimulator
import time
import math

# ---------- 1.  global design parameters (tune only these) ----------
kappa0   = 0.1          # main heading gain
kappa2   = 0.25         # main distance gain
epsilon  = 2.25         # ϵ  (called “varepsilon” in the paper, must be > 1)
xi       = 1e-3         # ξ  (small positive number)
delta    = 25.0         # δ  (tube size)
v_max    = 0.8          # [m/s]  linear-velocity limit
w_max    = 1.5          # [rad/s] angular-velocity limit
tiny     = 1e-6         # numerical zero

# ---------- 2.  constants derived once from the above ----------
gamma = kappa0*epsilon + xi
zeta  = 2*gamma + kappa0                       # ζ
kappa1 = (2*gamma + math.sqrt(kappa0**2 + 6*kappa0*zeta + zeta**2 + 1)) / 4

def riccati_solution():
    """Closed-form formulas from Alg. 1, step 2 (faster than an eigen-solver)."""
    P3 = (kappa0 + 3*zeta + math.sqrt(kappa0**2 + 6*kappa0*zeta + zeta**2)) / 4
    P2 = P3**2 - (kappa0 + zeta)*P3/2
    P1 = 2*P2**2 / zeta
    return P1, P2, P3            # order matches (19)

P1, P2, P3 = riccati_solution()

# ---------- 3.  helper functions ----------
def V(z0, z1, z2):
    """Lyapunov-like function (eq. 19)."""
    return P1*z1**2 - 2*P2*kappa0*z0*z1*z2 + P3*(kappa0*z0*z2)**2

def saturate(value, limit):
    """Hard saturation symmetric in ±limit."""
    return max(-limit, min(limit, value))

TWOPI = 2.0 * math.pi
def wrap(angle_rad: float) -> float:
    """Wrap any real angle to the interval (-π, π]."""
    return (angle_rad + math.pi) % TWOPI - math.pi

# ---------- 4.  the controller itself ----------
def parking_controller(xe, ye, the_e):
    """
    Inputs are the relative pose error already expressed in the robot frame
      xe, ye      [m]
      the_e (=θe) [rad]

    Returns (v, w) to feed the diff-drive low-level controller.
    """
    # --- build z-coordinates (eq. 8) -----------------
    z0 = -the_e
    z1 =  ye
    z2 = -xe

    # ---------- u0 (angular part) ----------
    if abs(z1) < tiny and abs(z2) < tiny:
        u0 = -math.copysign(abs(z0)**(1/3), z0)             # finite-time term
    elif V(z0, z1, z2) < delta * (kappa0*z0)**2 / epsilon:
        u0 = -kappa0*z0                                     # exponential term
    else:
        psi = z2 if abs(z2) > tiny else math.copysign(1.0, z0*z1)
        u0 = -kappa1 * z1 / psi                             # switching term

    # ---------- u1 (translational part) ----------
    if abs(z0) < tiny and abs(z1) < tiny:
        u1 = -math.copysign(abs(z2)**(1/3), z2)
    elif abs(u0) < tiny:
        u1 = -kappa2*z2
    else:
        u1 = -(P2/u0)*z1 - P3*z2

    # ---------- back-out Robot-level commands ----------
    v = u1 + z1*u0        # assume depth Z≈1
    w = u0

    # ---------- saturation ----------
    v = saturate(v, v_max)
    w = saturate(w, w_max)
    return v, w



if __name__ == "__main__":
    sim = StretchMujocoSimulator()
    sim.start()

    print("Loop!")
    goalx, goaly, goalt = (0.2, 0.2, 0.0)
    for _ in range(10000):
        b = sim.pull_status().base
        currx, curry, currt = (b.x, b.y, b.theta)
        print(f"Current: ({currx:.3f}, {curry:.3f}, {currt:.3f})")
        errx, erry, errt = (goalx-currx, goaly-curry, wrap(goalt-currt))
        print(f"Delta: ({errx:.3f}, {erry:.3f}, {errt:.3f})")
        v, w = parking_controller(errx, erry, errt)
        print(f"Cmd: ({v:.4f}, {w:.4f})")
        sim.set_base_velocity(v, w)
        time.sleep(0.01)

    from pprint import pprint
    pprint(sim.pull_status())
    print('Done!')
    sim.stop()

