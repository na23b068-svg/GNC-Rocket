"""Linearisation of the 6-DOF dynamics about points on the nominal boost trajectory.

Numeric (the reference; signs cannot drift from gnc.dynamics):
  12-state error vector e = [dp(3), dv(3), dphi(3), domega(3)] with the attitude perturbation as a body-frame
  rotation vector, q = q0 (x) exp(dphi/2), so the unit-norm constraint never enters the Jacobian.
  e_dot is evaluated through gnc.dynamics.derivatives and central-differenced:
      dphi_dot = 2 vec( q0* (x) (q_dot(x) - q_dot0 (x) dq) )   (exact first-order error kinematics)
  dp, dv are then expressed in the nominal body axes, giving two decoupled single-gimbal planes:
      pitch plane  [z_b, w, theta, q]  input delta_p   (states 2, 5, 7, 10 of the body-axes error vector)
      yaw   plane  [y_b, v, psi,   r]  input delta_y   (states 1, 4, 8, 11)

Analytic (insight; must match the numeric one, tested): see `analytic_plane`.
The attitude PID uses the 2nd-order short-period reduction theta'' + a1 theta' + a0 theta = b delta.
"""
import numpy as np

from .frames import quat_conj, quat_mul, quat_to_dcm

PITCH = [2, 5, 7, 10]
YAW = [1, 4, 8, 11]
STATE_NAMES = {"pitch": ["z_b", "w", "theta", "q"], "yaw": ["y_b", "v", "psi", "r"]}


def numeric_jacobian(f, x0, u0, eps_x=1e-6, eps_u=1e-6):
    """Central-difference Jacobians A = df/dx, B = df/du of any f(x, u) -> xdot."""
    x0, u0 = np.asarray(x0, float), np.asarray(u0, float)
    f0 = np.asarray(f(x0, u0))
    A = np.zeros((f0.size, x0.size))
    B = np.zeros((f0.size, u0.size))
    for i in range(x0.size):
        h = eps_x * max(1.0, abs(x0[i]))
        d = np.zeros_like(x0); d[i] = h
        A[:, i] = (np.asarray(f(x0 + d, u0)) - np.asarray(f(x0 - d, u0))) / (2 * h)
    for j in range(u0.size):
        h = eps_u * max(1.0, abs(u0[j]))
        d = np.zeros_like(u0); d[j] = h
        B[:, j] = (np.asarray(f(x0, u0 + d)) - np.asarray(f(x0, u0 - d))) / (2 * h)
    return A, B


def _dq(dphi):
    a = np.linalg.norm(dphi)
    if a < 1e-12:
        return np.concatenate([[1.0], 0.5 * dphi])
    return np.concatenate([[np.cos(a / 2)], np.sin(a / 2) * dphi / a])


def error_state_jacobian(deriv, x0, u0=(0.0, 0.0), eps=1e-6):
    """A (12x12), B (12x2) of the error dynamics about state x0 for xdot = deriv(x, u) (13-state).
    Rows/cols ordered [dp_enu, dv_enu, dphi_b, domega_b]."""
    x0 = np.asarray(x0, float)
    q0 = x0[6:10]
    f0 = np.asarray(deriv(x0, np.asarray(u0, float)))

    def edot(e, u):
        dq = _dq(e[6:9])
        x = np.concatenate([x0[0:6] + e[0:6], quat_mul(q0, dq), x0[10:13] + e[9:12]])
        f = np.asarray(deriv(x, u))
        dqdot = quat_mul(quat_conj(q0), f[6:10] - quat_mul(f0[6:10], dq))
        return np.concatenate([f[0:6] - f0[0:6], 2 * dqdot[1:4], f[10:13] - f0[10:13]])

    return numeric_jacobian(edot, np.zeros(12), np.asarray(u0, float), eps, eps)


def to_body_axes(A, B, x0):
    """Express dp, dv in the (instantaneous) nominal body axes: e_b = T e, T = diag(R0^T, R0^T, I, I).
    The rotating-frame term -[omega0 x] is included."""
    R0 = quat_to_dcm(x0[6:10])
    w = x0[10:13]
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    T = np.eye(12)
    T[0:3, 0:3] = T[3:6, 3:6] = R0.T
    Ab = T @ A @ T.T
    Ab[0:3, 0:3] -= W
    Ab[3:6, 3:6] -= W
    return Ab, T @ B


def plane(Ab, Bb, which="pitch"):
    """Single-gimbal plane model (4x4, 4x1) and its coupling to the rest of the 12-state system
    (Frobenius norm of the off-plane columns entering the plane rows, relative to the in-plane block)."""
    idx, j = (PITCH, 0) if which == "pitch" else (YAW, 1)
    A, B = Ab[np.ix_(idx, idx)], Bb[idx, j:j + 1]
    others = [i for i in range(12) if i not in idx]
    coupling = np.linalg.norm(Ab[np.ix_(idx, others)]) / max(np.linalg.norm(A), 1e-12)
    return A, B, float(coupling)


def short_period(A_plane, B_plane):
    """(a1, a0, b) of theta'' + a1 theta' + a0 theta = b delta from a [pos, vel, angle, rate] plane model.
    a0 is the attitude (alpha) stiffness; the slow w-coupling (A[3,1]) is left to the full-model check."""
    return float(-A_plane[3, 3]), float(-A_plane[3, 2]), float(B_plane[3, 0])


def analytic_plane(rocket, t, x0, env, which="pitch"):
    """Analytic linear model of one plane, [pos, vel, angle, rate], about a nominal state with zero AoA:
        w'     = -(F_x/m) theta - (qbar S CNa / m)(w/V + theta) - (T/m) delta
        theta' = q
        q'     = -(l_cp qbar S CNa / I)(w/V + theta) + (qbar S d^2 cmq / (2 V I) - mdot l_g^2 / I) q - (l_g T/I) delta
    with F_x = T - qbar S Cd, l_cp = x_cp - x_cg (static margin), l_g = x_gimbal - x_cg.
    The yaw plane is the mirror image (sign flips on the angle terms of v' and on v in r')."""
    from .atmosphere import isa
    T, m, xcg, _, I, mdot = rocket._thrust_mass(t)
    V = float(np.linalg.norm(x0[3:6]))
    rho, a_snd = isa(env.elevation_m + x0[2])[2:4]
    qbar = 0.5 * rho * V * V
    cd, cna, xcp = rocket._aero(V / a_snd, T > 0)
    S, d = rocket.S, rocket.d
    N = qbar * S * cna
    Fx = T - qbar * S * cd
    lcp, lg = xcp - xcg, rocket.x_gimbal - xcg
    Mq = 0.25 * rho * V * S * d * d * rocket.cmq / I - mdot * lg * lg / I
    s = 1.0 if which == "pitch" else -1.0      # yaw: angle psi tilts the nose toward +y
    A = np.array([[0, 1, 0, 0],
                  [0, -N / (m * V), -s * (Fx + N) / m, 0],
                  [0, 0, 0, 1],
                  [0, -s * lcp * N / (I * V), -lcp * N / I, Mq]])
    B = np.array([[0], [-s * T / m], [0], [-lg * T / I]])
    return A, B, {"T": T, "m": m, "I": I, "V": V, "qbar": qbar, "mach": V / a_snd, "cna": cna,
                  "static_margin_m": lcp, "l_gimbal_m": lg, "x_cg": xcg, "x_cp": xcp}


def linearize_at(rocket, t, x0, env, u0=(0.0, 0.0)):
    """Full numeric linearisation at (t, x0): returns dict with 12-state body-axes A, B and both planes."""
    from .dynamics import derivatives
    A, B = error_state_jacobian(lambda x, u: derivatives(t, x, u, rocket, env), x0, u0)
    Ab, Bb = to_body_axes(A, B, np.asarray(x0, float))
    out = {"t": float(t), "A12": Ab, "B12": Bb}
    for w in ("pitch", "yaw"):
        Ap, Bp, c = plane(Ab, Bb, w)
        out[w] = {"A": Ap, "B": Bp, "coupling": c, "short_period": short_period(Ap, Bp)}
    return out


def nominal_trajectory(rocket, cfg=None):
    """Passive, noiseless, calm-air boost trajectory (gimbal 0) up to shortly after burnout."""
    from .sim import SimConfig, simulate
    if cfg is None:
        cfg = SimConfig(t_max=rocket.burn_time + 0.5, sensor_noise_scale=0.0, stop_at="apogee")
    return simulate(rocket, cfg, controller=None)


def operating_points(rocket, times=None, traj=None, env=None):
    """Numeric linearisations at several times along the nominal powered trajectory (after rail exit).
    Returns a list of dicts (see linearize_at) augmented with the analytic model and flight condition."""
    from .dynamics import make_env
    traj = nominal_trajectory(rocket) if traj is None else traj
    env = make_env(rocket) if env is None else env
    t_rail = traj["events"].get("rail_exit", 0.3)
    if times is None:
        times = np.linspace(t_rail, 0.95 * rocket.burn_time, 7)
    pts = []
    for t in times:
        x0 = np.array([np.interp(t, traj["t"], traj["x"][:, i]) for i in range(13)])
        x0[6:10] /= np.linalg.norm(x0[6:10])
        lin = linearize_at(rocket, t, x0, env)
        Aa, Ba, cond = analytic_plane(rocket, t, x0, env, "pitch")
        lin.update({"x0": x0, "analytic": {"A": Aa, "B": Ba}, "flight": cond})
        pts.append(lin)
    return pts
